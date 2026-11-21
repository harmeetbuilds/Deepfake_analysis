// All requests go through /api, which vite.config.js forwards to http://127.0.0.1:8000
const API = "/api";

export async function fetchSpec() {
  const r = await fetch(`${API}/openapi.json`);
  if (!r.ok) throw new TypeError("Server not reachable");
  return r.json();
}

const deref = (spec, p) => p.replace("#/", "").split("/").reduce((o, k) => o && o[k], spec);

// Finds the upload endpoint (multipart POST with a file field) from the FastAPI schema.
function pickEndpoint(spec, file) {
  const list = [];
  for (const [path, item] of Object.entries((spec && spec.paths) || {})) {
    const c = item.post?.requestBody?.content?.["multipart/form-data"];
    if (!c) continue;
    let sch = c.schema;
    if (sch?.$ref) sch = deref(spec, sch.$ref);
    for (const [k, v0] of Object.entries(sch?.properties || {})) {
      const v = v0.$ref ? deref(spec, v0.$ref) : v0;
      if (v.format === "binary" || v.contentMediaType || v.items?.format === "binary" || v.items?.contentMediaType) {
        let s = 0;
        const p = path.toLowerCase();
        if (file.type.startsWith("video") && p.includes("video")) s += 3;
        if (file.type.startsWith("image") && p.includes("image")) s += 3;
        if (/predict|detect|analy|classif/.test(p)) s += 1;
        list.push({ path, field: k, s });
        break;
      }
    }
  }
  list.sort((a, b) => b.s - a.s);
  return list[0] || { path: "/predict", field: "file" };
}

function errText(d, t, code) {
  if (typeof d?.detail === "string") return d.detail;
  if (Array.isArray(d?.detail)) return d.detail.map((x) => x.msg).join(". ");
  return `Server returned ${code}. ${(t || "").slice(0, 160)}`;
}

export async function predict(spec, file) {
  const ep = pickEndpoint(spec, file);
  const fd = new FormData();
  fd.append(ep.field, file);
  const res = await fetch(API + ep.path, { method: "POST", body: fd });
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(errText(data, text, res.status));
  return data;
}

/* ---- read whatever shape the API returns ---- */
function find(o, keys, d = 0) {
  if (!o || typeof o !== "object" || d > 2) return;
  for (const k of keys) if (k in o && o[k] != null && typeof o[k] !== "object") return { k, v: o[k] };
  for (const v of Object.values(o)) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const f = find(v, keys, d + 1);
      if (f) return f;
    }
  }
}

function findArray(o, d = 0) {
  if (!o || typeof o !== "object" || d > 2) return;
  for (const v of Object.values(o)) {
    if (Array.isArray(v) && v.length > 1) {
      const n = v.map((x) =>
        typeof x === "number" ? x : x && typeof x === "object"
          ? find(x, ["fake_probability", "fake_prob", "probability", "score", "prob"])?.v : NaN);
      if (n.every((x) => typeof x === "number" && !isNaN(x))) return n.map((x) => (x > 1 ? x / 100 : x));
    }
    const f = findArray(v, d + 1);
    if (f) return f;
  }
}

const num = (x) => { x = Number(x); return x > 1 && x <= 100 ? x / 100 : x; };

export function normalize(r) {
  const used = new Set();
  const lab = find(r, ["label", "prediction", "verdict", "result", "class", "predicted_class", "is_fake", "is_deepfake"]);
  const pr = find(r, ["fake_probability", "fake_prob", "prob_fake", "p_fake", "fake_score", "probability", "score", "prob"]);
  const cf = find(r, ["confidence"]);
  const thr = find(r, ["threshold"]);
  let fake;
  if (lab) {
    used.add(lab.k);
    if (typeof lab.v === "boolean") fake = lab.v;
    else if (typeof lab.v === "number") fake = lab.v === 1;
    else if (/fake|deepfake|manipulat|synthetic/i.test(lab.v)) fake = true;
    else if (/real|authentic|genuine/i.test(lab.v)) fake = false;
  }
  let p;
  if (pr) {
    used.add(pr.k);
    p = num(pr.v);
    const generic = ["probability", "score", "prob"].includes(pr.k);
    if (generic && fake !== undefined && ((fake && p < 0.5) || (!fake && p > 0.5))) p = fake ? p : 1 - p;
  } else if (cf && fake !== undefined) {
    used.add(cf.k);
    const c = num(cf.v);
    p = fake ? c : 1 - c;
  }
  const threshold = thr ? num(thr.v) : 0.5; 
  if (thr) used.add("threshold");
  if (fake === undefined && p !== undefined) fake = p >= threshold;

  const details = [];
  for (const [k, v] of Object.entries(r || {})) {
    if (used.has(k) || v === null || typeof v === "object" || String(v).length > 60) continue;
    const val = typeof v === "number" ? (Number.isInteger(v) ? v : +v.toFixed(3)) : typeof v === "boolean" ? (v ? "Yes" : "No") : String(v);
    const name = k.replace(/_/g, " ");
    details.push([name.charAt(0).toUpperCase() + name.slice(1), val]);
  }
  return { fake, p: isNaN(p) ? undefined : p, threshold, frames: findArray(r), details, raw: r };
}
