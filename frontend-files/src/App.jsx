import { useEffect, useRef, useState } from "react";
import { fetchSpec, predict, normalize } from "./api";
import "./App.css";

function Status({ state }) {
  const text = { checking: "Checking server", ok: "Server connected", bad: "Server not reachable" }[state];
  return (
    <div className={`pill ${state === "checking" ? "" : state}`}>
      <span className="dot" />
      <span>{text}</span>
    </div>
  );
}

function Dropzone({ file, scanning, onFile }) {
  const [over, setOver] = useState(false);
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (!file) return setUrl(null);
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  if (file && url) {
    return (
      <div className={`drop has ${scanning ? "scanning" : ""}`}>
        {file.type.startsWith("video")
          ? <video src={url} controls muted playsInline />
          : <img src={url} alt="Selected file preview" />}
      </div>
    );
  }
  return (
    <label
      className={`drop ${over ? "over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files[0]); }}
    >
      <input type="file" accept="video/*,image/*" onChange={(e) => onFile(e.target.files[0])} />
      <strong>Drop a video or photo here</strong>
      <span>or click to choose a file (MP4, MOV, JPG, PNG)</span>
    </label>
  );
}

function Scale({ p, threshold }) {
  const [pos, setPos] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setPos(p * 100)));
    return () => cancelAnimationFrame(id);
  }, [p]);
  const t = threshold * 100;
  return (
    <>
      <div className="scale">
        <i className="r" style={{ width: `${t}%` }} />
        <i className="f" style={{ width: `${100 - t}%` }} />
        <div className="mark" style={{ left: `${pos}%` }}>
          <b />
          <em>{Math.round(p * 100)}%</em>
        </div>
      </div>
      <div className="ends">
        <span>Real</span>
        <span>Cut-off {threshold.toFixed(2)}</span>
        <span>Fake</span>
      </div>
    </>
  );
}

function Result({ busy, error, result, secs }) {
  if (busy) {
    return (
      <div className="empty">
        <h2>Analyzing</h2>
        <p>Finding the face and scoring each frame. Videos on CPU can take a minute.</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="empty">
        <h2 className="err">Could not get a result</h2>
        <p>{error}</p>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="empty">
        <h2>No result yet</h2>
        <p>Add a file and select Analyze. Short clips of 5 to 10 seconds work best, and the first run can take a little longer.</p>
      </div>
    );
  }
  if (result.fake === undefined) {
    return (
      <div className="empty">
        <h2>Result received</h2>
        <p>The server answered, but no label or probability was found in it.</p>
        <p className="mono">{JSON.stringify(result.raw).slice(0, 300)}</p>
      </div>
    );
  }
  const { fake, p, threshold, frames, details } = result;
  return (
    <>
      <p className="verdict">{fake ? "Likely fake" : "Likely real"}</p>
      {p !== undefined && (
        <>
          <p className="sub">The model rates this a {Math.round(p * 100)}% chance of being manipulated.</p>
          <Scale p={p} threshold={threshold} />
          {Math.abs(p - threshold) < 0.1 && (
            <p className="note">This is close to the cut-off, so treat the result as uncertain. Try a longer clip or a clearer face.</p>
          )}
        </>
      )}
      {frames && (
        <>
          <h3>Frame by frame</h3>
          <div className="frames">
            {frames.slice(0, 120).map((v, i) => (
              <i key={i} className={v >= threshold ? "f" : ""} style={{ height: `${Math.max(6, v * 100)}%` }} title={`${Math.round(v * 100)}% fake`} />
            ))}
          </div>
          <p className="cap">Each bar is one analyzed frame. Taller and pink means more likely fake.</p>
        </>
      )}
      <h3>Details</h3>
      <dl>
        {[...details, ["Time taken", `${secs.toFixed(1)} s`]].slice(0, 9).map(([a, b]) => (
          <div key={a} className="row"><dt>{a}</dt><dd>{b}</dd></div>
        ))}
      </dl>
    </>
  );
}

export default function App() {
  const [spec, setSpec] = useState(null);
  const [status, setStatus] = useState("checking");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [secs, setSecs] = useState(0);
  const pickRef = useRef(null);

  useEffect(() => {
    fetchSpec().then((s) => { setSpec(s); setStatus("ok"); }).catch(() => setStatus("bad"));
  }, []);

  const choose = (f) => { if (f) { setFile(f); setResult(null); setError(""); } };
  const clear = () => { setFile(null); setResult(null); setError(""); };

  async function analyze() {
    if (!file) return;
    setBusy(true); setResult(null); setError("");
    const t0 = performance.now();
    try {
      let s = spec;
      if (!s) { s = await fetchSpec(); setSpec(s); }
      const data = await predict(s, file);
      setStatus("ok");
      setSecs((performance.now() - t0) / 1000);
      setResult(normalize(data || {}));
    } catch (e) {
      if (e instanceof TypeError) {
        setStatus("bad");
        setError("The server did not respond. Start it with: uvicorn backend.main:app --reload --port 8000");
      } else setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const tone = !busy && result && result.fake !== undefined ? (result.fake ? "fake" : "real") : "";

  return (
    <div className="wrap">
      <header>
        <div className="brand">
          <svg width="30" height="30" viewBox="0 0 30 30" fill="none" stroke="#14213D" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
            <path d="M3 10V5a2 2 0 0 1 2-2h5M20 3h5a2 2 0 0 1 2 2v5M27 20v5a2 2 0 0 1-2 2h-5M10 27H5a2 2 0 0 1-2-2v-5" />
            <circle cx="15" cy="13" r="4" />
            <path d="M8 22c1.5-3 4-4 7-4s5.5 1 7 4" />
          </svg>
          Deepfake detector
        </div>
        <Status state={status} />
      </header>

      <main className="grid">
        <section className="panel">
          <h1>Is this face real?</h1>
          <p className="lead">Upload a video or photo with a clearly visible face. The model checks the face itself and gives a probability that it was manipulated.</p>
          <Dropzone file={file} scanning={busy} onFile={choose} />
          <div className="actions">
            <button className="primary" onClick={analyze} disabled={!file || busy}>{busy ? "Analyzing" : "Analyze"}</button>
            {file && <button className="ghost" onClick={clear} disabled={busy}>Choose another file</button>}
            {file && <span className="fname">{file.name}</span>}
          </div>
        </section>

        <section className={`panel result ${tone}`} aria-live="polite" ref={pickRef}>
          <Result busy={busy} error={error} result={result} secs={secs} />
        </section>
      </main>

      <footer>Results are estimates from a small model and can be wrong. Do not use them as proof.</footer>
    </div>
  );
}
