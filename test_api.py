"""Quick API test:  python test_api.py path/to/file.(jpg|png|mp4)   [--url http://localhost:8000]"""
import json, sys
import requests

url = "http://localhost:8000"
args = [a for a in sys.argv[1:]]
if "--url" in args:
    i = args.index("--url"); url = args[i + 1]; del args[i:i + 2]
if not args:
    sys.exit(__doc__)

print("health:", requests.get(f"{url}/health").json())
print("model :", requests.get(f"{url}/api/model-info").json())
with open(args[0], "rb") as f:
    r = requests.post(f"{url}/api/predict", files={"file": (args[0].split("\\")[-1].split("/")[-1], f)})
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2))
