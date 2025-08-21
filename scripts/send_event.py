import requests, sys

BASE = "http://172.20.10.3:8000"  # sizning IP

if len(sys.argv) > 1 and sys.argv[1] == "clear":
    r = requests.post(f"{BASE}/clear", timeout=2)
else:
    payload = {
        "type": "wheelchair",
        "confidence": 0.9,
        "extend_sec": 5,
        "intersection_id": "INT-001"
    }
    r = requests.post(f"{BASE}/event", json=payload, timeout=2)

print(r.status_code, r.text)