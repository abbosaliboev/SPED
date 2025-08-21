import requests

# ❗ Bu yerda IP manzilni almashtirdik
BASE = "http://172.20.10.3:8000"

def notify_event(event_type="wheelchair", extend_sec=5, conf=0.9, inter="INT-001"):
    try:
        requests.post(
            f"{BASE}/event",
            json={
                "type": event_type,
                "confidence": float(conf),
                "extend_sec": int(extend_sec),
                "intersection_id": inter,
            },
            timeout=0.5,
        )
    except:
        pass

def clear_event():
    try:
        requests.post(f"{BASE}/clear", timeout=0.5)
    except:
        pass