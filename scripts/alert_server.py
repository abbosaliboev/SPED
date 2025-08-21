from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

app = FastAPI()

BASE_DIR = Path(__file__).parent           # scripts/
STATIC_DIR = BASE_DIR / "static"           # scripts/static
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

last_alert = {"active": False, "msg": "✅ Safe. No special pedestrians."}
clients: set[WebSocket] = set()

@app.get("/")
async def root():
    return HTMLResponse("<a href='/static/index.html'>Open Driver Page</a>")

@app.get("/state")
async def state():
    return last_alert

@app.post("/event")
async def event(req: Request):
    data = await req.json()
    t = data.get("type", "priority pedestrian")
    ext = int(data.get("extend_sec", 5))
    msg = f"⚠️ {t} detected near crosswalk. Please wait. (+{ext}s)"
    last_alert.update({"active": True, "msg": msg})

    payload = json.dumps({"kind": "PED_ALERT", "msg": msg})
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except:
            dead.append(ws)
    for d in dead:
        clients.discard(d)
    return {"ok": True}

@app.post("/clear")
async def clear():
    last_alert.update({"active": False, "msg": "✅ Safe. No special pedestrians."})
    payload = json.dumps({"kind": "CLEAR"})
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except:
            dead.append(ws)
    for d in dead:
        clients.discard(d)
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # pings ignored
    except:
        pass
    finally:
        clients.discard(ws)