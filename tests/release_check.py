"""The GPU on request (README "Releasing the GPU"), against the running tts_server.

  docker run --rm --network logus2k_network -e SPEECH_ADMIN_SECRET -v "$PWD/tests:/t:ro" \
      --entrypoint python cortex:0.2 /t/release_check.py [http://tts_server:7700]

(any image with python-socketio[asyncio_client] and httpx). Checks: the admin routes refuse a missing or wrong secret;
a release frees the model's memory; a sentence spoken while released is served (audio arrives) and brings the model
back; a renewed release; a release racing three sentences (all served); load; a lapsed lease brings the model back by
itself (~80 s). Leaves the model loaded.
"""
import asyncio, os, sys, time, uuid

import httpx
import socketio

URL = sys.argv[1] if len(sys.argv) > 1 else "http://tts_server:7700"
H = {"X-Admin-Secret": os.environ["SPEECH_ADMIN_SECRET"]}
failures = []


def check(ok, what, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail != "" else ""), flush=True)
    if not ok:
        failures.append(what)


async def speaker():
    """One page-like client: speak(text) -> bytes of audio received for it."""
    cid = uuid.uuid4().hex[:8]
    sio = socketio.AsyncClient(reconnection=False)
    got, done = [], asyncio.Event()
    sio.on("tts_audio_chunk", lambda d: got.append(len(d.get("audio_buffer") or b"")))
    sio.on("tts_response_complete", lambda d: done.set())
    await sio.connect(f"{URL}?type=browser&format=binary&main_client_id={cid}", transports=["websocket"], wait_timeout=10)
    await sio.emit("register_audio_client", {"main_client_id": cid, "connection_type": "browser", "mode": "tts", "format": "binary"})
    await asyncio.sleep(0.5)

    async def speak(text):
        got.clear(); done.clear()
        await sio.emit("tts_configure_client", {"client_id": cid, "voice": "pf_dora", "speed": 1.1})
        await sio.emit("tts_text_chunk", {"chunk": text, "target_client_id": cid, "final": True})
        await asyncio.wait_for(done.wait(), 60)
        return sum(got)
    return sio, speak


async def main():
    a = httpx.AsyncClient(base_url=URL, timeout=120)
    check((await a.get("/admin/state")).status_code == 403, "no secret: refused")
    check((await a.get("/admin/state", headers={"X-Admin-Secret": "wrong"})).status_code == 403, "a wrong secret: refused")
    st = (await a.get("/admin/state", headers=H)).json()
    check(st["state"] == "loaded", "loaded at start", st)
    sio, speak = await speaker()
    n = await speak("Bom dia. Esta é uma frase de teste.")
    check(n > 10000, "a sentence spoken", f"{n} bytes of audio")

    r = (await a.post("/admin/release", headers=H, json={"reason": "check", "minutes": 5, "by": "release_check"})).json()
    check(r["state"] == "released" and r["freed_mib"] > 100, "a release frees the model's memory", f"{r['freed_mib']} MiB, holds {r['gpu_mib']} MiB")
    t = time.time(); n = await speak("Esta frase chega enquanto a voz está libertada."); dt = time.time() - t
    st = (await a.get("/admin/state", headers=H)).json()
    check(n > 10000 and st["state"] == "loaded", "spoken while released: served, the model back", f"{n} bytes in {dt:.2f} s")

    await a.post("/admin/release", headers=H, json={"reason": "check", "minutes": 5, "by": "release_check"})
    r2 = (await a.post("/admin/release", headers=H, json={"reason": "again", "minutes": 7, "by": "release_check"})).json()
    check(r2["state"] == "released" and r2["lease"]["seconds_left"] > 400, "a second release renews the lease", r2["lease"])
    await a.post("/admin/load", headers=H)

    others = [await speaker() for _ in range(3)]
    res, rel = await asyncio.gather(
        asyncio.gather(*[sp("Uma de três frases ao mesmo tempo que a libertação.") for _, sp in others]),
        a.post("/admin/release", headers=H, json={"reason": "race", "minutes": 5, "by": "release_check"}))
    check(all(x > 10000 for x in res) and rel.json().get("ok"), "a release racing three sentences: all served", f"{res} {rel.json().get('state')}")
    r = (await a.post("/admin/load", headers=H)).json()
    check(r["state"] == "loaded" and r["lease"]["until"] is None and r["active"] == 0, "load: back, the lease closed, nothing in flight", r)

    await a.post("/admin/release", headers=H, json={"reason": "lapse", "minutes": 1, "by": "release_check"})
    for _ in range(12):
        await asyncio.sleep(10)
        st = (await a.get("/admin/state", headers=H)).json()
        if st["state"] == "loaded":
            break
    check(st["state"] == "loaded" and st["lease"]["until"] is None, "a lapsed lease brings the model back by itself", st["state"])
    for s, _ in [(sio, None)] + others:
        await s.disconnect()

asyncio.run(main())
print(f"\n{len(failures)} failed" + (": " + "; ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
