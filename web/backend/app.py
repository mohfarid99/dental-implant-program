# -*- coding: utf-8 -*-
"""
app.py — FastAPI backend for the Dental Implant web app.

Endpoints
---------
  POST   /api/session                      create a working session from a CBCT folder
  GET    /api/session/{sid}                session info (shape, rotation state)
  DELETE /api/session/{sid}                drop session
  GET    /api/session/{sid}/slice/{z}/png  one slice as PNG (base volume)
  POST   /api/session/{sid}/rotate         {angle, axis, side:left|right}
  POST   /api/session/{sid}/reset-rotation restore original volume
  POST   /api/session/{sid}/missing-range  {start, end} set manual range
  POST   /api/session/{sid}/missing-auto   run EfficientNet auto detection
  POST   /api/session/{sid}/run            full pipeline: segment + recommend + apply best
  POST   /api/session/{sid}/apply          apply a different candidate {index}
  GET    /api/session/{sid}/overlay/{z}.png  AB/MC/implant overlay for slice z
  GET    /api/plants                        implant database
  POST   /api/plants                        add row
  PUT    /api/plants/{id}                   update row
  DELETE /api/plants/{id}                   delete row
  static   /                               web frontend
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import time
import uuid

# ---- OpenMP / MKL crash fix (MANDATORY on some platforms, same as desktop app) ----
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage as ndi
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../web/backend
WEB_DIR = os.path.dirname(BASE_DIR)                              # .../web
FRONTEND_DIR = os.path.join(WEB_DIR, "frontend")
PROJECT_DIR = os.path.dirname(WEB_DIR)                           # repo root
CLASSIFICATION_MODEL = os.path.join(PROJECT_DIR, "classification.pt")
SEGMENTATION_MODEL = os.path.join(PROJECT_DIR, "segmentation.pt")
IMPLANTS_CSV = os.path.join(PROJECT_DIR, "implant_database.csv")
WORK_ROOT = os.path.join(WEB_DIR, "work")
os.makedirs(WORK_ROOT, exist_ok=True)
MAX_UPLOAD_MB = 300          # side_volume.npy is tens of MB; keep headroom

app = FastAPI(title="Dental Implant Planning — Web", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------------- implants
def _load_plants():
    rows = []
    if os.path.isfile(IMPLANTS_CSV):
        with open(IMPLANTS_CSV, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("id,"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        rows.append({"id": parts[0], "diameter_mm": float(parts[1]), "height_mm": float(parts[2])})
                    except ValueError:
                        pass
    return rows

def _save_plants(rows):
    with open(IMPLANTS_CSV, "w", encoding="utf-8") as f:
        f.write("ID,Diameter(mm),Height(mm)\n")
        for r in rows:
            f.write(f"{r['id']},{r['diameter_mm']},{r['height_mm']}\n")

# ------------------------------------------------------------------- sessions
class Session:
    def __init__(self, sid, patient_dir):
        self.sid = sid
        self.patient_dir = patient_dir            # .../d_side_volume
        self.lock = threading.RLock()
        self.state = {
            "status": "idle",
            "message": "",
            "progress": 0.0,
            "volume_source": "original",          # original | rotated
            "rotations": [],
            "missing_start": None,
            "missing_end": None,
            "candidates": [],
            "logs": [],
            "no_implant_reason": None,
            "applied": None,
        }
        self._clf_model = None

    # -- volume access
    def original_npy(self):
        return os.path.join(self.patient_dir, "side_volume.npy")

    def rotated_npy(self):
        return os.path.join(self.patient_dir, "rotated_current.npy")

    def current_npy(self):
        p = self.rotated_npy()
        if os.path.isfile(p):
            return p
        return self.original_npy()

    def volume(self):
        p = self.current_npy()
        if not os.path.isfile(p):
            raise HTTPException(404, "side_volume.npy not found in the CBCT folder")
        return np.load(p)

    def slice_png(self, z):
        vol = self.volume()
        if z < 0 or z >= vol.shape[0]:
            raise HTTPException(404, f"slice {z} out of range")
        s = engine.slice_to_uint8(vol[z])
        ok, buf = cv2.imencode(".png", s)
        if not ok:
            raise HTTPException(500, "failed to encode slice")
        return buf.tobytes()
    def state_path(self):
        return os.path.join(WORK_ROOT, f"{self.sid}_state.json")

    def save_state(self, _hold_lock=False):
        """Persist state to disk. If `_hold_lock` is True the caller already
        holds self.lock (avoids re-acquiring a non-reentrant lock)."""
        if _hold_lock:
            snapshot = dict(self.state)
        else:
            with self.lock:
                snapshot = dict(self.state)
        try:
            with open(self.state_path(), "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=1, default=str)
        except Exception:
            pass

    def load_state(self):
        p = self.state_path()
        if not os.path.isfile(p):
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.update(data)
        except Exception:
            pass

SESSIONS = {}

def get_session(sid):
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "unknown session")
    return s

# ------------------------------------------------------------------- models
class PatientIn(BaseModel):
    patient_dir: str        # path to the FOLDER containing d_side_volume

class RotateIn(BaseModel):
    angle: float
    axis: str = "z"         # z | x
    side: str = "left"      # left | right

class RangeIn(BaseModel):
    start: int
    end: int

class ApplyIn(BaseModel):
    index: int

class PlantIn(BaseModel):
    id: str
    diameter_mm: float
    height_mm: float

# ------------------------------------------------------------------- API
_SAFE_NAME = re.compile(r"[\w.\-]+")

def _safe_name(raw):
    base = os.path.basename(raw.replace("\\", "/").replace("%20", " ").strip())
    base = base[:120]
    if not _SAFE_NAME.fullmatch(base) or base in (".", ".."):
        raise HTTPException(400, "invalid file name")
    return base

@app.post("/api/upload")
async def upload_volume(file: UploadFile):
    """Browser upload: pick the `side_volume.npy` file from anywhere on your
    machine — no server path needed. Creates a working session around it."""
    name = _safe_name(file.filename or "side_volume.npy")
    if not name.lower().endswith(".npy"):
        name += ".npy"
    # validate it really is a numpy .npy (magic header check, no full load)
    head = await file.read(6)
    if len(head) < 6 or (head[:4] != b"\x93NUM"):
        raise HTTPException(400, "file is not a valid NumPy .npy array")

    sid = uuid.uuid4().hex[:12]                       # same shape the UI expects
    sid_dir = os.path.join(WORK_ROOT, sid)
    d_side = os.path.join(sid_dir, "d_side_volume")   # Session(patient_dir=d_side)
    os.makedirs(d_side, exist_ok=True)

    dest = os.path.join(d_side, "side_volume.npy")
    size = 0
    try:
        with open(dest, "wb") as out:
            out.write(head)                     # keep the 6 magic bytes we read
            size += len(head)
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB limit")
                out.write(chunk)
    except Exception:
        try:
            os.remove(dest)          # never leave a partial .npy behind
        except OSError:
            pass
        raise
    finally:
        await file.close()
    if size == 0:
        os.remove(dest)
        raise HTTPException(400, "uploaded file is empty")

    s = Session(sid, d_side)
    SESSIONS[sid] = s
    vol = s.volume()
    return {"sid": sid, "patient_dir": d_side,
            "shape": list(vol.shape), "slices": int(vol.shape[0]),
            "source": name, "bytes": size}

@app.post("/api/session")
def create_session(body: PatientIn):
    root = os.path.abspath(body.patient_dir.strip())
    d_side = os.path.join(root, "d_side_volume")
    candidate = d_side if os.path.isdir(d_side) else root
    if not os.path.isfile(os.path.join(candidate, "side_volume.npy")):
        raise HTTPException(
            400,
            "side_volume.npy not found. Point this at the patient folder that "
            "contains d_side_volume/ (generated by the desktop app pipeline)."
        )
    sid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(WORK_ROOT, sid), exist_ok=True)
    s = Session(sid, candidate)
    SESSIONS[sid] = s
    vol = s.volume()
    return {"sid": sid, "patient_dir": s.patient_dir,
            "shape": list(vol.shape), "slices": int(vol.shape[0])}

@app.get("/api/session/{sid}")
def session_info(sid: str):
    s = get_session(sid)
    vol = s.volume()
    out = dict(s.state)
    out["shape"] = list(vol.shape)
    out["slices"] = int(vol.shape[0])
    out["has_rotation"] = os.path.isfile(s.rotated_npy())
    return out

@app.delete("/api/session/{sid}")
def delete_session(sid: str):
    s = SESSIONS.pop(sid, None)
    if s is None:
        raise HTTPException(404, "unknown session")
    s.save_state()
    return {"ok": True}

@app.get("/api/session/{sid}/slice/{z}/png")
def slice_png(sid: str, z: int):
    return Response(content=get_session(sid).slice_png(z), media_type="image/png")

@app.get("/api/session/{sid}/volume-raw")
def volume_raw(sid: str, max_side: int = 96):
    """Whole volume as raw uint8 for the 3D ray-marcher.
    Downscaled so the largest axis is <= max_side (keeps payload ~1 MB)."""
    vol = get_session(sid).volume()
    vol = np.asarray(vol, dtype=np.float32)
    z, y, x = vol.shape
    f = max_side / float(max(z, y, x))
    if f < 1.0:
        vol = ndi.zoom(vol, zoom=(f, f, f), order=1)
    # robust normalization: min .. 99th percentile
    lo = float(vol.min())
    hi = np.percentile(vol, 99)
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    buf = (vol * 255.0).round().astype(np.uint8).tobytes()
    dims = f"{vol.shape[2]},{vol.shape[1]},{vol.shape[0]}"   # w,h,d for Data3DTexture
    return Response(
        content=buf,
        media_type="application/octet-stream",
        headers={"X-Vol-Dims": dims, "Access-Control-Expose-Headers": "X-Vol-Dims"})

@app.post("/api/session/{sid}/rotate")
def rotate(sid: str, body: RotateIn):
    if not -360.0 < body.angle < 360.0:
        raise HTTPException(400, "angle must be between -360 and 360 (exclusive)")
    s = get_session(sid)
    with s.lock:
        try:
            # chain: rotate current (rotated if present, else original)
            base = s.current_npy()
            vol = np.load(base)
            ang = body.angle
            if body.side == "right":
                ang = (360.0 - body.angle) % 360.0
            rotated = engine.rotate_volume(vol, ang, axis=body.axis)
            if rotated.ndim != 3:
                raise HTTPException(400, "rotation produced non-3D output")
            np.save(s.rotated_npy(), rotated)
            # cleanup exported slice folders the desktop app would have made
            for sub in ("tab7_slices",):
                p = os.path.join(s.patient_dir, sub)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            s.state["status"] = "idle"
            s.state["volume_source"] = "rotated"
            s.state["rotations"].append({"axis": body.axis, "side": body.side,
                                         "angle_in": body.angle, "applied": ang})
            zmax = int(rotated.shape[0]) - 1
            s.state["missing_start"] = 0
            s.state["missing_end"] = zmax
            s.save_state()
            return {"ok": True, "shape": list(rotated.shape), "applied_angle": ang}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"rotation failed: {e}")

@app.post("/api/session/{sid}/reset-rotation")
def reset_rotation(sid: str):
    s = get_session(sid)
    with s.lock:
        p = s.rotated_npy()
        if os.path.isfile(p):
            os.remove(p)
        s.state["volume_source"] = "original"
        s.state["rotations"] = []
        vol = s.volume()
        s.state["missing_start"] = 0
        s.state["missing_end"] = int(vol.shape[0]) - 1
        s.save_state()
        return {"ok": True, "shape": list(vol.shape)}

@app.post("/api/session/{sid}/missing-range")
def set_range(sid: str, body: RangeIn):
    s = get_session(sid)
    vol = s.volume()
    Z = int(vol.shape[0])
    if body.start < 0 or body.end < 0 or body.start >= Z or body.end >= Z:
        raise HTTPException(400, f"slices must be within [0, {Z-1}]")
    if body.end < body.start:
        raise HTTPException(400, "end slice must be >= start slice")
    s.state["missing_start"] = body.start
    s.state["missing_end"] = body.end
    s.save_state()
    return {"ok": True, "start": body.start, "end": body.end}

@app.post("/api/session/{sid}/missing-auto")
def missing_auto(sid: str):
    s = get_session(sid)
    if not os.path.isfile(CLASSIFICATION_MODEL):
        raise HTTPException(500, "classification.pt not found in project root")
    with s.lock:
        try:
            vol = s.volume()
            if s._clf_model is None:
                s._clf_model = engine.load_classifier(CLASSIFICATION_MODEL, device="cpu")
            start, end, preds = engine.classify_volume_missing_range(
                vol, s._clf_model, device="cpu",
                progress_cb=lambda p: s.state.update(progress=p, status="classifying"))
        except Exception as e:
            s.state.update(status="error", message=f"auto-detect failed: {e}")
            s.save_state()
            raise HTTPException(500, f"auto-detect failed: {e}")
    if start is None or end is None:
        raise HTTPException(422, "classifier found no missing slices in this volume")
    s.state["missing_start"] = start
    s.state["missing_end"] = end
    s.state.update(status="idle", progress=1.0,
                   message=f"auto-detected missing range [{start}, {end}]")
    s.save_state()
    return {"ok": True, "start": start, "end": end}

@app.post("/api/session/{sid}/run")
def run_pipeline(sid: str):
    """Full pipeline: YOLO segmentation -> recommendation -> apply best."""
    s = get_session(sid)
    start = s.state.get("missing_start")
    end = s.state.get("missing_end")
    if start is None or end is None:
        raise HTTPException(400, "set the missing-teeth slice range first (manual or auto)")
    if not os.path.isfile(SEGMENTATION_MODEL):
        raise HTTPException(500, "segmentation.pt not found in project root")
    s.state.update(status="running", progress=0.05, message="segmenting…")
    s.save_state()

    def work():
        try:
            vol = s.volume()
            Z = int(vol.shape[0])
            s.state.update(progress=0.15, message="YOLO segmentation…")
            s.save_state()
            engine.run_yolo_segmentation(
                vol, int(start), int(end),
                ab_dir=os.path.join(s.patient_dir, "segmented_AB"),
                mc_dir=os.path.join(s.patient_dir, "segmented_MC"),
                model_path=SEGMENTATION_MODEL,
                progress_cb=lambda p: (s.state.update(progress=0.15 + 0.55 * p), s.save_state()))

            plants = _load_plants()
            imps = [(r["id"], r["diameter_mm"], r["height_mm"]) for r in plants]
            s.state.update(progress=0.75, message="computing recommendation…")
            s.save_state()
            result = engine.compute_recommendation(
                os.path.join(s.patient_dir, "segmented_AB"),
                os.path.join(s.patient_dir, "segmented_MC"),
                int(start), int(end), imps)

            s.state["candidates"] = [list(c) for c in result["candidates"]]
            s.state["logs"] = result["logs"]
            s.state["no_implant_reason"] = result["no_implant_reason"]

            if result["candidates"]:
                s.state.update(progress=0.9, message="applying best implant…")
                s.save_state()
                resolution = (int(vol.shape[2]), int(vol.shape[1]))
                chosen = tuple(result["candidates"][0])
                out = engine.apply_implant(
                    s.patient_dir, chosen, int(start), resolution, Z,
                    ab_dir=os.path.join(s.patient_dir, "segmented_AB"),
                    mc_dir=os.path.join(s.patient_dir, "segmented_MC"))
                s.state["applied"] = out["applied"]

            s.state.update(status="done", progress=1.0, message="pipeline complete")
        except Exception as e:
            s.state.update(status="error", message=str(e))
        finally:
            s.save_state()

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "status": "running"}

@app.post("/api/session/{sid}/apply")
def apply_implant(sid: str, body: ApplyIn):
    s = get_session(sid)
    cands = s.state.get("candidates") or []
    if body.index < 0 or body.index >= len(cands):
        raise HTTPException(400, "invalid candidate index")
    with s.lock:
        try:
            vol = s.volume()
            Z = int(vol.shape[0])
            start = int(s.state["missing_start"])
            resolution = (int(vol.shape[2]), int(vol.shape[1]))
            out = engine.apply_implant(
                s.patient_dir, tuple(cands[body.index]), start, resolution, Z,
                ab_dir=os.path.join(s.patient_dir, "segmented_AB"),
                mc_dir=os.path.join(s.patient_dir, "segmented_MC"))
            s.state["applied"] = out["applied"]
            s.save_state()
            return {"ok": True, "applied": out["applied"]}
        except Exception as e:
            raise HTTPException(500, f"apply failed: {e}")

@app.get("/api/session/{sid}/overlay/{z}.png")
def overlay_png(sid: str, z: int):
    s = get_session(sid)
    W, H = None, None
    ab_f = os.path.join(s.patient_dir, "segmented_AB", f"merged_segs_{z}.jpg")
    mc_f = os.path.join(s.patient_dir, "segmented_MC", f"merged_segs_{z}.jpg")
    imp_f = os.path.join(s.patient_dir, "implant_only", f"implant_{z}.png")
    ab = cv2.imread(ab_f, cv2.IMREAD_GRAYSCALE) if os.path.isfile(ab_f) else None
    mc = cv2.imread(mc_f, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mc_f) else None
    imp = cv2.imread(imp_f, cv2.IMREAD_GRAYSCALE) if os.path.isfile(imp_f) else None
    if ab is None and mc is None and imp is None:
        raise HTTPException(404, "no segmentation result for this slice yet")
    if ab is not None:
        H, W = ab.shape
    elif mc is not None:
        H, W = mc.shape
    else:
        H, W = imp.shape
    rgba = engine.build_overlay_slice(ab, mc, imp, (W, H))
    ok, buf = cv2.imencode(".png", rgba)
    return Response(content=buf.tobytes(), media_type="image/png")

# ------------------------------------------------------------------- plants
@app.get("/api/plants")
def list_plants():
    return {"plants": _load_plants()}

@app.post("/api/plants")
def add_plant(body: PlantIn):
    rows = _load_plants()
    if any(r["id"].lower() == body.id.lower() for r in rows):
        raise HTTPException(409, f"implant '{body.id}' already exists")
    if body.diameter_mm <= 0 or body.height_mm <= 0:
        raise HTTPException(400, "diameter and height must be positive")
    rows.append({"id": body.id.strip(), "diameter_mm": float(body.diameter_mm),
                 "height_mm": float(body.height_mm)})
    _save_plants(rows)
    return {"ok": True, "plants": rows}

@app.put("/api/plants/{implant_id}")
def update_plant(implant_id: str, body: PlantIn):
    rows = _load_plants()
    target = next((r for r in rows if r["id"].lower() == implant_id.lower()), None)
    if target is None:
        raise HTTPException(404, f"implant '{implant_id}' not found")
    new_id = body.id.strip()
    if new_id.lower() != implant_id.lower() and any(
            r["id"].lower() == new_id.lower() for r in rows):
        raise HTTPException(409, f"implant '{new_id}' already exists")
    target["id"] = new_id
    target["diameter_mm"] = float(body.diameter_mm)
    target["height_mm"] = float(body.height_mm)
    _save_plants(rows)
    return {"ok": True, "plants": rows}

@app.delete("/api/plants/{implant_id}")
def delete_plant(implant_id: str):
    rows = _load_plants()
    new_rows = [r for r in rows if r["id"].lower() != implant_id.lower()]
    if len(new_rows) == len(rows):
        raise HTTPException(404, f"implant '{implant_id}' not found")
    _save_plants(new_rows)
    return {"ok": True, "plants": new_rows}

# ------------------------------------------------------------------- static
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
