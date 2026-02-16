from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import json
import os
import io
import base64
from core_stub import analyze_image_with_rois

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def decode_any_image(file_bytes: bytes, filename: str):
    # 1) Try OpenCV (normal images)
    npbuf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # 2) Try RAW via rawpy
    ext = os.path.splitext((filename or "").lower())[1]
    raw_exts = {".dng",".nef",".cr2",".cr3",".arw",".rw2",".orf",".raf",".pef",".srw",".3fr"}
    if ext in raw_exts:
        try:
            import rawpy
            with rawpy.imread(io.BytesIO(file_bytes)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("RAW decode error:", e)
            return None

    return None


@app.get("/", response_class=HTMLResponse)
def home():
    # Robust read (handles non-utf8/BOM/encoding issues)
    data = open("static/index.html", "rb").read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


@app.post("/preview")
async def preview(image: UploadFile = File(...)):
    data = await image.read()
    img_bgr = decode_any_image(data, image.filename or "")
    if img_bgr is None:
        return JSONResponse({"error": "Cannot decode image/RAW. rawpy نصب است؟"}, status_code=400)

    ok, jpg = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return JSONResponse({"error": "JPEG encode failed"}, status_code=500)

    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), payload: str = Form(...)):
    data = await image.read()
    img_bgr = decode_any_image(data, image.filename or "")
    if img_bgr is None:
        return JSONResponse(
            {"error": "فرمت تصویر قابل decode نیست. برای RAW باید rawpy نصب باشد."},
            status_code=400,
        )

    try:
        payload_obj = json.loads(payload)
    except Exception:
        return JSONResponse({"error": "payload JSON نامعتبر است."}, status_code=400)

    return analyze_image_with_rois(img_bgr, payload_obj)

# ============================================================
# PATCH: ROI PATCHES endpoint (send only ROI crops, not full image)
# ============================================================
def _decode_patch_obj(obj):
    # obj: {w,h,bgr_b64, points, low_trim, high_trim}
    w = int(obj["w"])
    h = int(obj["h"])
    bgr = base64.b64decode(obj["bgr_b64"])
    arr = np.frombuffer(bgr, dtype=np.uint8).reshape((h, w, 3))
    points = obj.get("points")
    low_trim = float(obj.get("low_trim", 0))
    high_trim = float(obj.get("high_trim", 0))
    return arr, points, low_trim, high_trim

@app.post("/analyze_patches")
async def analyze_patches(payload: dict = Body(...)):
    """
    payload = {
      mode: "FA"/...,
      light: {w,h,bgr_b64, points, low_trim, high_trim},
      dark:  {w,h,bgr_b64, points, low_trim, high_trim},
      teeth: [{w,h,bgr_b64, points, low_trim, high_trim}, ...]
    }
    """
    try:
        mode = payload.get("mode", "fa")
        light_obj = payload.get("light") or {}
        dark_obj  = payload.get("dark")  or {}
        teeth_arr = payload.get("teeth") or []

        light_bgr, light_pts, light_lo, light_hi = _decode_patch_obj(light_obj)
        dark_bgr,  dark_pts,  dark_lo,  dark_hi  = _decode_patch_obj(dark_obj)

        # Build a "payload-like" structure for core logic (but on patches)
        patch_payload = {
            "mode": mode,
            "light": {"points": light_pts, "low_trim": light_lo, "high_trim": light_hi, "img_bgr": light_bgr},
            "dark":  {"points": dark_pts,  "low_trim": dark_lo,  "high_trim": dark_hi,  "img_bgr": dark_bgr},
            "teeth": []
        }

        for t in teeth_arr:
            tbgr, tpts, tlo, thi = _decode_patch_obj(t)
            patch_payload["teeth"].append({"points": tpts, "low_trim": tlo, "high_trim": thi, "img_bgr": tbgr})

        # Use new helper in core_stub
        from core_stub import analyze_patches_with_rois
        return analyze_patches_with_rois(patch_payload)

    except Exception as e:
        return JSONResponse({"error": f"analyze_patches failed: {str(e)}"}, status_code=400)

