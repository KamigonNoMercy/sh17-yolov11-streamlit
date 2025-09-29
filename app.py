# app.py — SH17 Safety Equipment Detector (Streamlit)
# ---------------------------------------------------
# Jalankan (lokal):  streamlit run app.py

import io
import time
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import torch  # untuk info env

TITLE = "Safety Equipment Detection (YOLOv11 • SH17)"
PAGE_ICON = "🦺"
IMGKW = dict(use_column_width=True)  # kompatibel streamlit lama

# Mapping class id -> nama (12 kelas sesuai model-mu)
ID2NAME = {
    0: "person",
    1: "ear",
    2: "face",
    3: "face-protection",
    4: "tool",
    5: "glasses",
    6: "gloves",
    7: "helmet",
    8: "hands",
    9: "head",
    10: "shoes",
    11: "protective-suit-vest",
}

# ---------- Helpers ----------
def get_device_info(device_str: str) -> str:
    """Label device ringkas untuk caption."""
    if device_str == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    # CPU label (Streamlit Cloud = CPU only)
    return "CPU"

def find_default_weights():
    """Coba cari weights best.pt otomatis di runs_sh17/**/weights/."""
    here = Path(__file__).resolve().parent
    guesses = [
        here / "runs_sh17" / "yolov11n_12cls_960_ft" / "weights" / "best.pt",
        here.parent / "runs_sh17" / "yolov11n_12cls_960_ft" / "weights" / "best.pt",
    ]
    for g in guesses:
        if g.is_file():
            return g
    rel = list(here.glob("**/runs_sh17/**/weights/best.pt"))
    if rel:
        rel.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return rel[0]
    rel2 = glob.glob("**/runs_sh17/**/weights/best.pt", recursive=True)
    if rel2:
        rel2.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(rel2[0])
    return None

def _resolve_weight_path(p: str):
    pth = Path(p)
    if pth.is_file():
        return pth
    # kalau yang diisi folder, cari best.pt di dalamnya
    cand = list(pth.glob("**/weights/best.pt"))
    if cand:
        cand.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cand[0]
    raise FileNotFoundError(f"Tidak menemukan weights .pt di: {p}")

@st.cache_resource(show_spinner=True)
def load_model(path: str, device_str: str = "cpu"):  # << CPU-ONLY DEFAULT
    weight_path = _resolve_weight_path(path)
    m = YOLO(weight_path)
    # Pastikan ada names
    try:
        if not getattr(m.model, "names", None):
            m.model.names = ID2NAME
    except Exception:
        pass
    m.to(device_str)
    return m

def _class_ids_from_names(names_sel):
    if not names_sel:
        return None
    name2id = {v: k for k, v in ID2NAME.items()}
    return [name2id[n] for n in names_sel if n in name2id]

# ---------- COLOR-SAFE HELPERS ----------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL (RGB) -> BGR numpy untuk Ultralytics/OpenCV."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def uploaded_image_to_bgr(file) -> np.ndarray:
    """Streamlit uploaded file -> BGR numpy via OpenCV decode."""
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    return bgr

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def annotate_to_rgb(results) -> np.ndarray:
    """results[0].plot() (BGR) -> RGB untuk display."""
    out_bgr = results[0].plot()
    return bgr_to_rgb(out_bgr)

# ---------- UI ----------
st.set_page_config(page_title=TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {TITLE}")

_auto_w = find_default_weights()
if _auto_w is None:
    st.sidebar.warning("Tidak menemukan weights di 'runs_sh17/**/weights/'. Isi path manual di bawah.")
else:
    st.sidebar.success(f"Auto-detected model: {str(_auto_w)}")

with st.sidebar:
    st.header("Controls")

    model_path = st.text_input(
        "Model path (.pt) / folder weights",
        str(_auto_w) if _auto_w else "",
        help="Bisa file .pt langsung atau folder yang berisi /weights/best.pt",
    )

    # -------- CPU-ONLY CHANGES --------
    device = "cpu"
    st.info("Running on **CPU** (Streamlit Cloud).")
    # ----------------------------------

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.01)

    # default sedikit lebih kecil agar ringan di CPU
    imgsz = st.select_slider("Inference size (px)", [640, 800, 960, 1280], value=640)

    classes_filter = st.multiselect(
        "Detect only classes (opsional)",
        options=list(ID2NAME.values()),
        default=[],
    )
    show_fps_overlay = st.checkbox("Show FPS overlay (Video/Live)", value=True)
    save_annot = st.checkbox("Save annotated outputs", value=False)
    out_dir = st.text_input("Output folder", "outputs")

    # Info env
    dev_label = get_device_info(device)
    st.caption(f"**Device:** {dev_label} · Torch {torch.__version__} · CUDA available: {torch.cuda.is_available()}")

# Load model sekali
try:
    model = load_model(model_path, device_str=device)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# ⬇️ Tabs
tab_img, tab_vid, tab_cam, tab_live = st.tabs(["📷 Image", "🎬 Video", "📸 Webcam Snapshot", "🔴 Live Webcam"])

# ---------------- IMAGE ----------------
with tab_img:
    st.subheader("Image Inference")
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file:
        bgr = uploaded_image_to_bgr(file)
        st.image(bgr_to_rgb(bgr), caption="Original", **IMGKW)

        if st.button("Run detection", key="run_img"):
            t0 = time.perf_counter()
            cls_ids = _class_ids_from_names(classes_filter)
            results = model.predict(
                source=bgr,
                conf=conf,
                classes=cls_ids,
                device=device,
                imgsz=imgsz,
                verbose=False,
            )
            ms = (time.perf_counter() - t0) * 1000
            out_rgb = annotate_to_rgb(results)
            st.image(out_rgb, caption=f"Annotated • {ms:.1f} ms • {dev_label}", **IMGKW)

            if save_annot:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(out_dir) / "annot_image.jpg"
                Image.fromarray(out_rgb).save(save_path)
                st.success(f"Saved to {save_path}")

# ---------------- VIDEO ----------------
with tab_vid:
    st.subheader("Video Inference")
    vfile = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="vid")
    if vfile:
        tmp_in = Path("tmp_in.mp4")
        with open(tmp_in, "wb") as f:
            f.write(vfile.read())

        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = None
            out_path = None
            if save_annot:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_path = Path(out_dir) / "annot_video.mp4"
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            stframe = st.empty()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            pbar = st.progress(0, text="Processing…")
            cls_ids = _class_ids_from_names(classes_filter)
            i = 0

            fps_ema = None
            total_pred_s = 0.0
            total_frames = 0

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                t1 = time.perf_counter()
                results = model.predict(
                    source=frame_bgr,  # BGR
                    conf=conf,
                    classes=cls_ids,
                    device=device,
                    imgsz=imgsz,
                    verbose=False,
                )
                annot_bgr = results[0].plot()  # BGR
                t2 = time.perf_counter()

                dt = max(t2 - t1, 1e-9)
                fps_inst = 1.0 / dt
                fps_ema = fps_inst if fps_ema is None else (0.90 * fps_ema + 0.10 * fps_inst)
                total_pred_s += dt
                total_frames += 1

                if show_fps_overlay:
                    label = f"{dev_label} | {fps_ema:.1f} FPS"
                    cv2.rectangle(annot_bgr, (8, 8), (8 + 420, 38), (0, 0, 0), -1)
                    cv2.putText(annot_bgr, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                annot_rgb = bgr_to_rgb(annot_bgr)
                stframe.image(annot_rgb, **IMGKW)
                if writer:
                    writer.write(annot_bgr)

                i += 1
                if n_frames:
                    pbar.progress(min(i / n_frames, 1.0), text=f"Processing…  {fps_ema:.1f} FPS" if fps_ema else "Processing…")

            cap.release()
            if writer:
                writer.release()
                st.success(f"Saved: {out_path}")
            pbar.empty()

            if total_frames:
                avg_fps = total_frames / total_pred_s
                st.info(f"**Average FPS** ({dev_label}, imgsz={imgsz}): {avg_fps:.1f} FPS · Frames: {total_frames}")

# --------------- WEBCAM SNAPSHOT ---------------
with tab_cam:
    st.subheader("Webcam Snapshot")
    snap = st.camera_input("Take a snapshot")
    if snap is not None:
        pil = Image.open(io.BytesIO(snap.getvalue())).convert("RGB")
        bgr = pil_to_bgr(pil)
        cls_ids = _class_ids_from_names(classes_filter)
        results = model.predict(
            source=bgr,
            conf=conf,
            classes=cls_ids,
            device=device,
            imgsz=imgsz,
            verbose=False,
        )
        out_rgb = annotate_to_rgb(results)
        st.image(out_rgb, caption=f"Annotated snapshot • {get_device_info(device)}", **IMGKW)

# --------------- 🔴 LIVE WEBCAM (Realtime) ---------------
with tab_live:
    st.subheader("Live Webcam (Realtime)")
    cam_index = st.number_input("Camera index", min_value=0, value=0, step=1, help="0 = default webcam")
    start_live = st.toggle("Start live detection", value=False)
    if start_live:
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Tidak bisa membuka webcam.")
        else:
            stframe = st.empty()
            fps_ema = None
            total_pred_s = 0.0
            total_frames = 0
            stop_btn = st.button("Stop")

            while True:
                if stop_btn:
                    break
                ret, frame_bgr = cap.read()
                if not ret:
                    st.warning("Frame webcam tidak terbaca.")
                    break

                t1 = time.perf_counter()
                results = model.predict(
                    source=frame_bgr,
                    conf=conf,
                    classes=_class_ids_from_names(classes_filter),
                    device=device,
                    imgsz=imgsz,
                    verbose=False,
                )
                annot_bgr = results[0].plot()
                t2 = time.perf_counter()

                dt = max(t2 - t1, 1e-9)
                fps_inst = 1.0 / dt
                fps_ema = fps_inst if fps_ema is None else (0.90 * fps_ema + 0.10 * fps_inst)
                total_pred_s += dt
                total_frames += 1

                if show_fps_overlay:
                    label = f"{dev_label} | {fps_ema:.1f} FPS"
                    cv2.rectangle(annot_bgr, (8, 8), (8 + 420, 38), (0, 0, 0), -1)
                    cv2.putText(annot_bgr, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                stframe.image(bgr_to_rgb(annot_bgr), **IMGKW)

            cap.release()
            if total_frames:
                avg_fps = total_frames / total_pred_s
                st.info(f"**Average FPS (Live)** ({dev_label}, imgsz={imgsz}): {avg_fps:.1f} FPS")
