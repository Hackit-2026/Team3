"""顔検出・ホワイトリスト照合のコアロジック。

Team3リポジトリの `safelist` ブランチにある Streamlit 版 `app.py` から、
UI(Streamlit)依存を取り除いた純粋関数だけを移植したもの。検出アルゴリズム・
しきい値・パディング量などのロジックは変更していない。

このモジュールはモザイクの「検出」だけを担当し、実際のモザイク描画は
既存の `/api/detect-text` と同じ設計に合わせてフロントエンド側のCanvasで行う
（座標・輪郭ポリゴンだけをJSONで返す）。
"""

import io
import math
import os
import pickle
import urllib.request

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_DETECTOR_PATH = os.path.join(MODEL_DIR, "blaze_face_full_range.tflite")
FACE_EMBEDDER_PATH = os.path.join(MODEL_DIR, "mobilenet_v3_small.tflite")

MODEL_URLS = {
    FACE_LANDMARKER_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    FACE_DETECTOR_PATH: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
    FACE_EMBEDDER_PATH: "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite",
}

WHITELIST_STORE_PATH = os.path.join(MODEL_DIR, "..", "whitelist_store.pkl")

# frontend(React)側の「顔検出の精度」選択に対応する2モード。
# safelistアプリの「高精度（集合写真）」「限界突破（群衆・交差点）」に相当する
# パラメータ値はそのまま、呼び名だけを 標準/高精度 に変更している。
SCAN_MODES = {
    "standard": {"grids": [1, 2], "confidence": 0.35},
    "high": {"grids": [1, 2, 3], "confidence": 0.20},
}
DEFAULT_SCAN_MODE = "standard"

# frontend(React)側にパーツ選択・しきい値のUIが無いため既定値を固定する。
DEFAULT_MASK_TARGETS = ["顔全体"]
DEFAULT_WHITELIST_THRESHOLD = 0.70

INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
INDEX_NOSE = [1, 2, 98, 327, 278, 48]
INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]


def ensure_models_exist() -> None:
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
                out_file.write(response.read())


# 検出モードごとにしきい値(confidence)が異なるため、Detectorはconfidence値
# ごとにキャッシュする(Landmarkerはconfidenceに依存しないため単一キャッシュ)。
_detectors: dict = {}
_landmarker = None
_embedder = None


def _get_detector(confidence: float):
    if confidence in _detectors:
        return _detectors[confidence]

    ensure_models_exist()
    base_options_det = mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH)
    options_det = vision.FaceDetectorOptions(
        base_options=base_options_det,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=confidence,
    )
    detector = vision.FaceDetector.create_from_options(options_det)
    _detectors[confidence] = detector
    return detector


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    ensure_models_exist()
    base_options_lm = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options_lm = vision.FaceLandmarkerOptions(
        base_options=base_options_lm,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    _landmarker = vision.FaceLandmarker.create_from_options(options_lm)
    return _landmarker


def get_face_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    ensure_models_exist()
    base_options = mp_python.BaseOptions(model_asset_path=FACE_EMBEDDER_PATH)
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        l2_normalize=True,
        quantize=False,
    )
    _embedder = vision.ImageEmbedder.create_from_options(options)
    return _embedder


def get_polygon_hull_pil(pts):
    if len(pts) < 3:
        return [(int(p[0]), int(p[1])) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    sorted_pts = sorted(pts, key=lambda p: (p[0], p[1]))
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [(int(p[0]), int(p[1])) for p in (lower[:-1] + upper[:-1])]


def py_nms(boxes, iou_threshold=0.35):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    x1, y1, w, h, scores = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3], boxes_np[:, 4]
    x2, y2 = x1 + w, y1 + h
    areas = w * h

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i][:4])
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _detect_raw_faces(image: Image.Image, grids: list, confidence: float) -> list:
    detector = _get_detector(confidence)
    orig_w, orig_h = image.size
    raw_faces = []

    crops = []
    for g in grids:
        if g == 1:
            crops.append((0, 0, orig_w, orig_h))
            continue
        step_x = orig_w // g
        step_y = orig_h // g
        overlap_x = int(step_x * 0.3)
        overlap_y = int(step_y * 0.3)
        for i in range(g):
            for j in range(g):
                x1 = max(0, i * step_x - overlap_x)
                y1 = max(0, j * step_y - overlap_y)
                x2 = min(orig_w, (i + 1) * step_x + overlap_x)
                y2 = min(orig_h, (j + 1) * step_y + overlap_y)
                crops.append((x1, y1, x2 - x1, y2 - y1))

    for cx, cy, cw, ch in crops:
        if cw < 20 or ch < 20:
            continue
        crop_pil = image.crop((cx, cy, cx + cw, cy + ch))
        crop_np = np.array(crop_pil)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_np))
        try:
            results = detector.detect(mp_image)
            if results and results.detections:
                for detection in results.detections:
                    score = detection.categories[0].score
                    bbox = detection.bounding_box
                    real_x = cx + bbox.origin_x
                    real_y = cy + bbox.origin_y
                    real_w = bbox.width
                    real_h = bbox.height

                    if real_w < 6 or real_h < 6:
                        continue
                    if real_w > orig_w * 0.40 or real_h > orig_h * 0.40:
                        continue
                    aspect = real_w / max(1, real_h)
                    if aspect < 0.45 or aspect > 1.6:
                        continue

                    raw_faces.append([real_x, real_y, real_w, real_h, score])
        except Exception:
            pass

    return raw_faces


def get_mask_boxes_locally(image: Image.Image, mask_targets: list, grids: list, confidence: float) -> list:
    orig_w, orig_h = image.size
    landmarker = _get_landmarker()

    raw_faces = _detect_raw_faces(image, grids, confidence)
    padded_faces = []
    for real_x, real_y, real_w, real_h, score in raw_faces:
        pad_w = real_w * 0.12
        pad_h = real_h * 0.12
        bx = max(0, real_x - pad_w)
        by = max(0, real_y - pad_h * 1.4)
        bw = min(orig_w - bx, real_w + pad_w * 2)
        bh = min(orig_h - by, real_h + pad_h * 2.3)
        padded_faces.append([bx, by, bw, bh, score])

    unique_faces = py_nms(padded_faces, iou_threshold=0.35)
    if not unique_faces:
        return []

    detected_items = []
    need_parts = any(
        t in mask_targets for t in ["目元（両目）", "右目 (解剖学的)", "左目 (解剖学的)", "鼻", "口元"]
    )

    for fx, fy, fw, fh in unique_faces:
        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)

        cx_box, cy_box = fx + fw / 2, fy + fh / 2
        rx, ry = fw / 2, fh / 2 * 1.10
        ellipse_pts = [
            (cx_box + rx * math.cos(a), cy_box + ry * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, 16)
        ]
        default_polygon = get_polygon_hull_pil(ellipse_pts)

        face_polygon = None
        orig_pts_all = []

        if (need_parts or True) and landmarker is not None:
            pad_x = int(fw * 0.2)
            pad_y = int(fh * 0.2)
            cx1 = max(0, fx - pad_x)
            cy1 = max(0, fy - pad_y)
            cx2 = min(orig_w, fx + fw + pad_x)
            cy2 = min(orig_h, fy + fh + pad_y)

            face_crop = image.crop((cx1, cy1, cx2, cy2))
            crop_np = np.array(face_crop)
            mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_np))

            try:
                res_lm = landmarker.detect(mp_crop)
                if res_lm and res_lm.face_landmarks:
                    lm = res_lm.face_landmarks[0]
                    orig_pts_all = [
                        (cx1 + int(p.x * (cx2 - cx1)), cy1 + int(p.y * (cy2 - cy1))) for p in lm
                    ]
                    face_polygon = get_polygon_hull_pil(orig_pts_all)
            except Exception:
                pass

        if need_parts and orig_pts_all:
            def process_target(indices, target_type):
                target_pts = [orig_pts_all[i] for i in indices if i < len(orig_pts_all)]
                if not target_pts:
                    return
                tx = [p[0] for p in target_pts]
                ty = [p[1] for p in target_pts]
                tw, th = max(tx) - min(tx), max(ty) - min(ty)
                if tw < 2 or th < 2:
                    return
                pad_tw, pad_th = tw * 0.1, th * 0.1
                final_box = (min(tx) - pad_tw, min(ty) - pad_th, tw + pad_tw * 2, th + pad_th * 2)
                detected_items.append(
                    {"box": final_box, "polygon": get_polygon_hull_pil(target_pts), "type": target_type}
                )

            if "目元（両目）" in mask_targets:
                process_target(INDEX_RIGHT_EYE + INDEX_LEFT_EYE, "eyes")
            if "右目 (解剖学的)" in mask_targets:
                process_target(INDEX_RIGHT_EYE, "eye_r")
            if "左目 (解剖学的)" in mask_targets:
                process_target(INDEX_LEFT_EYE, "eye_l")
            if "鼻" in mask_targets:
                process_target(INDEX_NOSE, "nose")
            if "口元" in mask_targets:
                process_target(INDEX_MOUTH, "mouth")

        if "顔全体" in mask_targets:
            final_poly = face_polygon if face_polygon is not None else default_polygon
            xs = [p[0] for p in final_poly]
            ys = [p[1] for p in final_poly]
            poly_box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            detected_items.append({"box": poly_box, "polygon": final_poly, "type": "face"})

    return detected_items


# ---------------------------------------------------------------------------
# ホワイトリスト（登録者はモザイク除外）
# ---------------------------------------------------------------------------


def load_whitelist() -> list:
    if os.path.exists(WHITELIST_STORE_PATH):
        try:
            with open(WHITELIST_STORE_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_whitelist(whitelist: list) -> None:
    with open(WHITELIST_STORE_PATH, "wb") as f:
        pickle.dump(whitelist, f)


def compute_face_embedding(image: Image.Image, box):
    embedder = get_face_embedder()
    x, y, w, h = box
    orig_w, orig_h = image.size
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(orig_w, int(x + w))
    bottom = min(orig_h, int(y + h))
    if right - left < 8 or bottom - top < 8:
        return None
    face_crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop_np = np.ascontiguousarray(np.array(face_crop))
    try:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_np)
        res = embedder.embed(mp_img)
        return np.asarray(res.embeddings[0].embedding, dtype=np.float32)
    except Exception:
        return None


def _cosine(a, b):
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def register_face_from_image(pil_image: Image.Image, grids: list = None, confidence: float = None):
    """登録用写真から最も大きい顔を1つ検出・特徴量化して返す。

    戻り値: (embedding: list[float], thumb: bytes) 。顔が見つからなければ None。
    """
    mode = SCAN_MODES[DEFAULT_SCAN_MODE]
    grids = grids if grids is not None else mode["grids"]
    confidence = confidence if confidence is not None else mode["confidence"]

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    raw_faces = _detect_raw_faces(pil_image, grids, confidence)
    if not raw_faces:
        return None
    bx, by, bw, bh, _score = max(raw_faces, key=lambda b: b[2] * b[3])
    emb = compute_face_embedding(pil_image, (bx, by, bw, bh))
    if emb is None:
        return None
    thumb = pil_image.crop((int(bx), int(by), int(bx + bw), int(by + bh)))
    thumb.thumbnail((120, 120))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return emb.tolist(), buf.getvalue()


def compute_whitelist_exemptions(
    image: Image.Image, grids: list, whitelist: list, threshold: float, confidence: float
):
    exempt_boxes = []
    if not whitelist:
        return exempt_boxes

    wl_embs = [np.asarray(e["embedding"], dtype=np.float32) for e in whitelist]
    raw_faces = _detect_raw_faces(image, grids, confidence)

    for fx, fy, fw, fh, _score in raw_faces:
        emb = compute_face_embedding(image, (fx, fy, fw, fh))
        if emb is None:
            continue
        sims = [_cosine(emb, w) for w in wl_embs]
        best_sim = max(sims) if sims else -1.0
        if best_sim >= threshold:
            exempt_boxes.append((fx, fy, fw, fh))
    return exempt_boxes


def _box_center_inside(item_box, regions, tol_ratio=0.15):
    ix, iy, iw, ih = item_box
    cx, cy = ix + iw / 2.0, iy + ih / 2.0
    for (rx, ry, rw, rh) in regions:
        px, py = rw * tol_ratio, rh * tol_ratio
        if (rx - px) <= cx <= (rx + rw + px) and (ry - py) <= cy <= (ry + rh + py):
            return True
    return False


def filter_out_whitelisted(items: list, exempt_boxes: list) -> list:
    if not exempt_boxes:
        return items
    return [it for it in items if not _box_center_inside(it["box"], exempt_boxes)]
