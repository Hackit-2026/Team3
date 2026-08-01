import io
import os
import math
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import traceback

# -------------------------------------------------------------------
# 1. ページ基本設定
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Masking Pro Ultra", page_icon="🛡️", layout="wide"
)

st.title("🛡️ AI Smart Masking Pro Ultra")
st.caption("通信なし・完全オフライン。Landmarker + Detector ハイブリッドエンジン ＆ 100分割超拡大スキャン搭載。")

# -------------------------------------------------------------------
# 2. モデルファイルの自動ダウンロード ＆ ロード
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
FACE_DETECTOR_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")

# 最新のGoogle公式直リンクURLに修正
MODEL_URLS = {
    FACE_LANDMARKER_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    FACE_DETECTOR_PATH: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
}

def ensure_models_exist():
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            filename = os.path.basename(path)
            with st.spinner(f"📦 必要なモデルファイル ({filename}) を初期ダウンロード中..."):
                try:
                    req = urllib.request.Request(
                        url, 
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                        out_file.write(response.read())
                    st.success(f"✨ {filename} のダウンロードが完了しました！")
                except Exception as e:
                    if filename == "face_landmarker.task":
                        st.error(f"❌ 必須モデル ({filename}) のダウンロードに失敗しました: {e}")
                        st.stop()
                    else:
                        st.warning(f"⚠️ サブモデル ({filename}) のダウンロードをスキップしました (Landmarker単体で動作します): {e}")

ensure_models_exist()

@st.cache_resource
def load_mediapipe_tasks():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        # 精密ランドマーク用 (必須)
        landmarker = None
        if os.path.exists(FACE_LANDMARKER_PATH):
            base_options_lm = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
            options_lm = vision.FaceLandmarkerOptions(
                base_options=base_options_lm,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=150,
                min_face_detection_confidence=0.01,
                min_face_presence_confidence=0.01,
                min_tracking_confidence=0.01,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options_lm)
        
        # 高感度検出用 (オプション)
        detector = None
        if os.path.exists(FACE_DETECTOR_PATH):
            try:
                base_options_det = mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH)
                options_det = vision.FaceDetectorOptions(
                    base_options=base_options_det,
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.01,
                )
                detector = vision.FaceDetector.create_from_options(options_det)
            except Exception:
                pass

        return mp, landmarker, detector
    except Exception as e:
        st.error(f"初期化エラー: {e}")
        return None, None, None

mp, landmarker, detector = load_mediapipe_tasks()

# セッション状態の初期化
if "boxes" not in st.session_state:
    st.session_state.boxes = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# -------------------------------------------------------------------
# 3. サイドバー設定
# -------------------------------------------------------------------
st.sidebar.header("⚙️ マスク設定")
st.sidebar.success("🔒 完全ローカルモード (Wi-FiオフOK)")

# 🎯 検出精度モード
st.sidebar.subheader("🔍 検出精度")
precision_level = st.sidebar.radio(
    "精度モードを選択",
    [
        "普通（標準バランス）",
        "高精度（集合写真・少人数）",
        "超高精度（大人数・上限150人）",
        "限界突破・群衆（高密度・極小顔）",
    ],
    index=3,
    help="「限界突破」は100分割 ＆ 3倍超拡大ハイブリッドスキャンを行います。処理時間はかかりますが、最強の検知精度です。"
)

if precision_level == "普通（標準バランス）":
    grids = [1, 2]
    scale_up_factor = 1.2
    dup_thresh = 0.45
elif precision_level == "高精度（集合写真・少人数）":
    grids = [1, 2, 3]
    scale_up_factor = 1.5
    dup_thresh = 0.40
elif precision_level == "超高精度（大人数・上限150人）":
    grids = [1, 2, 4, 6]
    scale_up_factor = 2.2
    dup_thresh = 0.35
else:  # 限界突破
    grids = [1, 2, 4, 7, 10]
    scale_up_factor = 3.0
    dup_thresh = 0.30

st.sidebar.markdown("---")

# 🎯 マスク対象
st.sidebar.subheader("🎯 マスク対象")
mask_targets = st.sidebar.multiselect(
    "加工したい部位を選んでください",
    [
        "目元（両目）",
        "右目 (解剖学的)",
        "左目 (解剖学的)",
        "鼻",
        "口元",
        "顔全体",
    ],
    default=["顔全体"],
)

# 🔷 マスク形状
st.sidebar.subheader("📐 マスクの形状")
mask_shape = st.sidebar.radio(
    "形状スタイル",
    ["顔の形（輪郭に沿う）", "四角（矩形）"],
    index=0,
)

# 🎨 スタイル選択
st.sidebar.subheader("🎨 マスキングスタイル")
mask_type = st.sidebar.selectbox(
    "スタイル",
    [
        "ピクセルモザイク",
        "ぼかし（ブラー）",
        "絵文字スタンプ",
        "塗りつぶし（カラー指定）",
        "タイル状モザイク (グリッド)",
    ],
    index=0,
)

grid_size = 30
mosaic_size = 15
blur_radius = 20
fill_color = "#000000"
emoji_char = "🌸"
emoji_scale = 110
emoji_angle = 0
offset_x = 0
offset_y = 0

if mask_type == "タイル状モザイク (グリッド)":
    grid_size = st.sidebar.slider("タイルの大きさ (ピクセル)", 10, 100, 30, 5)
elif mask_type == "絵文字スタンプ":
    emoji_char = st.sidebar.text_input("使用する絵文字を入力", value="🌸")
    emoji_scale = st.sidebar.slider("絵文字の倍率 (%)", 70, 200, 110, 5)
    emoji_angle = st.sidebar.slider("絵文字の回転角度 (°)", -180, 180, 0, 5)
    st.sidebar.markdown("##### 📍 位置微調整")
    offset_y = st.sidebar.slider("上下位置 (Y軸)", -50, 50, 0, 2)
    offset_x = st.sidebar.slider("左右位置 (X軸)", -50, 50, 0, 2)
elif mask_type == "ピクセルモザイク":
    mosaic_size = st.sidebar.slider("モザイクの粗さ", 5, 30, 15)
elif mask_type == "ぼかし（ブラー）":
    blur_radius = st.sidebar.slider("ぼかしの強さ", 5, 50, 20)
elif mask_type == "塗りつぶし（カラー指定）":
    fill_color = st.sidebar.color_picker("塗りつぶしの色", "#000000")


# -------------------------------------------------------------------
# 4. Pure Python / Pillow 画像処理ユーティリティ
# -------------------------------------------------------------------
def get_emoji_font(font_size: int):
    font_candidates = ["seguiemj.ttf", "NotoColorEmoji.ttf", "Apple Color Emoji.ttc", "arial.ttf"]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_cropped_emoji_image(emoji_char: str, target_size: int) -> Image.Image:
    canvas_size = max(250, int(target_size * 1.5))
    temp_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    font = get_emoji_font(int(canvas_size * 0.7))

    try:
        draw.text((canvas_size / 2, canvas_size / 2), emoji_char, font=font, anchor="mm", embedded_color=True)
    except Exception:
        draw.text((canvas_size / 2, canvas_size / 2), emoji_char, font=font, anchor="mm", fill=(0, 0, 0))

    bbox = temp_img.getbbox()
    cropped = temp_img.crop(bbox) if bbox else temp_img
    w, h = cropped.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    aspect = w / h
    new_w = target_size if aspect > 1 else max(1, int(target_size * aspect))
    new_h = max(1, int(target_size / aspect)) if aspect > 1 else target_size
    return cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)


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

    hull = lower[:-1] + upper[:-1]
    return [(int(p[0]), int(p[1])) for p in hull]

def py_nms(items, iou_threshold):
    if not items:
        return []

    scores = np.array([item["score"] for item in items])
    boxes = np.array([ [item["box"][0], item["box"][1], item["box"][0]+item["box"][2], item["box"][1]+item["box"][3]] for item in items ])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(items[i])

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# -------------------------------------------------------------------
# 5. ハイブリッド検出 ＆ 限界突破スキャンエンジン
# -------------------------------------------------------------------
def get_mask_boxes_locally(
    image: Image.Image,
    mask_targets: list,
    grids: list,
    scale_factor: float = 3.0,
    dup_thresh: float = 0.30,
):
    if landmarker is None:
        st.error("FaceLandmarkerエンジンの初期化に失敗しています。")
        return []

    orig_w, orig_h = image.size
    all_raw_items = []

    INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    INDEX_NOSE = [1, 2, 98, 327, 278, 48]
    INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]

    crops = []
    for g in grids:
        if g == 1:
            crops.append((0, 0, orig_w, orig_h))
            continue
        step_x = orig_w // g
        step_y = orig_h // g
        overlap_x = int(step_x * 0.45)
        overlap_y = int(step_y * 0.45)

        for i in range(g):
            for j in range(g):
                x1 = max(0, i * step_x - overlap_x)
                y1 = max(0, j * step_y - overlap_y)
                x2 = min(orig_w, (i + 1) * step_x + overlap_x)
                y2 = min(orig_h, (j + 1) * step_y + overlap_y)
                crops.append((x1, y1, x2 - x1, y2 - y1))

    progress_bar = st.progress(0)
    num_crops = len(crops)

    for idx, (cx_off, cy_off, cw, ch) in enumerate(crops):
        progress_bar.progress(int((idx + 1) / num_crops * 100), text=f"画像を分割スキャン中... ({idx+1}/{num_crops})")
        if cw < 10 or ch < 10: continue

        crop_pil = image.crop((cx_off, cy_off, cx_off + cw, cy_off + ch))

        if scale_factor > 1.0 and (cw * scale_factor < 6000 and ch * scale_factor < 6000):
            scaled_crop = crop_pil.resize((int(cw * scale_factor), int(ch * scale_factor)), Image.Resampling.BICUBIC)
            curr_scale = scale_factor
        else:
            scaled_crop = crop_pil
            curr_scale = 1.0

        crop_np = np.array(scaled_crop)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_np))

        # --- A. FaceDetector (利用可能な場合) ---
        if detector is not None:
            try:
                result_det = detector.detect(mp_image)
                if result_det and result_det.detections:
                    for detection in result_det.detections:
                        score = detection.categories[0].score
                        bbox = detection.bounding_box
                        real_x = cx_off + (bbox.origin_x / curr_scale)
                        real_y = cy_off + (bbox.origin_y / curr_scale)
                        real_w = bbox.width / curr_scale
                        real_h = bbox.height / curr_scale

                        if real_w < 3 or real_h < 3: continue

                        center_pt = (real_x + real_w/2, real_y + real_h/2)
                        axes = (max(1, int(real_w / 2 * 1.1)), max(1, int(real_h * 0.7)))
                        num_pts = 20
                        ellipse_pts = []
                        for i in range(num_pts):
                            angle = 2 * math.pi * i / num_pts
                            ellipse_pts.append((center_pt[0] + axes[0]*math.cos(angle), center_pt[1] + axes[1]*math.sin(angle)))

                        all_raw_items.append({
                            "box": (real_x, real_y, real_w, real_h),
                            "center": center_pt,
                            "polygon": get_polygon_hull_pil(ellipse_pts),
                            "score": score,
                            "engine": "detector",
                            "indices": None
                        })
            except: pass

        # --- B. FaceLandmarker ---
        try:
            result_lm = landmarker.detect(mp_image)
            if result_lm and result_lm.face_landmarks:
                for face_landmarks in result_lm.face_landmarks:
                    score = sum([lm.presence for lm in face_landmarks]) / len(face_landmarks)
                    
                    orig_pts_all = []
                    for lm in face_landmarks:
                        lx = getattr(lm, 'x', lm['x'] if isinstance(lm, dict) else 0)
                        ly = getattr(lm, 'y', lm['y'] if isinstance(lm, dict) else 0)
                        orig_pts_all.append((cx_off + (lx * cw * curr_scale) / curr_scale, cy_off + (ly * ch * curr_scale) / curr_scale))

                    if len(orig_pts_all) < 10: continue

                    x_coords = [p[0] for p in orig_pts_all]
                    y_coords = [p[1] for p in orig_pts_all]
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    real_w, real_h = xmax - xmin, ymax - ymin
                    if real_w < 3 or real_h < 3: continue

                    all_raw_items.append({
                        "box": (xmin, ymin, real_w, real_h),
                        "center": ((xmin+xmax)/2, (ymin+ymax)/2),
                        "polygon": get_polygon_hull_pil(orig_pts_all),
                        "score": score + 0.5,
                        "engine": "landmarker",
                        "indices": {
                            "all": range(len(orig_pts_all)),
                            "eyes": INDEX_RIGHT_EYE + INDEX_LEFT_EYE,
                            "eye_r": INDEX_RIGHT_EYE,
                            "eye_l": INDEX_LEFT_EYE,
                            "nose": INDEX_NOSE,
                            "mouth": INDEX_MOUTH
                        }
                    })
        except: pass

    progress_bar.empty()
    
    if not all_raw_items:
        return []

    with st.spinner("検出結果を統合中..."):
        kept_items = py_nms(all_raw_items, dup_thresh)

    detected_items = []
    for item in kept_items:
        if item["indices"] is None:
            if "顔全体" in mask_targets:
                detected_items.append({"box": item["box"], "polygon": item["polygon"], "type": "face"})
            continue

        indices_map = item["indices"]
        def process_target(indices_key, target_type):
            target_pts = [ item["polygon"][i] for i in indices_map[indices_key] if i < len(item["polygon"]) ]
            if not target_pts: return
            x_coords = [p[0] for p in target_pts]
            y_coords = [p[1] for p in target_pts]
            w, h = max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)
            if w < 2 or h < 2: return
            detected_items.append({"box": (min(x_coords), min(y_coords), w, h), "polygon": get_polygon_hull_pil(target_pts), "type": target_type})

        if "顔全体" in mask_targets:
            detected_items.append({"box": item["box"], "polygon": item["polygon"], "type": "face"})
        if "目元（両目）" in mask_targets: process_target("eyes", "eyes")
        if "右目 (解剖学的)" in mask_targets: process_target("eye_r", "eye_r")
        if "左目 (解剖学的)" in mask_targets: process_target("eye_l", "eye_l")
        if "鼻" in mask_targets: process_target("nose", "nose")
        if "口元" in mask_targets: process_target("mouth", "mouth")

    return detected_items


# -------------------------------------------------------------------
# 6. マスキング画像適用処理
# -------------------------------------------------------------------
def apply_masking(
    image: Image.Image,
    items: list,
    mask_shape: str,
    mask_type: str,
    grid_size: int,
    mosaic_size: int,
    blur_radius: int,
    fill_color: str,
    offset_x: int,
    offset_y: int,
    emoji_char: str,
    emoji_scale: int,
    emoji_angle: int,
) -> Image.Image:
    result_img = image.copy().convert("RGB")
    img_w, img_h = image.size

    if mask_type == "タイル状モザイク (グリッド)":
        target_rects = [item["box"] for item in items]
        for gx in range(0, img_w, grid_size):
            for gy in range(0, img_h, grid_size):
                tile_l, tile_t = gx, gy
                tile_r, tile_b = min(img_w, gx + grid_size), min(img_h, gy + grid_size)

                has_face = False
                for bx, by, bw, bh in target_rects:
                    if max(tile_l, bx) < min(tile_r, bx + bw) and max(tile_t, by) < min(tile_b, by + bh):
                        has_face = True
                        break

                if has_face:
                    tile_region = result_img.crop((tile_l, tile_t, tile_r, tile_b))
                    small_tile = tile_region.resize((1, 1), resample=Image.Resampling.NEAREST)
                    mosaic_tile = small_tile.resize((tile_r - tile_l, tile_b - tile_t), resample=Image.Resampling.NEAREST)
                    result_img.paste(mosaic_tile, (tile_l, tile_t))
        return result_img

    for item in items:
        bx, by, bw, bh = item["box"]
        polygon = item["polygon"]

        left = max(0, min(img_w - 1, int(bx)))
        top = max(0, min(img_h - 1, int(by)))
        right = max(0, min(img_w, int(bx + bw)))
        bottom = max(0, min(img_h, int(by + bh)))
        box_w, box_h = right - left, bottom - top

        if box_w <= 0 or box_h <= 0: continue

        cx, cy = left + box_w / 2, top + box_h / 2
        masked_box = result_img.crop((left, top, right, bottom))

        if mask_type == "ピクセルモザイク":
            m_w = max(1, box_w // max(1, mosaic_size))
            m_h = max(1, box_h // max(1, mosaic_size))
            small = masked_box.resize((m_w, m_h), resample=Image.Resampling.NEAREST)
            processed_box = small.resize((box_w, box_h), resample=Image.Resampling.NEAREST)

        elif mask_type == "ぼかし（ブラー）":
            processed_box = masked_box.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        elif mask_type == "塗りつぶし（カラー指定）":
            processed_box = Image.new("RGB", (box_w, box_h), fill_color)

        elif mask_type == "絵文字スタンプ":
            target_size = max(int(max(box_w, box_h) * 0.9), min(int(max(box_w, box_h) * 1.15 * (emoji_scale / 100.0)), int(max(box_w, box_h) * 2.2)))
            emoji_img = create_cropped_emoji_image(emoji_char, target_size=target_size)
            if emoji_angle != 0: emoji_img = emoji_img.rotate(-emoji_angle, expand=True, resample=Image.Resampling.BICUBIC)
            ew, eh = emoji_img.size
            result_img.paste(emoji_img, (int(cx - ew / 2 + offset_x), int(cy - eh / 2 + offset_y)), emoji_img)
            continue

        if mask_shape == "顔の形（輪郭に沿う）" and polygon and len(polygon) >= 3:
            mask_crop = Image.new("L", (box_w, box_h), 0)
            draw_mask = ImageDraw.Draw(mask_crop)
            local_polygon = [(p[0] - left, p[1] - top) for p in polygon]
            draw_mask.polygon(local_polygon, fill=255)
            result_img.paste(processed_box, (left, top), mask_crop)
        else:
            result_img.paste(processed_box, (left, top))

    return result_img


# -------------------------------------------------------------------
# 7. メイン画面レイアウト
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "群衆・交差点の写真をアップロードしてください (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.file_id != current_file_id:
        st.session_state.file_id = current_file_id
        st.session_state.boxes = None
        st.session_state.confirmed = False

    input_image = Image.open(uploaded_file).convert("RGB")

    if st.button("🚀 ローカル AI で解析を開始（限界突破モードは時間がかかります）", type="primary"):
        if not mask_targets:
            st.warning("⚠️ 「マスク対象」を少なくとも1つ選択してください。")
        else:
            if precision_level == "限界突破・群衆（高密度・極小顔）":
                st.info("⚠️ 【限界突破モード】 画像を100分割し、ハイブリッド超拡大スキャンを行います。しばらくお待ちください。")
            
            with st.spinner("AIがフル解析中..."):
                try:
                    st.session_state.boxes = get_mask_boxes_locally(
                        input_image,
                        mask_targets,
                        grids=grids,
                        scale_factor=scale_up_factor,
                        dup_thresh=dup_thresh,
                    )
                    st.session_state.confirmed = False
                    if not st.session_state.boxes:
                        st.info("顔が検出されませんでした。精度モードを上げるか、別の画像をお試しください。")
                    else:
                        st.success(f"解析成功！ 「{precision_level}」モードで {len(st.session_state.boxes)} 箇所の部位を検出し、重複を統合しました。")
                except Exception as e:
                    st.error(f"解析エラーが発生しました: {e}")
                    st.text(traceback.format_exc())

    if st.session_state.boxes is not None:
        processed_image = apply_masking(
            input_image,
            st.session_state.boxes,
            mask_shape,
            mask_type,
            grid_size,
            mosaic_size,
            blur_radius,
            fill_color,
            offset_x,
            offset_y,
            emoji_char,
            emoji_scale,
            emoji_angle,
        )

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📷 元の画像")
            st.image(input_image, use_column_width=True)
        with col_right:
            st.subheader("🔍 マスキング結果")
            st.image(processed_image, use_column_width=True)

        st.markdown("---")
        confirm_col1, confirm_col2 = st.columns([2, 3])
        with confirm_col1:
            if not st.session_state.confirmed:
                if st.button("✅ これで決定", type="primary", use_container_width=True):
                    st.session_state.confirmed = True
                    st.rerun()
            else:
                st.success("✨ 位置とデザインが確定しました！")
                if st.button("🔄 もう一度微調整する", use_container_width=True):
                    st.session_state.confirmed = False
                    st.rerun()
        with confirm_col2:
            if st.session_state.confirmed:
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                st.download_button(
                    label="💾 加工画像をダウンロード",
                    data=buf.getvalue(),
                    file_name="ultra_masked.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary",
                )
else:
    st.info("👆 写真ファイルをアップロードして「🚀 ローカル AI で解析を開始」をクリックしてください。")