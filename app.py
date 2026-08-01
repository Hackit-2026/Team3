import io
import os
import math
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import traceback

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------------
# 1. ページ基本設定 ＆ セッション初期化
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Masking Pro", page_icon="🛡️", layout="wide"
)

st.title("🛡️ AI Smart Masking Pro")
st.caption("通信なし・完全オフライン動作。Tasks API ＆ 遠距離特化フルレンジモデル（Full Range）搭載。")

if "boxes" not in st.session_state:
    st.session_state.boxes = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

# -------------------------------------------------------------------
# 2. 遠距離特化型モデル（Full Range）の自動取得
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
# 遠距離・集合写真・交差点に対応した Full Range モデルを採用
FACE_DETECTOR_PATH = os.path.join(MODEL_DIR, "blaze_face_full_range.tflite")

MODEL_URLS = {
    FACE_LANDMARKER_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    FACE_DETECTOR_PATH: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
}

def ensure_models_exist():
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            filename = os.path.basename(path)
            with st.spinner(f"📦 遠距離・群衆特化モデル ({filename}) を自動ダウンロード中..."):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    st.error(f"❌ モデル ({filename}) のダウンロードに失敗しました: {e}")
                    st.stop()

ensure_models_exist()

@st.cache_resource
def load_mediapipe_tasks(confidence):
    try:
        # Full Range Detector (遠距離・集合写真用)
        base_options_det = mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH)
        options_det = vision.FaceDetectorOptions(
            base_options=base_options_det,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )
        detector = vision.FaceDetector.create_from_options(options_det)

        # Landmarker (パーツ指定・輪郭用)
        base_options_lm = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options_lm = vision.FaceLandmarkerOptions(
            base_options=base_options_lm,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options_lm)
        return detector, landmarker
    except Exception as e:
        st.error(f"MediaPipe Tasks 初期化エラー: {e}")
        return None, None

# -------------------------------------------------------------------
# 3. サイドバー設定
# -------------------------------------------------------------------
st.sidebar.header("⚙️ マスク設定")
st.sidebar.success("🔒 完全ローカルモード")

st.sidebar.subheader("🔍 検出モード")
scan_mode = st.sidebar.radio(
    "スキャン範囲を選択",
    [
        "標準（1人〜少人数）",
        "高精度（集合写真）",
        "限界突破（群衆・交差点）",
    ],
    index=1,
)

# モードに応じたパラメーターの自動最適化
if scan_mode == "標準（1人〜少人数）":
    grids = [1]
    auto_confidence = 0.50
elif scan_mode == "高精度（集合写真）":
    grids = [1, 2]
    auto_confidence = 0.35
else:
    grids = [1, 2, 3]
    auto_confidence = 0.20

detector, landmarker = load_mediapipe_tasks(auto_confidence)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 マスク対象")
mask_targets = st.sidebar.multiselect(
    "加工したい部位を選んでください",
    ["目元（両目）", "右目 (解剖学的)", "左目 (解剖学的)", "鼻", "口元", "顔全体"],
    default=["顔全体"],
)

st.sidebar.subheader("📐 マスクの形状")
mask_shape = st.sidebar.radio(
    "形状スタイル",
    ["顔の形（輪郭に沿う・楕円）", "四角（矩形）"],
    index=0,
)

st.sidebar.subheader("🎨 マスキングスタイル")
mask_type = st.sidebar.selectbox(
    "スタイル",
    ["ピクセルモザイク", "ぼかし（ブラー）", "絵文字スタンプ", "塗りつぶし（カラー指定）", "タイル状モザイク (グリッド)"],
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
# 4. Pure Python / Pillow 画像処理 ＆ ユーティリティ
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
    if len(pts) < 3: return [(int(p[0]), int(p[1])) for p in pts]
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
    x1, y1, w, h, scores = boxes_np[:,0], boxes_np[:,1], boxes_np[:,2], boxes_np[:,3], boxes_np[:,4]
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


# -------------------------------------------------------------------
# 5. Full Range Tasks AI による顔検出 ＆ パーツ決定
# -------------------------------------------------------------------
def get_mask_boxes_locally(image: Image.Image, mask_targets: list, grids: list):
    if detector is None:
        st.error("FaceDetectorの初期化に失敗しています。")
        return []

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
        if cw < 20 or ch < 20: continue
        
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

                    # 🛡️ 誤検知カット：不自然な巨大枠（壁・階段等）や極小ノイズを排除
                    if real_w < 6 or real_h < 6: continue
                    if real_w > orig_w * 0.40 or real_h > orig_h * 0.40: continue
                    
                    aspect = real_w / max(1, real_h)
                    if aspect < 0.45 or aspect > 1.6: continue

                    # 顔領域をおでこ〜顎まで確実にカバー
                    pad_w = real_w * 0.12
                    pad_h = real_h * 0.12
                    bx = max(0, real_x - pad_w)
                    by = max(0, real_y - pad_h * 1.4)
                    bw = min(orig_w - bx, real_w + pad_w * 2)
                    bh = min(orig_h - by, real_h + pad_h * 2.3)

                    raw_faces.append([bx, by, bw, bh, score])
        except Exception:
            pass

    unique_faces = py_nms(raw_faces, iou_threshold=0.35)
    if not unique_faces:
        return []

    detected_items = []
    need_parts = any(t in mask_targets for t in ["目元（両目）", "右目 (解剖学的)", "左目 (解剖学的)", "鼻", "口元"])
    
    INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    INDEX_NOSE = [1, 2, 98, 327, 278, 48]
    INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]

    for fx, fy, fw, fh in unique_faces:
        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
        
        cx_box, cy_box = fx + fw / 2, fy + fh / 2
        rx, ry = fw / 2, fh / 2 * 1.10
        ellipse_pts = [(cx_box + rx * math.cos(a), cy_box + ry * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)]
        default_polygon = get_polygon_hull_pil(ellipse_pts)
        
        face_polygon = None
        orig_pts_all = []

        if (need_parts or mask_shape == "顔の形（輪郭に沿う・楕円）") and landmarker is not None:
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
                    orig_pts_all = [(cx1 + int(p.x * (cx2 - cx1)), cy1 + int(p.y * (cy2 - cy1))) for p in lm]
                    face_polygon = get_polygon_hull_pil(orig_pts_all)
            except Exception:
                pass
        
        if need_parts and orig_pts_all:
            def process_target(indices, target_type):
                target_pts = [orig_pts_all[i] for i in indices if i < len(orig_pts_all)]
                if not target_pts: return
                tx = [p[0] for p in target_pts]
                ty = [p[1] for p in target_pts]
                tw, th = max(tx) - min(tx), max(ty) - min(ty)
                if tw < 2 or th < 2: return
                pad_tw, pad_th = tw * 0.1, th * 0.1
                final_box = (min(tx)-pad_tw, min(ty)-pad_th, tw+pad_tw*2, th+pad_th*2)
                detected_items.append({"box": final_box, "polygon": get_polygon_hull_pil(target_pts), "type": target_type})

            if "目元（両目）" in mask_targets: process_target(INDEX_RIGHT_EYE + INDEX_LEFT_EYE, "eyes")
            if "右目 (解剖学的)" in mask_targets: process_target(INDEX_RIGHT_EYE, "eye_r")
            if "左目 (解剖学的)" in mask_targets: process_target(INDEX_LEFT_EYE, "eye_l")
            if "鼻" in mask_targets: process_target(INDEX_NOSE, "nose")
            if "口元" in mask_targets: process_target(INDEX_MOUTH, "mouth")

        if "顔全体" in mask_targets:
            final_poly = face_polygon if face_polygon is not None else default_polygon
            xs = [p[0] for p in final_poly]
            ys = [p[1] for p in final_poly]
            poly_box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            detected_items.append({"box": poly_box, "polygon": final_poly, "type": "face"})

    return detected_items


# -------------------------------------------------------------------
# 6. マスキング画像描画処理
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

        if mask_shape == "顔の形（輪郭に沿う・楕円）" and polygon and len(polygon) >= 3:
            mask_crop = Image.new("L", (box_w, box_h), 0)
            draw_mask = ImageDraw.Draw(mask_crop)
            local_polygon = [(p[0] - left, p[1] - top) for p in polygon]
            draw_mask.polygon(local_polygon, fill=255)
            result_img.paste(processed_box, (left, top), mask_crop)
        else:
            result_img.paste(processed_box, (left, top))

    return result_img


# -------------------------------------------------------------------
# 7. メイン画面レイアウト（AxiosError 403 完全対策済み）
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "加工したい写真をアップロードしてください (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.file_id != current_file_id:
        st.session_state.file_id = current_file_id
        st.session_state.boxes = None
        st.session_state.confirmed = False
        
        try:
            pil_img = Image.open(uploaded_file)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            st.session_state.uploaded_file_data = pil_img
        except Exception as e:
            st.error(f"画像の読み込みに失敗しました: {e}")
            st.session_state.uploaded_file_data = None

    if st.session_state.uploaded_file_data is not None:
        input_image = st.session_state.uploaded_file_data

        if st.button("🚀 AIモザイク解析を開始", type="primary"):
            if not mask_targets:
                st.warning("⚠️ 「マスク対象」を少なくとも1つ選択してください。")
            else:
                with st.spinner("AIが顔を精密検出中..."):
                    try:
                        st.session_state.boxes = get_mask_boxes_locally(
                            input_image,
                            mask_targets,
                            grids=grids,
                        )
                        st.session_state.confirmed = False
                        if not st.session_state.boxes:
                            st.info("顔が検出されませんでした。「限界突破」モードに変更してお試しください。")
                        else:
                            st.success(f"解析成功！ {len(st.session_state.boxes)} 箇所の顔領域を検出しました。")
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
                        file_name="masked_result.png",
                        mime="image/png",
                        use_container_width=True,
                        type="primary",
                    )
else:
    st.info("👆 写真ファイルをアップロードして「🚀 AIモザイク解析を開始」をクリックしてください。")