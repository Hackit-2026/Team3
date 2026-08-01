import io
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import traceback

# -------------------------------------------------------------------
# 1. ページ基本設定
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Masking Pro", page_icon="🛡️", layout="wide"
)

st.title("🛡️ AI Smart Masking Pro")
st.caption("通信なし・完全オフライン動作。Tasks API ＋ 多層フォールバックエンジンによる超高精度モザイク処理。")

# -------------------------------------------------------------------
# 2. モデルファイルのロード ＆ OpenCVカスケード準備
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

@st.cache_resource
def load_mediapipe_tasks():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        landmarker = None
        if os.path.exists(FACE_LANDMARKER_PATH):
            base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=150,
                min_face_detection_confidence=0.05,
                min_face_presence_confidence=0.05,
                min_tracking_confidence=0.05,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # バックアップ用 OpenCV 顔検出カスケード
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        return mp, landmarker, face_cascade
    except Exception as e:
        st.error(f"初期化エラー: {e}")
        return None, None, None

mp, landmarker, face_cascade = load_mediapipe_tasks()

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
        "限界突破（スクランブル交差点・高密度150人）",
    ],
    index=3,
    help="「限界突破」はマルチレイヤースキャンと超解像処理を行い、交差点の極小顔まで漏らさず自動検出します。"
)

if precision_level == "普通（標準バランス）":
    grids = [1, 2]
    scale_up_factor = 1.0
    dup_thresh = 0.04
    use_clahe = False
elif precision_level == "高精度（集合写真・少人数）":
    grids = [1, 2, 3]
    scale_up_factor = 1.3
    dup_thresh = 0.02
    use_clahe = False
elif precision_level == "超高精度（大人数・上限150人）":
    grids = [1, 2, 4, 5]
    scale_up_factor = 1.8
    dup_thresh = 0.012
    use_clahe = True
else:  # 限界突破
    grids = [1, 2, 4, 6, 8]
    scale_up_factor = 2.5
    dup_thresh = 0.005
    use_clahe = True

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
# 4. 画像処理ユーティリティ
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


def apply_clahe(img_np):
    try:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception:
        return img_np


def get_rotated_image_and_inv_matrix(image_np, angle):
    if angle == 0:
        return image_np, None

    h, w = image_np.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    
    rotated_img = cv2.warpAffine(image_np, M, (new_w, new_h))
    M_inv = cv2.invertAffineTransform(M)
    return rotated_img, M_inv


def map_points_back(points, M_inv):
    clean_pts = []
    for p in points:
        if len(p) >= 2:
            try:
                clean_pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass

    if len(clean_pts) == 0:
        return []

    if M_inv is None:
        return [(int(p[0]), int(p[1])) for p in clean_pts]

    m00, m01, m02 = float(M_inv[0][0]), float(M_inv[0][1]), float(M_inv[0][2])
    m10, m11, m12 = float(M_inv[1][0]), float(M_inv[1][1]), float(M_inv[1][2])

    res = []
    for x, y in clean_pts:
        x_orig = int(m00 * x + m01 * y + m02)
        y_orig = int(m10 * x + m11 * y + m12)
        res.append((x_orig, y_orig))
    return res


def get_polygon_from_pts(target_pts):
    if not target_pts or len(target_pts) < 3:
        return [(int(p[0]), int(p[1])) for p in target_pts]
    try:
        pts_np = np.array(target_pts, dtype=np.int32).reshape(-1, 2)
        hull = cv2.convexHull(pts_np)
        if hull is not None and len(hull) >= 3:
            return [(int(pt[0]), int(pt[1])) for pt in hull.reshape(-1, 2)]
    except Exception:
        pass
    return [(int(p[0]), int(p[1])) for p in target_pts]


# -------------------------------------------------------------------
# 5. マルチレイヤーハイブリッド検出スキャン
# -------------------------------------------------------------------
def get_mask_boxes_locally(
    image: Image.Image,
    mask_targets: list,
    grids: list,
    scale_factor: float = 2.0,
    dup_thresh: float = 0.01,
    apply_enhance: bool = False
):
    original_np = np.array(image)
    processed_np = apply_clahe(original_np) if apply_enhance else original_np
    orig_h, orig_w, _ = processed_np.shape

    detected_items = []
    processed_centers = []

    INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    INDEX_NOSE = [1, 2, 98, 327, 278, 48]
    INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]

    angles = [0, 15, -15]

    def is_duplicate(cx, cy):
        for pcx, pcy in processed_centers:
            if abs(pcx - cx) < dup_thresh and abs(pcy - cy) < dup_thresh:
                return True
        return False

    for angle in angles:
        rotated_np, M_inv = get_rotated_image_and_inv_matrix(processed_np, angle)
        rot_h, rot_w, _ = rotated_np.shape

        crops = []
        for g in grids:
            if g == 1:
                crops.append((0, 0, rot_w, rot_h))
                continue
            
            step_x = rot_w // g
            step_y = rot_h // g
            overlap_x = int(step_x * 0.4) 
            overlap_y = int(step_y * 0.4)

            for i in range(g):
                for j in range(g):
                    x1 = max(0, i * step_x - overlap_x)
                    y1 = max(0, j * step_y - overlap_y)
                    x2 = min(rot_w, (i + 1) * step_x + overlap_x)
                    y2 = min(rot_h, (j + 1) * step_y + overlap_y)
                    crops.append((x1, y1, x2 - x1, y2 - y1))

        for cx_off, cy_off, cw, ch in crops:
            crop_img = rotated_np[cy_off:cy_off+ch, cx_off:cx_off+cw]
            if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                continue

            if scale_factor > 1.0 and (cw * scale_factor < 5000 and ch * scale_factor < 5000):
                scaled_crop = cv2.resize(
                    crop_img, 
                    (int(cw * scale_factor), int(ch * scale_factor)), 
                    interpolation=cv2.INTER_CUBIC
                )
                curr_scale = scale_factor
            else:
                scaled_crop = crop_img
                curr_scale = 1.0

            # --- A. MediaPipe Tasks (FaceLandmarker) ---
            if landmarker is not None:
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(scaled_crop))
                    result_lm = landmarker.detect(mp_image)
                    if result_lm and result_lm.face_landmarks:
                        for face_landmarks in result_lm.face_landmarks:
                            rot_pts_all = []
                            for lm in face_landmarks:
                                lx = getattr(lm, 'x', lm['x'] if isinstance(lm, dict) else 0)
                                ly = getattr(lm, 'y', lm['y'] if isinstance(lm, dict) else 0)
                                pt_x = cx_off + (lx * cw * curr_scale) / curr_scale
                                pt_y = cy_off + (ly * ch * curr_scale) / curr_scale
                                rot_pts_all.append((pt_x, pt_y))
                            
                            orig_pts_all = map_points_back(rot_pts_all, M_inv)
                            if len(orig_pts_all) == 0:
                                continue

                            abs_cx = sum(p[0] for p in orig_pts_all) / len(orig_pts_all) / orig_w
                            abs_cy = sum(p[1] for p in orig_pts_all) / len(orig_pts_all) / orig_h

                            if is_duplicate(abs_cx, abs_cy):
                                continue

                            def process_target(indices, target_type):
                                target_pts = [orig_pts_all[i] for i in indices if i < len(orig_pts_all)]
                                if not target_pts:
                                    return
                                x_coords = [p[0] for p in target_pts]
                                y_coords = [p[1] for p in target_pts]
                                xmin, xmax = min(x_coords), max(x_coords)
                                ymin, ymax = min(y_coords), max(y_coords)

                                w_box, h_box = xmax - xmin, ymax - ymin
                                if w_box < 2 or h_box < 2:
                                    return

                                polygon = get_polygon_from_pts(target_pts)
                                processed_centers.append((abs_cx, abs_cy))
                                detected_items.append({
                                    "box": (xmin, ymin, w_box, h_box),
                                    "polygon": polygon,
                                    "type": target_type
                                })

                            if "顔全体" in mask_targets:
                                process_target(range(0, min(468, len(orig_pts_all))), "face")
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
                except Exception:
                    pass

            # --- B. OpenCV Cascade (遠距離・小顔補完エンジン) ---
            if face_cascade is not None and not face_cascade.empty():
                try:
                    gray_crop = cv2.cvtColor(scaled_crop, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray_crop,
                        scaleFactor=1.08,
                        minNeighbors=3,
                        minSize=(8, 8)
                    )
                    for (fx, fy, fw, fh) in faces:
                        orig_bx = cx_off + (fx / curr_scale)
                        orig_by = cy_off + (fy / curr_scale)
                        orig_bw = fw / curr_scale
                        orig_bh = fh / curr_scale

                        center_pt = (orig_bx + orig_bw / 2, orig_by + orig_bh / 2)
                        abs_cx = center_pt[0] / orig_w
                        abs_cy = center_pt[1] / orig_h

                        if is_duplicate(abs_cx, abs_cy):
                            continue

                        axes = (max(1, int(orig_bw / 2)), max(1, int(orig_bh * 0.6)))
                        ellipse_pts = cv2.ellipse2Poly((int(center_pt[0]), int(center_pt[1])), axes, 0, 0, 360, 15)
                        orig_polygon = map_points_back(ellipse_pts, M_inv)

                        processed_centers.append((abs_cx, abs_cy))
                        detected_items.append({
                            "box": (int(orig_bx), int(orig_by), int(orig_bw), int(orig_bh)),
                            "polygon": orig_polygon,
                            "type": "face"
                        })
                except Exception:
                    pass

    if len(detected_items) > 150:
        detected_items = detected_items[:150]

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

    # --- タイル状モザイクの場合 ---
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

    # --- 通常マスキングスタイル ---
    for item in items:
        bx, by, bw, bh = item["box"]
        polygon = item["polygon"]

        left = max(0, min(img_w - 1, int(bx)))
        top = max(0, min(img_h - 1, int(by)))
        right = max(0, min(img_w, int(bx + bw)))
        bottom = max(0, min(img_h, int(by + bh)))
        box_w, box_h = right - left, bottom - top

        if box_w <= 0 or box_h <= 0:
            continue

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
            if emoji_angle != 0:
                emoji_img = emoji_img.rotate(-emoji_angle, expand=True, resample=Image.Resampling.BICUBIC)
            ew, eh = emoji_img.size
            result_img.paste(emoji_img, (int(cx - ew / 2 + offset_x), int(cy - eh / 2 + offset_y)), emoji_img)
            continue

        # 合成処理
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

    if st.button("🚀 ローカル AI で限界突破解析を開始", type="primary"):
        if not mask_targets:
            st.warning("⚠️ 「マスク対象」を少なくとも1つ選択してください。")
        else:
            if precision_level == "限界突破（スクランブル交差点・高密度150人）":
                st.info("⚠️ 【限界突破モード】 多層分割 ＋ OpenCV補完スキャンで全自動モザイク処理を行います。")
            
            with st.spinner("AIがフル解析中..."):
                try:
                    st.session_state.boxes = get_mask_boxes_locally(
                        input_image,
                        mask_targets,
                        grids=grids,
                        scale_factor=scale_up_factor,
                        dup_thresh=dup_thresh,
                        apply_enhance=use_clahe
                    )
                    st.session_state.confirmed = False
                    if not st.session_state.boxes:
                        st.info("顔が検出されませんでした。別の画像をお試しください。")
                    else:
                        st.success(f"解析成功！ 「{precision_level}」モードで {len(st.session_state.boxes)} 箇所の部位を検出しました。")
                except Exception as e:
                    st.error(f"解析エラーが発生しました: {e}\n\n{traceback.format_exc()}")

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
                    file_name="crowd_masked.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary",
                )
else:
    st.info("👆 写真ファイルをアップロードして「🚀 ローカル AI で限界突破解析を開始」をクリックしてください。")