import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# -------------------------------------------------------------------
# 1. ページ基本設定
# -------------------------------------------------------------------
st.set_page_config(
    page_title="完全ローカル AI マスキング", page_icon="🛡️", layout="wide"
)

st.title("🛡️ 完全ローカル AI スマートマスキング WebApp")
st.caption("通信なし・完全オフライン動作。全自動AIスキャンで顔・目・パーツを精密検出します。")


# --- MediaPipe モジュールの安全ロード ---
@st.cache_resource
def load_mediapipe_models():
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        import mediapipe.python.solutions.face_detection as mp_face_detection
        return mp_face_mesh, mp_face_detection
    except Exception:
        try:
            import mediapipe as mp
            return mp.solutions.face_mesh, mp.solutions.face_detection
        except Exception as e:
            st.error(f"MediaPipeの読み込みに失敗しました: {e}")
            return None, None


mp_face_mesh, mp_face_detection = load_mediapipe_models()

# セッション状態の初期化
if "boxes" not in st.session_state:
    st.session_state.boxes = None
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# -------------------------------------------------------------------
# 2. サイドバー設定
# -------------------------------------------------------------------
st.sidebar.header("⚙️ マスク設定")
st.sidebar.success("🔒 完全ローカルモード (Wi-FiオフOK)")

# 🎯 検出精度モード
st.sidebar.subheader("🔍 検出精度")
precision_level = st.sidebar.radio(
    "精度モードを選択",
    [
        "普通（標準バランス）",
        "高精度（集合写真・誤爆防止）",
        "超高精度（極小・斜め顔特化）",
    ],
    index=1,
    help="「高精度」は背景の誤爆を防ぎつつ集合写真を正確に検出します。"
)

# モードごとのパラメータ設計
if precision_level == "普通（標準バランス）":
    scan_mode_key = "標準"
    max_grid_count = 2
    scale_up_factor = 1.2
    conf_threshold = 0.35
    min_face_size = 15
elif precision_level == "高精度（集合写真・誤爆防止）":
    scan_mode_key = "高精度"
    max_grid_count = 3
    scale_up_factor = 1.4
    conf_threshold = 0.35
    min_face_size = 12
else:  # 超高精度
    scan_mode_key = "超高精度"
    max_grid_count = 4
    scale_up_factor = 1.8
    conf_threshold = 0.30
    min_face_size = 8

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
# 3. 100%安全な幾何・座標変換処理（行列エラー修正済み）
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


def get_rotated_image_and_inv_matrix(image_np, angle):
    """回転画像と逆変換行列を生成"""
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
    
    # 修正: OpenCVのinvertAffineTransformは1つの値（2x3行列）を返すため正しく受け取る
    M_inv = cv2.invertAffineTransform(M)
    
    return rotated_img, M_inv


def map_points_back(points, M_inv):
    """どのような型のデータが入ってきても型安全に(x, y)タプルリストへ変換してアフィン逆変換"""
    clean_pts = []
    if isinstance(points, np.ndarray):
        pts_2d = points.reshape(-1, 2)
        for p in pts_2d:
            clean_pts.append((float(p[0]), float(p[1])))
    else:
        for p in points:
            if isinstance(p, (tuple, list, np.ndarray)) and len(p) >= 2:
                clean_pts.append((float(p[0]), float(p[1])))

    if len(clean_pts) == 0:
        return []

    if M_inv is None:
        return [(int(p[0]), int(p[1])) for p in clean_pts]

    # 行列（2次元配列）から安全に値を取得
    m00, m01, m02 = float(M_inv[0, 0]), float(M_inv[0, 1]), float(M_inv[0, 2])
    m10, m11, m12 = float(M_inv[1, 0]), float(M_inv[1, 1]), float(M_inv[1, 2])

    res = []
    for x, y in clean_pts:
        x_orig = int(m00 * x + m01 * y + m02)
        y_orig = int(m10 * x + m11 * y + m12)
        res.append((x_orig, y_orig))
    return res


def get_polygon_from_pts(target_pts):
    """ConvexHullを安全に計算"""
    if not target_pts:
        return []
    if len(target_pts) < 3:
        return [(int(p[0]), int(p[1])) for p in target_pts]

    try:
        pts_np = np.array(target_pts, dtype=np.int32).reshape(-1, 2)
        hull = cv2.convexHull(pts_np)
        polygon = []
        if hull is not None:
            hull_flat = hull.reshape(-1, 2)
            for pt in hull_flat:
                polygon.append((int(pt[0]), int(pt[1])))
        if not polygon:
            polygon = [(int(p[0]), int(p[1])) for p in target_pts]
        return polygon
    except Exception:
        return [(int(p[0]), int(p[1])) for p in target_pts]


# -------------------------------------------------------------------
# 4. マルチスキャン検出エンジン
# -------------------------------------------------------------------
def get_mask_boxes_locally(
    image: Image.Image,
    mask_targets: list,
    scan_key: str,
    max_grid: int = 3,
    scale_factor: float = 1.4,
    conf_thresh: float = 0.35,
    min_size: int = 12
):
    if mp_face_mesh is None or mp_face_detection is None:
        st.error("MediaPipeが正常に読み込まれていません。")
        return []

    original_np = np.array(image)
    orig_h, orig_w, _ = original_np.shape
    detected_items = []
    processed_centers = []

    INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    INDEX_NOSE = [1, 2, 98, 327, 278, 48]
    INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]

    # 服のシワなどの誤爆を防ぐ自然な角度制限
    if scan_key == "超高精度":
        angles = [0, 15, -15, 30, -30, 45, -45]
    elif scan_key == "高精度":
        angles = [0, 20, -20]
    else:
        angles = [0]

    def is_duplicate(cx, cy):
        for pcx, pcy in processed_centers:
            if abs(pcx - cx) < 0.04 and abs(pcy - cy) < 0.04:
                return True
        return False

    for angle in angles:
        rotated_np, M_inv = get_rotated_image_and_inv_matrix(original_np, angle)
        rot_h, rot_w, _ = rotated_np.shape

        crops = [(0, 0, rot_w, rot_h)]

        for g in range(2, max_grid + 1):
            step_x = rot_w // g
            step_y = rot_h // g
            overlap_x = int(step_x * 0.3)
            overlap_y = int(step_y * 0.3)

            for i in range(g):
                for j in range(g):
                    x1 = max(0, i * step_x - overlap_x)
                    y1 = max(0, j * step_y - overlap_y)
                    x2 = min(rot_w, (i + 1) * step_x + overlap_x)
                    y2 = min(rot_h, (j + 1) * step_y + overlap_y)
                    crops.append((x1, y1, x2 - x1, y2 - y1))

        # --- 1. FaceMesh 精密検出 ---
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=conf_thresh,
        ) as face_mesh:

            for cx_off, cy_off, cw, ch in crops:
                crop_img = rotated_np[cy_off:cy_off+ch, cx_off:cx_off+cw]
                if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                    continue

                if scale_factor > 1.0 and (cw * scale_factor < 3500 and ch * scale_factor < 3500):
                    scaled_crop = cv2.resize(
                        crop_img, 
                        (int(cw * scale_factor), int(ch * scale_factor)), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    curr_scale = scale_factor
                else:
                    scaled_crop = crop_img
                    curr_scale = 1.0

                results = face_mesh.process(scaled_crop)
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        rot_pts_all = [
                            (
                                cx_off + (lm.x * cw * curr_scale) / curr_scale, 
                                cy_off + (lm.y * ch * curr_scale) / curr_scale
                            ) 
                            for lm in landmarks
                        ]
                        
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
                            if w_box < min_size or h_box < min_size:
                                return

                            polygon = get_polygon_from_pts(target_pts)

                            processed_centers.append((abs_cx, abs_cy))
                            detected_items.append({
                                "box": (xmin, ymin, w_box, h_box),
                                "polygon": polygon,
                                "type": target_type
                            })

                        if "顔全体" in mask_targets:
                            process_target(range(0, 468), "face")
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

        # --- 2. 遠距離専用 Face Detection バックアップ ---
        with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=conf_thresh
        ) as face_detector:

            for cx_off, cy_off, cw, ch in crops:
                crop_img = rotated_np[cy_off:cy_off+ch, cx_off:cx_off+cw]
                if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                    continue

                if scale_factor > 1.0 and (cw * scale_factor < 3500 and ch * scale_factor < 3500):
                    scaled_crop = cv2.resize(
                        crop_img, 
                        (int(cw * scale_factor), int(ch * scale_factor)), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    curr_scale = scale_factor
                else:
                    scaled_crop = crop_img
                    curr_scale = 1.0

                results = face_detector.process(scaled_crop)
                if results and results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        rx = cx_off + (bbox.xmin * cw * curr_scale) / curr_scale
                        ry = cy_off + (bbox.ymin * ch * curr_scale) / curr_scale
                        rw = (bbox.width * cw * curr_scale) / curr_scale
                        rh = (bbox.height * ch * curr_scale) / curr_scale

                        if rw < min_size or rh < min_size:
                            continue

                        center_pt = (rx + rw // 2, ry + rh // 2)
                        axes = (max(1, int(rw // 2)), max(1, int(rh * 0.6)))
                        ellipse_pts = cv2.ellipse2Poly((int(center_pt[0]), int(center_pt[1])), axes, 0, 0, 360, 15)
                        
                        pts_list = [(float(p[0]), float(p[1])) for p in ellipse_pts]
                        orig_polygon = map_points_back(pts_list, M_inv)
                        if len(orig_polygon) == 0:
                            continue
                        
                        x_coords = [p[0] for p in orig_polygon]
                        y_coords = [p[1] for p in orig_polygon]
                        xmin, xmax = min(x_coords), max(x_coords)
                        ymin, ymax = min(y_coords), max(y_coords)

                        abs_cx = (xmin + xmax) / 2 / orig_w
                        abs_cy = (ymin + ymax) / 2 / orig_h

                        if is_duplicate(abs_cx, abs_cy):
                            continue
                        processed_centers.append((abs_cx, abs_cy))

                        detected_items.append({
                            "box": (xmin, ymin, xmax - xmin, ymax - ymin),
                            "polygon": orig_polygon,
                            "type": "face"
                        })

    return detected_items


# -------------------------------------------------------------------
# 5. マスキング画像適用処理
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
    result_img = image.copy()
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
        right = max(0, min(img_w - 1, int(bx + bw)))
        bottom = max(0, min(img_h - 1, int(by + bh)))
        box_w, box_h = right - left, bottom - top

        if box_w <= 0 or box_h <= 0:
            continue

        cx, cy = left + box_w / 2, top + box_h / 2

        masked_layer = result_img.copy()

        if mask_type == "ピクセルモザイク":
            box_region = result_img.crop((left, top, right, bottom))
            small_box = box_region.resize((max(1, box_w // mosaic_size), max(1, box_h // mosaic_size)), resample=Image.Resampling.NEAREST)
            mosaic_box = small_box.resize((box_w, box_h), resample=Image.Resampling.NEAREST)
            masked_layer.paste(mosaic_box, (left, top))

        elif mask_type == "ぼかし（ブラー）":
            box_region = result_img.crop((left, top, right, bottom))
            masked_layer.paste(box_region.filter(ImageFilter.GaussianBlur(radius=blur_radius)), (left, top))

        elif mask_type == "塗りつぶし（カラー指定）":
            draw_layer = ImageDraw.Draw(masked_layer)
            draw_layer.rectangle([left, top, right, bottom], fill=fill_color)

        elif mask_type == "絵文字スタンプ":
            target_size = max(int(max(box_w, box_h) * 0.9), min(int(max(box_w, box_h) * 1.15 * (emoji_scale / 100.0)), int(max(box_w, box_h) * 2.2)))
            emoji_img = create_cropped_emoji_image(emoji_char, target_size=target_size)
            if emoji_angle != 0:
                emoji_img = emoji_img.rotate(-emoji_angle, expand=True, resample=Image.Resampling.BICUBIC)
            ew, eh = emoji_img.size
            result_img.paste(emoji_img, (int(cx - ew / 2 + offset_x), int(cy - eh / 2 + offset_y)), emoji_img)
            continue

        if mask_shape == "顔の形（輪郭に沿う）" and len(polygon) >= 3:
            mask_img = Image.new("L", (img_w, img_h), 0)
            draw_mask = ImageDraw.Draw(mask_img)
            draw_mask.polygon(polygon, fill=255)
            result_img = Image.composite(masked_layer, result_img, mask_img)
        else:
            result_img = masked_layer

    return result_img


# -------------------------------------------------------------------
# 6. メイン画面レイアウト
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "画像をアップロードしてください (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.file_id != current_file_id:
        st.session_state.file_id = current_file_id
        st.session_state.boxes = None
        st.session_state.confirmed = False

    input_image = Image.open(uploaded_file).convert("RGB")

    if st.button("🚀 ローカル AI で解析を実行", type="primary"):
        if not mask_targets:
            st.warning("⚠️ 「マスク対象」を少なくとも1つ選択してください。")
        else:
            with st.spinner("AIが指定された精度モードで画像を解析中..."):
                try:
                    st.session_state.boxes = get_mask_boxes_locally(
                        input_image,
                        mask_targets,
                        scan_key=scan_mode_key,
                        max_grid=max_grid_count,
                        scale_factor=scale_up_factor,
                        conf_thresh=conf_threshold,
                        min_size=min_face_size
                    )
                    st.session_state.confirmed = False
                    if not st.session_state.boxes:
                        st.info("顔が検出されませんでした。別の画像をお試しください。")
                    else:
                        st.success(f"解析成功！ 「{precision_level}」モードで {len(st.session_state.boxes)} 箇所の部位を検出しました。")
                except Exception as e:
                    st.error(f"解析エラー: {e}")

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
            st.subheader("📷 元の画像 (比較用)")
            st.image(input_image, use_container_width=True)
        with col_right:
            st.subheader("🎉 確定した画像" if st.session_state.confirmed else "🔍 リアルタイム調整プレビュー")
            st.image(processed_image, use_container_width=True)

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
                    label="💾 確定画像を保存 (ダウンロード)",
                    data=buf.getvalue(),
                    file_name="masked_image.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary",
                )
else:
    st.info("👆 画像ファイルをアップロードして「🚀 ローカル AI で解析を実行」ボタンを押してください。")