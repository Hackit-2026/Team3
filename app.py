import io
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
st.caption("通信なし・完全オフライン動作。最大150人の高密度検知 ＆ 壁・天井への誤判定防止フィルター搭載。")


# --- MediaPipe 標準モジュールの安全ロード ---
@st.cache_resource
def load_mediapipe_solutions():
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh, mp.solutions.face_detection
    except Exception as e:
        st.error(f"MediaPipeの読み込みに失敗しました: {e}")
        return None, None


mp_face_mesh, mp_face_detection = load_mediapipe_solutions()

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
        "高精度（集合写真・少人数）",
        "超高精度（大人数・上限150人）",
        "限界突破（スクランブル交差点・高密度150人）",
    ],
    index=3,
    help="「限界突破」は壁や天井の誤検知を防ぎつつ、最大150人までの極小顔をハイパーピラミッドスキャンで検出します。"
)

# モードごとのパラメータ設計（検出上限150人＆誤検知防止チューニング）
if precision_level == "普通（標準バランス）":
    scan_mode_key = "標準"
    grid_levels = [1, 2]
    scale_up_factor = 1.0
    conf_threshold = 0.35
    min_face_size = 15
    dup_thresh = 0.04
    max_faces_limit = 50
    use_clahe = False
elif precision_level == "高精度（集合写真・少人数）":
    scan_mode_key = "高精度"
    grid_levels = [1, 2, 3]
    scale_up_factor = 1.3
    conf_threshold = 0.30
    min_face_size = 10
    dup_thresh = 0.02
    max_faces_limit = 100
    use_clahe = False
elif precision_level == "超高精度（大人数・上限150人）":
    scan_mode_key = "超高精度"
    grid_levels = [1, 2, 4, 5]
    scale_up_factor = 1.8
    conf_threshold = 0.25
    min_face_size = 6
    dup_thresh = 0.01
    max_faces_limit = 150  # 150人上限
    use_clahe = True
else:  # 限界突破 (スクランブル交差点・高密度150人)
    scan_mode_key = "群衆特化"
    grid_levels = [1, 2, 4, 6]    # ピラミッドスキャン
    scale_up_factor = 2.2         # 超拡大
    conf_threshold = 0.22         # 壁・天井のノイズを拾わない安全なしきい値
    min_face_size = 5             # 5px以下の点状ノイズを除外
    dup_thresh = 0.008
    max_faces_limit = 150         # 150人上限設定
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
# 3. 安全な幾何・画像処理ユーティリティ
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
    """暗部や影のコントラストを強調"""
    try:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
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
    """型安全な座標逆変換"""
    clean_pts = []
    if isinstance(points, np.ndarray):
        try:
            pts_2d = points.reshape(-1, 2)
            for p in pts_2d:
                clean_pts.append((float(p[0]), float(p[1])))
        except Exception:
            pass 
    else:
        for p in points:
            if isinstance(p, (tuple, list, np.ndarray)) and len(p) >= 2:
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
        polygon = []
        if hull is not None:
            try:
                hull_flat = hull.reshape(-1, 2)
                for pt in hull_flat:
                    polygon.append((int(pt[0]), int(pt[1])))
            except Exception:
                polygon = [(int(p[0]), int(p[1])) for p in target_pts]
        if not polygon:
            polygon = [(int(p[0]), int(p[1])) for p in target_pts]
        return polygon
    except Exception:
        return [(int(p[0]), int(p[1])) for p in target_pts]


# -------------------------------------------------------------------
# 4. マルチスキャン検出エンジン (誤判定防止フィルター搭載)
# -------------------------------------------------------------------
def get_mask_boxes_locally(
    image: Image.Image,
    mask_targets: list,
    scan_key: str,
    grids: list,
    scale_factor: float = 1.4,
    conf_thresh: float = 0.35,
    min_size: int = 12,
    dup_thresh: float = 0.02,
    max_faces: int = 150,
    apply_enhance: bool = False
):
    if mp_face_mesh is None or mp_face_detection is None:
        st.error("MediaPipeが正常に読み込まれていません。")
        return []

    original_np = np.array(image)
    processed_np = apply_clahe(original_np) if apply_enhance else original_np
    orig_h, orig_w, _ = processed_np.shape

    detected_items = []
    processed_centers = []

    INDEX_RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    INDEX_LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    INDEX_NOSE = [1, 2, 98, 327, 278, 48]
    INDEX_MOUTH = [61, 291, 37, 267, 0, 17, 18, 14, 87, 317]

    if scan_key == "群衆特化":
        angles = [0, 15, -15]
    elif scan_key == "超高精度":
        angles = [0, 15, -15]
    else:
        angles = [0]

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

        # --- 1. FaceMesh 精密検出 ---
        try:
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,  # 最大150名
                refine_landmarks=True,
                min_detection_confidence=conf_thresh,
            ) as face_mesh:

                for cx_off, cy_off, cw, ch in crops:
                    crop_img = rotated_np[cy_off:cy_off+ch, cx_off:cx_off+cw]
                    if crop_img.shape[0] < 15 or crop_img.shape[1] < 15:
                        continue

                    if scale_factor > 1.0 and (cw * scale_factor < 4000 and ch * scale_factor < 4000):
                        scaled_crop = cv2.resize(
                            crop_img, 
                            (int(cw * scale_factor), int(ch * scale_factor)), 
                            interpolation=cv2.INTER_CUBIC
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

                            # -------------------------------------------------------------
                            # 🛡️ 誤判定防止（壁・天井の模様除外）フィルター
                            # -------------------------------------------------------------
                            # 両目と口の位置関係チェック（目が上、口が下にあるか）
                            if len(orig_pts_all) >= 468:
                                eye_r_y = (orig_pts_all[33][1] + orig_pts_all[133][1]) / 2
                                eye_l_y = (orig_pts_all[362][1] + orig_pts_all[263][1]) / 2
                                mouth_y = (orig_pts_all[61][1] + orig_pts_all[291][1]) / 2
                                eye_avg_y = (eye_r_y + eye_l_y) / 2
                                # 目より口が上にある不自然な構造（天井のライト等）を除外
                                if mouth_y <= eye_avg_y - 2:
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
                                
                                # 最小サイズチェック
                                if w_box < min_size or h_box < min_size:
                                    return
                                
                                # アスペクト比チェック（縦横比 0.45〜2.2 以外の細長い壁の目地やライトを除外）
                                if w_box > 0 and h_box > 0:
                                    aspectRatio = w_box / h_box
                                    if aspectRatio > 2.2 or aspectRatio < 0.45:
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
        except Exception:
            pass

        # --- 2. 遠距離専用 Face Detection バックアップ ---
        try:
            with mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=conf_thresh + 0.05 # バックアップは少し厳しめに設定して誤検知カット
            ) as face_detector:

                for cx_off, cy_off, cw, ch in crops:
                    crop_img = rotated_np[cy_off:cy_off+ch, cx_off:cx_off+cw]
                    if crop_img.shape[0] < 15 or crop_img.shape[1] < 15:
                        continue

                    if scale_factor > 1.0 and (cw * scale_factor < 4000 and ch * scale_factor < 4000):
                        scaled_crop = cv2.resize(
                            crop_img, 
                            (int(cw * scale_factor), int(ch * scale_factor)), 
                            interpolation=cv2.INTER_CUBIC
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

                            # アスペクト比判定（壁や天井の四角を除外）
                            if rw > 0 and rh > 0:
                                aspectRatio = rw / rh
                                if aspectRatio > 2.0 or aspectRatio < 0.5:
                                    continue

                            center_pt = (rx + rw // 2, ry + rh // 2)
                            axes = (max(1, int(rw // 2)), max(1, int(rh * 0.6)))
                            ellipse_pts = cv2.ellipse2Poly((int(center_pt[0]), int(center_pt[1])), axes, 0, 0, 360, 15)
                            
                            orig_polygon = map_points_back(ellipse_pts, M_inv)
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
        except Exception:
            pass

    # 150名上限制限
    if len(detected_items) > max_faces_limit:
        detected_items = detected_items[:max_faces_limit]

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
                st.info("⚠️ 【限界突破モード】 画像を最大64分割し、50%重ね合わせながら超拡大スキャンを行います（最大150人まで高精度検知）。")
            
            with st.spinner("AIが指定された精度モードで画像をフル解析中..."):
                try:
                    st.session_state.boxes = get_mask_boxes_locally(
                        input_image,
                        mask_targets,
                        scan_key=scan_mode_key,
                        grids=grid_levels,
                        scale_factor=scale_up_factor,
                        conf_thresh=conf_threshold,
                        min_size=min_face_size,
                        dup_thresh=dup_thresh,
                        max_faces=max_faces_limit,
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