import io
import os
import math
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageEnhance
import traceback

# -------------------------------------------------------------------
# 1. ページ基本設定
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Masking Pro", page_icon="🛡️", layout="wide"
)

st.title("🛡️ AI Smart Masking Pro")
st.caption("通信なし・完全オフライン動作。高精度モード自動コントラスト補正搭載。")

# -------------------------------------------------------------------
# 2. ローカルモデルファイルのロード
# -------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

if not os.path.exists(FACE_LANDMARKER_PATH):
    st.error(f"エラー: `{FACE_LANDMARKER_PATH}` が見つかりません。`mp_models` フォルダ内に `face_landmarker.task` を配置してください。")
    st.stop()

@st.cache_resource
def load_mediapipe_tasks():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=50,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return mp, landmarker
    except Exception as e:
        st.error(f"初期化エラー: {e}")
        return None, None

mp, landmarker = load_mediapipe_tasks()

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
st.sidebar.success("🔒 完全ローカルモード")

# 🎯 検出精度選択（コントラスト補正と自動連動）
st.sidebar.subheader("🔍 検出精度")
precision_level = st.sidebar.radio(
    "精度を選択してください",
    [
        "標準（1人〜少人数）",
        "高精度（集合写真・暗所補正連動）",
        "超高精度（大人数・広域連動）",
    ],
    index=1,
)

# 選択肢に応じて分割数とコントラスト補正を自動割り振り
if precision_level == "標準（1人〜少人数）":
    grids = [1]
    enable_contrast = False
elif precision_level == "高精度（集合写真・暗所補正連動）":
    grids = [1, 2]
    enable_contrast = True
else:  # 超高精度
    grids = [1, 2, 3]
    enable_contrast = True

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
# 4. Pure Python / Pillow 描画ユーティリティ
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
    """Convex Hull（輪郭外包ポリゴン）計算"""
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


def apply_pillow_contrast(pil_img):
    """Pillowによるオートコントラスト＆明暗補正"""
    try:
        auto_img = ImageOps.autocontrast(pil_img, cutoff=2)
        enhancer = ImageEnhance.Contrast(auto_img)
        return enhancer.enhance(1.25)
    except Exception:
        return pil_img


# -------------------------------------------------------------------
# 5. 顔検出処理（精度選択と補正が完全連動）
# -------------------------------------------------------------------
def get_mask_boxes_locally(image: Image.Image, mask_targets: list, grids: list, enable_contrast: bool = False):
    if landmarker is None:
        st.error("FaceLandmarkerの初期化に失敗しています。")
        return []

    orig_w, orig_h = image.size
    detected_items = []
    processed_centers = []

    # 各パーツのインデックス
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
        overlap_x = int(step_x * 0.25)
        overlap_y = int(step_y * 0.25)

        for i in range(g):
            for j in range(g):
                x1 = max(0, i * step_x - overlap_x)
                y1 = max(0, j * step_y - overlap_y)
                x2 = min(orig_w, (i + 1) * step_x + overlap_x)
                y2 = min(orig_h, (j + 1) * step_y + overlap_y)
                crops.append((x1, y1, x2 - x1, y2 - y1))

    dup_pixel_thresh = min(orig_w, orig_h) * 0.05

    for cx_off, cy_off, cw, ch in crops:
        if cw < 20 or ch < 20:
            continue

        crop_pil = image.crop((cx_off, cy_off, cx_off + cw, cy_off + ch))

        # 精度設定に連動してコントラスト補正を自動適用
        if enable_contrast:
            scan_crop_pil = apply_pillow_contrast(crop_pil)
        else:
            scan_crop_pil = crop_pil

        crop_np = np.array(scan_crop_pil)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_np))

        try:
            result = landmarker.detect(mp_image)
            if result and result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    orig_pts_all = []
                    for lm in face_landmarks:
                        lx = getattr(lm, 'x', lm['x'] if isinstance(lm, dict) else 0)
                        ly = getattr(lm, 'y', lm['y'] if isinstance(lm, dict) else 0)
                        orig_pts_all.append((cx_off + lx * cw, cy_off + ly * ch))

                    if len(orig_pts_all) < 10:
                        continue

                    x_coords = [p[0] for p in orig_pts_all]
                    y_coords = [p[1] for p in orig_pts_all]
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    w_box, h_box = xmax - xmin, ymax - ymin

                    if w_box < 10 or h_box < 10:
                        continue

                    center_x = xmin + w_box / 2
                    center_y = ymin + h_box / 2
                    is_dup = False
                    for pcx, pcy in processed_centers:
                        if math.hypot(center_x - pcx, center_y - pcy) < dup_pixel_thresh:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    processed_centers.append((center_x, center_y))

                    def process_target(indices, target_type):
                        target_pts = [orig_pts_all[i] for i in indices if i < len(orig_pts_all)]
                        if not target_pts:
                            return
                        tx = [p[0] for p in target_pts]
                        ty = [p[1] for p in target_pts]
                        tw, th = max(tx) - min(tx), max(ty) - min(ty)
                        if tw < 3 or th < 3:
                            return
                        poly = get_polygon_hull_pil(target_pts)
                        detected_items.append({"box": (min(tx), min(ty), tw, th), "polygon": poly, "type": target_type})

                    if "顔全体" in mask_targets:
                        poly_all = get_polygon_hull_pil(orig_pts_all)
                        detected_items.append({"box": (xmin, ymin, w_box, h_box), "polygon": poly_all, "type": "face"})
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
    "加工したい写真をアップロードしてください (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.file_id != current_file_id:
        st.session_state.file_id = current_file_id
        st.session_state.boxes = None
        st.session_state.confirmed = False

    input_image = Image.open(uploaded_file).convert("RGB")

    if st.button("🚀 AIモザイク解析を開始", type="primary"):
        if not mask_targets:
            st.warning("⚠️ 「マスク対象」を少なくとも1つ選択してください。")
        else:
            with st.spinner("顔を検出中..."):
                try:
                    st.session_state.boxes = get_mask_boxes_locally(
                        input_image,
                        mask_targets,
                        grids=grids,
                        enable_contrast=enable_contrast,
                    )
                    st.session_state.confirmed = False
                    if not st.session_state.boxes:
                        st.info("顔が検出されませんでした。「高精度」や「超高精度」モードに変更してお試しください。")
                    else:
                        st.success(f"解析成功！ {len(st.session_state.boxes)} 箇所の顔領域を正しく検出しました。")
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