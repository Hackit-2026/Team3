# =====================================================
# AI Smart Masking Pro
# =====================================================

import io
import os
import math
import traceback

import cv2
import numpy as np
import streamlit as st

from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont
)

# =====================================================
# APP CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Smart Masking Pro",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Smart Masking Pro")

# =====================================================
# MODEL PATHS
# =====================================================

MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "mp_models"
)

FACE_DETECTOR_PATH = os.path.join(
    MODEL_DIR,
    "blaze_face_full_range.tflite"
)

FACE_LANDMARKER_PATH = os.path.join(
    MODEL_DIR,
    "face_landmarker.task"
)

# =====================================================
# GLOBAL SETTINGS
# =====================================================

RIGHT_EYE = []
LEFT_EYE = []
NOSE = []
MOUTH = []

# =====================================================
# ADD HERE
# =====================================================

# =====================================================
# MEDIAPIPE LOADER
# =====================================================

@st.cache_resource
def load_mediapipe():

    try:

        import mediapipe as mp

        from mediapipe.tasks import (
            python as mp_python
        )

        from mediapipe.tasks.python import (
            vision
        )

        return (
            mp,
            mp_python,
            vision
        )

    except Exception:

        st.error(
            traceback.format_exc()
        )

        st.stop()


mp, mp_python, mp_vision = (
    load_mediapipe()
)

# =====================================================
# MODEL CHECK
# =====================================================

if not os.path.exists(
    FACE_DETECTOR_PATH
):

    st.error(
        "blaze_face_full_range.tflite がありません"
    )

    st.stop()


if not os.path.exists(
    FACE_LANDMARKER_PATH
):

    st.error(
        "face_landmarker.task がありません"
    )

    st.stop()

# =====================================================
# DETECTOR
# =====================================================

@st.cache_resource
def create_detector():

    options = (
        mp_vision.FaceDetectorOptions(
            base_options=
            mp_python.BaseOptions(
                model_asset_path=
                FACE_DETECTOR_PATH
            ),

            running_mode=
            mp_vision.RunningMode.IMAGE,

            min_detection_confidence=0.25
        )
    )

    return (
        mp_vision.FaceDetector
        .create_from_options(
            options
        )
    )

# =====================================================
# LANDMARKER
# =====================================================

@st.cache_resource
def create_landmarker():

    options = (
        mp_vision.FaceLandmarkerOptions(

            base_options=
            mp_python.BaseOptions(
                model_asset_path=
                FACE_LANDMARKER_PATH
            ),

            running_mode=
            mp_vision.RunningMode.IMAGE,

            num_faces=300,

            min_face_detection_confidence=
            0.20,

            min_face_presence_confidence=
            0.20,

            min_tracking_confidence=
            0.20,

            output_face_blendshapes=
            False,

            output_facial_transformation_matrixes=
            False
        )
    )

    return (
        mp_vision.FaceLandmarker
        .create_from_options(
            options
        )
    )


detector = (
    create_detector()
)

landmarker = (
    create_landmarker()
)

# =====================================================
# MEDIAPIPE IMAGE HELPER
# =====================================================

def to_mp_image(image):

    image_np = np.array(
        image
    )

    return mp.Image(
        image_format=
        mp.ImageFormat.SRGB,
        data=
        np.ascontiguousarray(
            image_np
        )
    )
# =====================================================
# FACE DETECTOR ENGINE
# =====================================================

def detect_faces_detector(
    image,
    confidence=0.25
):

    results_boxes = []

    try:

        mp_image = to_mp_image(
            image
        )

        detection_result = (
            detector.detect(
                mp_image
            )
        )

        if (
            detection_result is None
            or
            detection_result.detections is None
        ):
            return []

        for detection in (
            detection_result.detections
        ):

            try:

                bbox = (
                    detection.bounding_box
                )

                x = int(
                    bbox.origin_x
                )

                y = int(
                    bbox.origin_y
                )

                w = int(
                    bbox.width
                )

                h = int(
                    bbox.height
                )

                if (
                    w <= 0
                    or
                    h <= 0
                ):
                    continue

                results_boxes.append(
                    (
                        x,
                        y,
                        w,
                        h
                    )
                )

            except Exception:
                continue

    except Exception:

        st.error(
            f"FaceDetectorエラー\n\n"
            f"{traceback.format_exc()}"
        )

    return results_boxes

# =====================================================
# FACE DETECTOR DRAW
# =====================================================

def draw_detector_boxes(
    image,
    boxes
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for (
        x,
        y,
        w,
        h
    ) in boxes:

        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h
            ],
            outline="red",
            width=3
        )

    return preview

# =====================================================
# FACE DETECTOR TEST
# =====================================================

def run_detector_preview(
    image
):

    boxes = detect_faces_detector(
        image
    )

    preview = draw_detector_boxes(
        image,
        boxes
    )

    st.image(
        preview,
        caption=
        f"FaceDetector 検出数 : {len(boxes)}"
    )

    return boxes
# =====================================================
# FACELANDMARKER ENGINE
# =====================================================

def detect_faces_landmarker(
    image
):

    faces = []

    try:

        mp_image = to_mp_image(
            image
        )

        result = (
            landmarker.detect(
                mp_image
            )
        )

        if (
            result is None
            or
            result.face_landmarks is None
        ):
            return []

        img_w = image.width
        img_h = image.height

        for face_landmarks in (
            result.face_landmarks
        ):

            pts = []

            for lm in face_landmarks:

                x = int(
                    lm.x * img_w
                )

                y = int(
                    lm.y * img_h
                )

                pts.append(
                    (x, y)
                )

            faces.append(
                pts
            )

    except Exception:

        st.error(
            f"FaceLandmarkerエラー\n\n"
            f"{traceback.format_exc()}"
        )

    return faces

# =====================================================
# LANDMARK TO BOX
# =====================================================

def landmark_to_box(
    face_points
):

    xs = [
        p[0]
        for p in face_points
    ]

    ys = [
        p[1]
        for p in face_points
    ]

    x1 = min(xs)
    y1 = min(ys)

    x2 = max(xs)
    y2 = max(ys)

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )

# =====================================================
# LANDMARK BOXES
# =====================================================

def landmark_boxes(
    landmark_faces
):

    boxes = []

    for face in landmark_faces:

        box = landmark_to_box(
            face
        )

        boxes.append(
            box
        )

    return boxes

# =====================================================
# LANDMARK DRAW
# =====================================================

def draw_landmarks(
    image,
    landmark_faces,
    point_radius=1
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for face in landmark_faces:

        for x, y in face:

            draw.ellipse(
                (
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius
                ),
                fill="lime"
            )

    return preview

# =====================================================
# LANDMARK TEST
# =====================================================

def run_landmark_preview(
    image
):

    faces = detect_faces_landmarker(
        image
    )

    preview = draw_landmarks(
        image,
        faces
    )

    st.image(
        preview,
        caption=
        f"Landmark Face Count : {len(faces)}"
    )

    return faces

# =====================================================
# FACE HULL POLYGON
# =====================================================

def create_face_polygon(
    face_points
):

    if (
        face_points is None
        or
        len(face_points) < 3
    ):
        return []

    pts_np = np.array(
        face_points,
        dtype=np.int32
    )

    hull = cv2.convexHull(
        pts_np
    )

    polygon = []

    for pt in (
        hull.reshape(-1, 2)
    ):

        polygon.append(
            (
                int(pt[0]),
                int(pt[1])
            )
        )

    return polygon

# =====================================================
# FACIAL FEATURE INDICES
# =====================================================

RIGHT_EYE = [
    33,
    133,
    160,
    159,
    158,
    144,
    145,
    153
]

LEFT_EYE = [
    362,
    263,
    387,
    386,
    385,
    373,
    374,
    380
]

NOSE = [
    1,
    2,
    98,
    327,
    278,
    48
]

MOUTH = [
    61,
    291,
    37,
    267,
    0,
    17,
    18,
    14,
    87,
    317
]

# =====================================================
# GENERIC FEATURE POLYGON
# =====================================================

def create_feature_polygon(
    face_points,
    indices
):

    pts = []

    for idx in indices:

        if idx < len(face_points):

            pts.append(
                face_points[idx]
            )

    if len(pts) < 3:

        return []

    pts_np = np.array(
        pts,
        dtype=np.int32
    )

    hull = cv2.convexHull(
        pts_np
    )

    polygon = []

    for pt in (
        hull.reshape(-1, 2)
    ):

        polygon.append(
            (
                int(pt[0]),
                int(pt[1])
            )
        )

    return polygon

# =====================================================
# FACE POLYGON
# =====================================================

def get_face_polygon(
    face_points
):

    return create_face_polygon(
        face_points
    )

# =====================================================
# RIGHT EYE
# =====================================================

def get_right_eye_polygon(
    face_points
):

    return create_feature_polygon(
        face_points,
        RIGHT_EYE
    )

# =====================================================
# LEFT EYE
# =====================================================

def get_left_eye_polygon(
    face_points
):

    return create_feature_polygon(
        face_points,
        LEFT_EYE
    )

# =====================================================
# BOTH EYES
# =====================================================

def get_both_eyes_polygons(
    face_points
):

    return [
        get_right_eye_polygon(
            face_points
        ),

        get_left_eye_polygon(
            face_points
        )
    ]

# =====================================================
# NOSE
# =====================================================

def get_nose_polygon(
    face_points
):

    return create_feature_polygon(
        face_points,
        NOSE
    )


def get_mouth_polygon(
    face_points
):

    return create_feature_polygon(
        face_points,
        MOUTH
    )

# =====================================================
# POLYGON MASK ENGINE
# =====================================================

def create_polygon_mask(
    image_size,
    polygon
):

    mask = Image.new(
        "L",
        image_size,
        0
    )

    draw = ImageDraw.Draw(
        mask
    )

    draw.polygon(
        polygon,
        fill=255
    )

    return mask

# =====================================================
# POLYGON MOSAIC
# =====================================================

def apply_polygon_mosaic(
    image,
    polygon,
    mosaic_size=15
):

    mask = create_polygon_mask(
        image.size,
        polygon
    )

    x, y, w, h = polygon_bbox(
        polygon
    )

    if (
        w <= 0
        or
        h <= 0
    ):
        return image

    result = image.copy()

    region = image.crop(
        (
            x,
            y,
            x + w,
            y + h
        )
    )

    small = region.resize(
        (
            max(1, w // mosaic_size),
            max(1, h // mosaic_size)
        ),
        Image.Resampling.NEAREST
    )

    mosaic_region = small.resize(
        (
            w,
            h
        ),
        Image.Resampling.NEAREST
    )

    mosaic_layer = image.copy()

    mosaic_layer.paste(
        mosaic_region,
        (x, y)
    )

    return Image.composite(
        mosaic_layer,
        result,
        mask
    )

# =====================================================
# POLYGON BLUR
# =====================================================

def apply_polygon_blur(
    image,
    polygon,
    blur_radius=20
):

    mask = create_polygon_mask(
        image.size,
        polygon
    )

    blurred = image.filter(
        ImageFilter.GaussianBlur(
            blur_radius
        )
    )

    return Image.composite(
        blurred,
        image,
        mask
    )

# =====================================================
# POLYGON FILL
# =====================================================

def apply_polygon_fill(
    image,
    polygon,
    fill_color="#000000"
):

    layer = image.copy()

    draw = ImageDraw.Draw(
        layer
    )

    draw.polygon(
        polygon,
        fill=fill_color
    )

    mask = create_polygon_mask(
        image.size,
        polygon
    )

    return Image.composite(
        layer,
        image,
        mask
    )

# =====================================================
# POLYGON CENTER
# =====================================================

def polygon_center(
    polygon
):

    xs = [
        p[0]
        for p in polygon
    ]

    ys = [
        p[1]
        for p in polygon
    ]

    return (
        int(sum(xs) / len(xs)),
        int(sum(ys) / len(ys))
    )

# =====================================================
# POLYGON MASK DISPATCHER
# =====================================================

def apply_polygon_mask(
    image,
    polygon,
    mask_type,
    mosaic_size=15,
    blur_radius=20,
    fill_color="#000000"
):

    if len(polygon) < 3:
        return image

    if mask_type == "モザイク":

        return apply_polygon_mosaic(
            image,
            polygon,
            mosaic_size
        )

    elif mask_type == "ブラー":

        return apply_polygon_blur(
            image,
            polygon,
            blur_radius
        )

    elif mask_type == "塗りつぶし":

        return apply_polygon_fill(
            image,
            polygon,
            fill_color
        )

    return image

# =====================================================
# MULTI POLYGON APPLY
# =====================================================

def apply_polygon_masks(
    image,
    polygons,
    mask_type,
    mosaic_size=15,
    blur_radius=20,
    fill_color="#000000"
):

    result = image.copy()

    for polygon in polygons:

        result = apply_polygon_mask(
            result,
            polygon,
            mask_type,
            mosaic_size,
            blur_radius,
            fill_color
        )

    return result

# =====================================================
# POLYGON DEBUG VIEW
# =====================================================

def debug_polygon_preview(
    image,
    polygons
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for polygon in polygons:

        draw.polygon(
            polygon,
            outline="lime",
            width=3
        )

    return preview

# =====================================================
# EMOJI ENGINE
# =====================================================

def get_emoji_font(
    font_size
):

    font_candidates = [

        "seguiemj.ttf",

        "NotoColorEmoji.ttf",

        "Apple Color Emoji.ttc",

        "arial.ttf"

    ]

    for font_name in font_candidates:

        try:

            return ImageFont.truetype(
                font_name,
                font_size
            )

        except Exception:
            pass

    return ImageFont.load_default()

# =====================================================
# CREATE EMOJI IMAGE
# =====================================================
def create_emoji_image(
    emoji_char,
    target_size
):

    canvas_size = max(
        256,
        target_size * 2
    )

    img = Image.new(
        "RGBA",
        (
            canvas_size,
            canvas_size
        ),
        (
            0,
            0,
            0,
            0
        )
    )

    draw = ImageDraw.Draw(
        img
    )

    font = get_emoji_font(
        int(
            canvas_size * 0.7
        )
    )

    try:

        draw.text(
            (
                canvas_size // 2,
                canvas_size // 2
            ),
            emoji_char,
            font=font,
            anchor="mm",
            embedded_color=True
        )

    except Exception:

        draw.text(
            (
                canvas_size // 2,
                canvas_size // 2
            ),
            emoji_char,
            font=font,
            anchor="mm",
            fill="black"
        )

    bbox = img.getbbox()

    if bbox:

        img = img.crop(
            bbox
        )

    return img.resize(
        (
            target_size,
            target_size
        ),
        Image.Resampling.LANCZOS
    )
# =====================================================
# TILE MOSAIC ENGINE
# =====================================================

def box_intersects_tile(
    box,
    tile
):

    bx, by, bw, bh = box

    tx1, ty1, tx2, ty2 = tile

    return (

        max(
            bx,
            tx1
        )

        <

        min(
            bx + bw,
            tx2
        )

    ) and (

        max(
            by,
            ty1
        )

        <

        min(
            by + bh,
            ty2
        )

    )

# =====================================================
# POLYGON TO BOX
# =====================================================

def polygon_to_box(
    polygon
):

    if len(polygon) < 3:

        return (
            0,
            0,
            0,
            0
        )

    xs = [
        p[0]
        for p in polygon
    ]

    ys = [
        p[1]
        for p in polygon
    ]

    return (
        min(xs),
        min(ys),
        max(xs) - min(xs),
        max(ys) - min(ys)
    )

# =====================================================
# TILE MOSAIC
# =====================================================

def apply_tile_mosaic(
    image,
    polygons,
    grid_size=30
):

    result = image.copy()

    img_w, img_h = image.size

    target_boxes = []

    for polygon in polygons:

        target_boxes.append(
            polygon_to_box(
                polygon
            )
        )

    for gx in range(
        0,
        img_w,
        grid_size
    ):

        for gy in range(
            0,
            img_h,
            grid_size
        ):

            tile_left = gx

            tile_top = gy

            tile_right = min(
                img_w,
                gx + grid_size
            )

            tile_bottom = min(
                img_h,
                gy + grid_size
            )

            tile = (
                tile_left,
                tile_top,
                tile_right,
                tile_bottom
            )

            hit = False

            for box in target_boxes:

                if box_intersects_tile(
                    box,
                    tile
                ):

                    hit = True
                    break

            if not hit:
                continue

            tile_region = result.crop(
                (
                    tile_left,
                    tile_top,
                    tile_right,
                    tile_bottom
                )
            )

            small = tile_region.resize(
                (
                    1,
                    1
                ),
                Image.Resampling.NEAREST
            )

            mosaic = small.resize(
                (
                    tile_right - tile_left,
                    tile_bottom - tile_top
                ),
                Image.Resampling.NEAREST
            )

            result.paste(
                mosaic,
                (
                    tile_left,
                    tile_top
                )
            )

    return result

# =====================================================
# TILE MOSAIC SETTINGS
# =====================================================

def get_tile_mosaic_settings():

    grid_size = st.sidebar.slider(
        "タイルサイズ",
        10,
        120,
        30,
        5
    )

    return grid_size

# =====================================================
# TILE MOSAIC DISPATCHER
# =====================================================

def apply_tile_mask(
    image,
    polygons
):

    grid_size = (
        get_tile_mosaic_settings()
    )

    return apply_tile_mosaic(
        image,
        polygons,
        grid_size
    )

# =====================================================
# TILE DEBUG DRAW
# =====================================================

def draw_tile_grid(
    image,
    grid_size
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    img_w, img_h = image.size

    for x in range(
        0,
        img_w,
        grid_size
    ):

        draw.line(
            (
                x,
                0,
                x,
                img_h
            ),
            fill="lime",
            width=1
        )

    for y in range(
        0,
        img_h,
        grid_size
    ):

        draw.line(
            (
                0,
                y,
                img_w,
                y
            ),
            fill="lime",
            width=1
        )

    return preview

# =====================================================
# TILE MOSAIC PREVIEW
# =====================================================

def tile_mosaic_preview(
    image,
    polygons
):

    grid_size = (
        get_tile_mosaic_settings()
    )

    result = apply_tile_mosaic(
        image,
        polygons,
        grid_size
    )

    return result

# =====================================================
# IOU ENGINE
# =====================================================

def calculate_iou(
    box1,
    box2
):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(
        x1,
        x2
    )

    ya = max(
        y1,
        y2
    )

    xb = min(
        x1 + w1,
        x2 + w2
    )

    yb = min(
        y1 + h1,
        y2 + h2
    )

    intersection = (

        max(
            0,
            xb - xa
        )

        *

        max(
            0,
            yb - ya
        )

    )

    if intersection <= 0:

        return 0.0

    union = (

        (w1 * h1)

        +

        (w2 * h2)

        -

        intersection

    )

    if union <= 0:

        return 0.0

    return (
        intersection
        /
        union
    )

# =====================================================
# BOX CENTER
# =====================================================

def box_center(
    box
):

    x, y, w, h = box

    return (

        x + (w / 2),

        y + (h / 2)

    )

# =====================================================
# CENTER DISTANCE
# =====================================================

def center_distance(
    box1,
    box2
):

    cx1, cy1 = box_center(
        box1
    )

    cx2, cy2 = box_center(
        box2
    )

    return math.sqrt(

        (
            cx1 - cx2
        ) ** 2

        +

        (
            cy1 - cy2
        ) ** 2

    )

# =====================================================
# SIZE SIMILARITY
# =====================================================

def size_similarity(
    box1,
    box2
):

    _, _, w1, h1 = box1
    _, _, w2, h2 = box2

    area1 = w1 * h1
    area2 = w2 * h2

    if (
        area1 <= 0
        or
        area2 <= 0
    ):

        return 0.0

    return (

        min(
            area1,
            area2
        )

        /

        max(
            area1,
            area2
        )

    )

# =====================================================
# DUPLICATE SCORE
# =====================================================

def duplicate_score(
    box1,
    box2
):

    iou = calculate_iou(
        box1,
        box2
    )

    dist = center_distance(
        box1,
        box2
    )

    scale = size_similarity(
        box1,
        box2
    )

    dist_score = (

        1.0

        /

        (

            1.0

            +

            dist / 50.0

        )

    )

    score = (

        iou * 0.50

        +

        dist_score * 0.30

        +

        scale * 0.20

    )

    return score

# =====================================================
# DUPLICATE CHECK
# =====================================================

def is_duplicate_box(
    new_box,
    existing_boxes,
    threshold=0.55
):

    for box in existing_boxes:

        score = duplicate_score(
            new_box,
            box
        )

        if score >= threshold:

            return True

    return False

# =====================================================
# REMOVE DUPLICATES
# =====================================================

def remove_duplicate_boxes(
    boxes,
    threshold=0.55
):

    unique_boxes = []

    for box in boxes:

        if not is_duplicate_box(
            box,
            unique_boxes,
            threshold
        ):

            unique_boxes.append(
                box
            )

    return unique_boxes

# =====================================================
# BEST THRESHOLD
# =====================================================

def get_duplicate_threshold(
    precision_mode
):

    if precision_mode == "AI限界突破":

        return 0.45

    elif precision_mode == "群衆特化":

        return 0.50

    elif precision_mode == "超高精度":

        return 0.55

    return 0.60

# =====================================================
# IOU MERGE
# =====================================================

def merge_two_boxes(
    box1,
    box2
):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return (

        int(
            (x1 + x2) / 2
        ),

        int(
            (y1 + y2) / 2
        ),

        int(
            (w1 + w2) / 2
        ),

        int(
            (h1 + h2) / 2
        )

    )

# =====================================================
# GROUP DUPLICATES
# =====================================================

def group_duplicate_boxes(
    boxes,
    threshold=0.55
):

    groups = []

    for box in boxes:

        found_group = False

        for group in groups:

            for member in group:

                score = duplicate_score(
                    box,
                    member
                )

                if score >= threshold:

                    group.append(
                        box
                    )

                    found_group = True

                    break

            if found_group:
                break

        if not found_group:

            groups.append(
                [box]
            )

    return groups

# =====================================================
# MERGE GROUPS
# =====================================================

def merge_box_groups(
    groups
):

    merged = []

    for group in groups:

        xs = []
        ys = []
        ws = []
        hs = []

        for x, y, w, h in group:

            xs.append(x)
            ys.append(y)
            ws.append(w)
            hs.append(h)

        merged.append(

            (
                int(np.mean(xs)),
                int(np.mean(ys)),
                int(np.mean(ws)),
                int(np.mean(hs))
            )

        )

    return merged

# =====================================================
# FULL IOU DEDUP
# =====================================================

def deduplicate_boxes(
    boxes,
    precision_mode="標準"
):

    threshold = (
        get_duplicate_threshold(
            precision_mode
        )
    )

    groups = group_duplicate_boxes(
        boxes,
        threshold
    )

    return merge_box_groups(
        groups
    )

# =====================================================
# CROWD MERGE ENGINE
# =====================================================

def landmark_to_box(
    face_points
):

    xs = [
        p[0]
        for p in face_points
    ]

    ys = [
        p[1]
        for p in face_points
    ]

    return (

        min(xs),

        min(ys),

        max(xs) - min(xs),

        max(ys) - min(ys)

    )

# =====================================================
# LANDMARK -> BOXES
# =====================================================

def landmark_faces_to_boxes(
    landmark_faces
):

    boxes = []

    for face in landmark_faces:

        try:

            box = landmark_to_box(
                face
            )

            boxes.append(
                box
            )

        except Exception:
            pass

    return boxes

# =====================================================
# FACE QUALITY
# =====================================================

def face_box_area(
    box
):

    _, _, w, h = box

    return max(
        0,
        w * h
    )

# =====================================================
# BEST BOX SELECTOR
# =====================================================

def select_best_box(
    boxes
):

    if len(boxes) == 0:

        return None

    best_box = boxes[0]

    best_area = face_box_area(
        best_box
    )

    for box in boxes[1:]:

        area = face_box_area(
            box
        )

        if area > best_area:

            best_area = area

            best_box = box

    return best_box

# =====================================================
# CROWD DUPLICATE GROUPING
# =====================================================

def crowd_group_boxes(
    boxes,
    threshold=0.50
):

    groups = []

    for new_box in boxes:

        assigned = False

        for group in groups:

            matched = False

            for existing in group:

                score = duplicate_score(
                    new_box,
                    existing
                )

                if score >= threshold:

                    matched = True
                    break

            if matched:

                group.append(
                    new_box
                )

                assigned = True
                break

        if not assigned:

            groups.append(
                [new_box]
            )

    return groups

# =====================================================
# CROWD MERGE GROUP
# =====================================================

def crowd_merge_group(
    group
):

    xs = []
    ys = []
    ws = []
    hs = []

    for x, y, w, h in group:

        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

    return (

        int(np.mean(xs)),

        int(np.mean(ys)),

        int(np.mean(ws)),

        int(np.mean(hs))

    )

# =====================================================
# CROWD MERGE ALL
# =====================================================

def crowd_merge_boxes(
    boxes,
    precision_mode="群衆特化"
):

    threshold = (
        get_duplicate_threshold(
            precision_mode
        )
    )

    groups = crowd_group_boxes(
        boxes,
        threshold
    )

    merged_boxes = []

    for group in groups:

        merged_boxes.append(

            crowd_merge_group(
                group
            )

        )

    return merged_boxes

# =====================================================
# HYBRID MERGE
# =====================================================

def merge_detector_and_landmarker(
    detector_boxes,
    landmark_faces,
    precision_mode="群衆特化"
):

    landmark_boxes = (
        landmark_faces_to_boxes(
            landmark_faces
        )
    )

    merged = []

    #
    # FaceLandmarker優先
    #

    for box in landmark_boxes:

        merged.append(
            box
        )

    #
    # FaceDetector追加
    #

    for detector_box in detector_boxes:

        duplicate = False

        for landmark_box in landmark_boxes:

            score = duplicate_score(
                detector_box,
                landmark_box
            )

            if score >= 0.50:

                duplicate = True
                break

        if not duplicate:

            merged.append(
                detector_box
            )

    #
    # 群衆統合
    #

    merged = crowd_merge_boxes(
        merged,
        precision_mode
    )

    return merged

# =====================================================
# LANDMARK/POLYGON MATCH
# =====================================================

def match_landmarks_to_boxes(
    landmark_faces,
    boxes
):

    matched = []

    landmark_boxes = (
        landmark_faces_to_boxes(
            landmark_faces
        )
    )

    for box in boxes:

        best_face = None

        best_score = -1

        for idx, face_box in enumerate(
            landmark_boxes
        ):

            score = calculate_iou(
                box,
                face_box
            )

            if score > best_score:

                best_score = score

                best_face = idx

        matched.append(

            (
                box,
                best_face
            )

        )

    return matched

# =====================================================
# CROWD STATISTICS
# =====================================================

def crowd_statistics(
    before_count,
    after_count
):

    removed = (
        before_count
        -
        after_count
    )

    return {

        "before":
        before_count,

        "after":
        after_count,

        "removed":
        removed

    }

# =====================================================
# CROWD DEBUG
# =====================================================

def show_crowd_debug(
    raw_boxes,
    merged_boxes
):

    stats = crowd_statistics(

        len(raw_boxes),

        len(merged_boxes)

    )

    st.write(
        f"Raw: {stats['before']}"
    )

    st.write(
        f"Merged: {stats['after']}"
    )

    st.write(
        f"Removed: {stats['removed']}"
    )

# =====================================================
# MULTI SCAN ENGINE
# =====================================================

SCAN_PRESETS = {

    "高速": {
        "grids": [1],
        "angles": [0],
        "scale": 1.0
    },

    "標準": {
        "grids": [1, 2],
        "angles": [0],
        "scale": 1.2
    },

    "超高精度": {
        "grids": [1, 2, 4, 5],
        "angles": [0, 15, -15],
        "scale": 1.8
    },

    "群衆特化": {
        "grids": [1, 2, 4, 6, 8],
        "angles": [0, 15, -15, 30, -30],
        "scale": 2.5
    },

    "AI限界突破": {
        "grids": [1, 2, 4, 6, 8, 10],
        "angles": [
            0,
            15,
            -15,
            30,
            -30,
            45,
            -45
        ],
        "scale": 3.0
    }
}

# ============================================
# =====================================================
# HYBRID DETECT ENGINE
# =====================================================

def hybrid_detect(
    image,
    precision_mode="群衆特化"
):
    """
    FaceDetector
    +
    FaceLandmarker
    +
    Crowd Merge
    """

    detector_boxes = (
        detect_faces_detector(
            image
        )
    )

    landmark_faces = (
        detect_faces_landmarker(
            image
        )
    )

    merged_boxes = (
        merge_detector_and_landmarker(
            detector_boxes,
            landmark_faces,
            precision_mode
        )
    )

    merged_boxes = (
        deduplicate_boxes(
            merged_boxes,
            precision_mode
        )
    )

    return (
        merged_boxes,
        landmark_faces
    )


# =====================================================
# HYBRID MULTI SCAN
# =====================================================

def hybrid_multiscan_detect(
    image,
    precision_mode="群衆特化"
):

    scan_boxes, scan_faces = (
        run_multi_scan(
            image,
            precision_mode
        )
    )

    hybrid_boxes, hybrid_faces = (
        hybrid_detect(
            image,
            precision_mode
        )
    )

    all_boxes = []
    all_faces = []

    all_boxes.extend(
        scan_boxes
    )

    all_boxes.extend(
        hybrid_boxes
    )

    all_faces.extend(
        scan_faces
    )

    all_faces.extend(
        hybrid_faces
    )

    final_boxes = (
        crowd_merge_boxes(
            all_boxes,
            precision_mode
        )
    )

    final_boxes = (
        deduplicate_boxes(
            final_boxes,
            precision_mode
        )
    )

    return (
        final_boxes,
        all_faces
    )


# =====================================================
# BOX TO POLYGON
# =====================================================

def box_to_polygon(
    box
):

    x, y, w, h = box

    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]


# =====================================================
# MATCH FACE DATA
# =====================================================

def build_face_data(
    boxes,
    landmark_faces
):

    matched = (
        match_landmarks_to_boxes(
            landmark_faces,
            boxes
        )
    )

    results = []

    for (
        box,
        face_index
    ) in matched:

        face_points = None

        if (
            face_index is not None
            and
            face_index < len(
                landmark_faces
            )
        ):

            face_points = (
                landmark_faces[
                    face_index
                ]
            )

        results.append(
            {
                "box": box,
                "landmarks": face_points
            }
        )

    return results


# =====================================================
# TARGET POLYGONS
# =====================================================

def build_target_polygons(
    face_data,
    mask_target
):

    polygons = []

    for face in face_data:

        landmarks = (
            face["landmarks"]
        )

        if landmarks:

            polygons.extend(
                get_target_polygons(
                    landmarks,
                    mask_target
                )
            )

        else:

            polygons.append(
                box_to_polygon(
                    face["box"]
                )
            )

    return polygons


# =====================================================
# COMPLETE DETECTION PIPELINE
# =====================================================

def detect_all(
    image,
    precision_mode,
    mask_target
):

    boxes, faces = (
        hybrid_multiscan_detect(
            image,
            precision_mode
        )
    )

    face_data = (
        build_face_data(
            boxes,
            faces
        )
    )

    polygons = (
        build_target_polygons(
            face_data,
            mask_target
        )
    )

    return (
        boxes,
        faces,
        polygons
    )


# =====================================================
# DEBUG OVERLAY
# =====================================================

def draw_detection_overlay(
    image,
    boxes,
    polygons
):

    preview = image.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for x, y, w, h in boxes:

        draw.rectangle(
            (
                x,
                y,
                x + w,
                y + h
            ),
            outline="red",
            width=2
        )

    for polygon in polygons:

        if len(polygon) >= 3:

            draw.polygon(
                polygon,
                outline="lime",
                width=3
            )

    return preview


# =====================================================
# HYBRID PREVIEW
# =====================================================

def hybrid_preview(
    image,
    precision_mode,
    mask_target
):

    (
        boxes,
        faces,
        polygons

    ) = detect_all(
        image,
        precision_mode,
        mask_target
    )

    preview = draw_detection_overlay(
        image,
        boxes,
        polygons
    )

    st.image(
        preview,
        caption=f"Faces={len(boxes)}  Polygons={len(polygons)}"
    )

    return (
        boxes,
        faces,
        polygons
    )
# =====================================================
# MAIN MASK DISPATCHER
# =====================================================

def apply_mask(
    image,
    polygons,
    mask_type,
    mosaic_size=15,
    blur_radius=20,
    fill_color="#000000",
    emoji_char="🌸",
    emoji_scale=120,
    emoji_angle=0,
    offset_x=0,
    offset_y=0,
    tile_size=30
):

    result = image.copy()

    #
    # タイルモザイク
    #

    if mask_type == "タイルモザイク":

        return apply_tile_mosaic(
            result,
            polygons,
            tile_size
        )

    #
    # 絵文字
    #

    if mask_type == "絵文字":

        return apply_emoji_stamps(
            result,
            polygons,
            emoji_char,
            emoji_scale,
            emoji_angle,
            offset_x,
            offset_y
        )

    #
    # ポリゴンベース処理
    #

    for polygon in polygons:

        if len(polygon) < 3:
            continue


# =====================================================
# MAIN
# =====================================================

def main():

    st.sidebar.title(
        "AI Smart Masking Pro"
    )

    precision_mode = st.sidebar.selectbox(
        "検出モード",
        [
            "高速",
            "標準",
            "超高精度",
            "群衆特化",
            "AI限界突破"
        ],
        index=3
    )

    mask_target = st.sidebar.selectbox(
        "マスク対象",
        [
            "顔全体",
            "両目",
            "右目",
            "左目",
            "鼻",
            "口"
        ]
    )

    uploaded_file = st.file_uploader(
        "画像を選択してください",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )

    if uploaded_file is None:

        st.info(
            "画像をアップロードしてください"
        )

        return

    try:

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )

    except Exception:

        st.error(
            "画像を読み込めませんでした"
        )

        return

    st.image(
        image,
        caption="元画像",
    )

    run_masking_ui(
        image,
        precision_mode,
        mask_target
    )
def run_masking_ui(
    image,
    precision_mode,
    mask_target
):

    if st.button(
        "🚀 解析開始"
    ):

        with st.spinner(
            "解析中..."
        ):

            boxes, faces, polygons = detect_all(
                image,
                precision_mode,
                mask_target
            )

            result = apply_mask(
                image=image,
                polygons=polygons,
                mask_type="モザイク",
                mosaic_size=15
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "元画像"
                )

                st.image(image)

            with col2:

                st.subheader(
                    "結果"
                )

                st.image(result)

            st.success(
                f"検出数: {len(boxes)}"
            )

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        st.error(

            "予期しないエラーが発生しました"

        )

        st.code(
            traceback.format_exc()
        )