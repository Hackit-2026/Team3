import re

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

app = FastAPI(title="Text Mosaic Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 座標抽出に加え、行内容に応じたモザイク強制（メールアドレス等の検知）のため
# 文字認識（rec）も有効にしたフルパイプラインで起動する。
# enable_mkldnn=False は、この環境の paddlepaddle/paddleocr の組み合わせで
# デフォルトの oneDNN 実行モードが NotImplementedError を起こすため無効化している。
# text_det_thresh/box_thresh はデフォルト(0.3/0.6付近)だと、ガードレールに重なる
# など、コントラストの低い小さな文字を見逃すことが実写真の検証で分かったため、
# 0.2まで下げて再現率を優先している（誤検出の増加は検証画像では確認されなかった）。
# text_det_limit_side_len はモデルに入力する際の画像の最大辺サイズ。デフォルト
# 相当(960px)まで縮小されると、ガードレールに一部重なった細い文字が1本の線として
# 繋がって見えず、行の一部しか検出されなかった。1800まで上げて解像度を保つことで、
# 行全体が1つの枠として検出されるようになった（ノイズ検出の増加は極小の
# ボックスとしてほぼ吸収され、下記の最小サイズフィルターで除去できる）。
# text_rec_score_thresh=0.0 は、認識信頼度によって座標そのものが欠落しないように
# するため。座標は文字を判読するためではなく場所を特定するために使うので、
# 認識精度が低くてもモザイク対象からは外さない。
# use_doc_orientation_classify/use_doc_unwarping/use_textline_orientation は
# スキャン文書向けの前処理で、写真の文字検出には不要なため無効化し高速化する。
ocr_engine = PaddleOCR(
    lang="japan",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=1800,
    text_det_limit_type="max",
    text_det_thresh=0.2,
    text_det_box_thresh=0.2,
    text_rec_score_thresh=0.0,
    enable_mkldnn=False,
)

# 行のどこかにこれらのパターンが含まれていたら、個人情報（連絡先・住所）の
# 一部とみなし、面積フィルターに関わらずその行全体を必ずモザイク対象にする。
# 正規表現による簡易判定のため完全ではないが、「確実にそれと分かる」強いシグナルに絞っている。
SENSITIVE_LINE_PATTERNS = [
    re.compile(r"@"),  # メールアドレス
    re.compile(r"〒"),  # 郵便番号記号
    re.compile(r"TEL", re.IGNORECASE),  # 電話番号ラベル
    re.compile(r"\d{2,4}-\d{2,4}-\d{4}"),  # 電話番号らしき数字列
    re.compile(r"\d{3}-\d{4}"),  # 郵便番号らしき数字列
    re.compile(r"[都道府県].{0,15}[市区町村].{0,10}(丁目|番地|号)"),  # 住所らしき文字列
]


def is_sensitive_line(text: str) -> bool:
    return any(p.search(text) for p in SENSITIVE_LINE_PATTERNS)


# 模様やロゴのテクスチャを文字と誤検出した際にできる、数ピクセルのノイズ枠を除外する。
MIN_BOX_SIZE_PX = 10

# 看板などの巨大な文字枠を除外する閾値。画像全体に対する面積比で判定することで、
# 画像の解像度が変わっても一定の基準で「大きすぎる枠」を弾けるようにする。
# 当初は0.06（典型的な構図を想定した中間値）だったが、実写真での検証で
# 密集した文字が結合されて大きな枠になったり（カレンダーの数字）、近距離で
# 撮ったスライドの個人情報（メールアドレス等）自体が画面の大部分を占めたりして、
# 本来隠すべき個人情報を誤って除外してしまうケースが複数見つかった。
# 見逃し（個人情報の露出）は誤除外の逆（看板に一時的にモザイクがかかる）より
# 深刻なため、画面のほとんどを1つの被写体が占めるような極端なケースだけを
# 除外する0.25まで引き上げ、実質的な取りこぼしを最小化する方針にした。
DEFAULT_MAX_AREA_RATIO = 0.25


def detect_text_boxes(image: np.ndarray) -> list[dict]:
    results = ocr_engine.predict(image)

    texts = []
    for res in results:
        for text, box in zip(res["rec_texts"], res["rec_boxes"]):
            x_min, y_min, x_max, y_max = (int(v) for v in box)
            texts.append(
                {
                    "x": x_min,
                    "y": y_min,
                    "w": x_max - x_min,
                    "h": y_max - y_min,
                    "text": text,
                }
            )
    return texts


def expand_sensitive_lines(texts: list[dict], image_width: int) -> list[dict]:
    # メール/電話/住所などと判定された行は、前後の文字も含めて隠せるよう
    # 検出された枠の幅に関わらず行全体（画像の横幅いっぱい）をモザイク対象にする。
    expanded = []
    for t in texts:
        if is_sensitive_line(t["text"]):
            expanded.append({**t, "x": 0, "w": image_width})
        else:
            expanded.append(t)
    return expanded


def filter_oversized_boxes(
    texts: list[dict], image_width: int, image_height: int, max_area_ratio: float
) -> list[dict]:
    image_area = image_width * image_height
    max_area = image_area * max_area_ratio
    return [t for t in texts if t["w"] * t["h"] < max_area]


def filter_tiny_boxes(texts: list[dict], min_size: int) -> list[dict]:
    return [t for t in texts if t["w"] >= min_size and t["h"] >= min_size]


@app.post("/api/detect-text")
async def detect_text(
    file: UploadFile = File(...),
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
):
    data = await file.read()
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした。")

    image_height, image_width = image.shape[:2]

    texts = detect_text_boxes(image)
    texts = filter_tiny_boxes(texts, MIN_BOX_SIZE_PX)

    # メール/電話/住所などの強いシグナルを含む行は、看板フィルターの対象外として
    # 必ずモザイクをかける（面積フィルターより優先する）。
    sensitive = [t for t in texts if is_sensitive_line(t["text"])]
    normal = [t for t in texts if not is_sensitive_line(t["text"])]

    normal = filter_oversized_boxes(normal, image_width, image_height, max_area_ratio)
    sensitive = expand_sensitive_lines(sensitive, image_width)

    texts = sensitive + normal

    return {
        "status": "success",
        "image_width": image_width,
        "image_height": image_height,
        "texts": [
            {"id": f"t_{i + 1:03d}", "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]}
            for i, t in enumerate(texts)
        ],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
