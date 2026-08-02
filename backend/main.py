import re

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

# llm_classifier はインポート時に環境変数を読むため、.env の読み込みを先に行う。
load_dotenv()

import llm_classifier  # noqa: E402

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
# 繋がって見えず、行の一部しか検出されなかった。上げるほど解像度は保たれるが、
# 検出処理は解像度にほぼ比例して遅くなる（実測: 960px=6秒, 1800px=21秒/枚）。
# 1200であれば、ガードレール越しの行も1つの枠として検出できる状態を保ちつつ
# （実測で確認済み）、1800と比べて処理時間を半分程度に抑えられるためこの値にした。
# 完全に隙間なく検出したい場合は1800に戻すこともできるが、その分遅くなる。
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
    text_det_limit_side_len=1200,
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


def expand_to_full_line(texts: list[dict], image_width: int) -> list[dict]:
    # モザイク対象と判定された行は、前後の文字も含めて隠せるよう
    # 検出された枠の幅に関わらず行全体（画像の横幅いっぱい）をモザイク対象にする。
    return [{**t, "x": 0, "w": image_width} for t in texts]


def filter_tiny_boxes(texts: list[dict], min_size: int) -> list[dict]:
    return [t for t in texts if t["w"] >= min_size and t["h"] >= min_size]


@app.post("/api/detect-text")
async def detect_text(file: UploadFile = File(...)):
    data = await file.read()
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした。")

    image_height, image_width = image.shape[:2]

    texts = detect_text_boxes(image)
    texts = filter_tiny_boxes(texts, MIN_BOX_SIZE_PX)

    # 正規表現で強いシグナル（メール/電話/住所など）が取れた行は、LLMを呼ばずに
    # 即座にモザイク確定させる（通信不要で確実、ローカルLLM未接続でも最低限の
    # 保護を維持できる）。前後の文字も含めて隠せるよう行全体に幅を拡張する。
    sensitive = [t for t in texts if is_sensitive_line(t["text"])]
    other = [t for t in texts if not is_sensitive_line(t["text"])]
    sensitive = expand_to_full_line(sensitive, image_width)

    # 残りは「一般に広く知られている公開情報だとLLMが確信した場合のみ隠さない」
    # という発想の反転判定に委ねる。既定はモザイク対象とし、面積の大小では判定
    # しない（看板のような大きな文字列でも、公開情報だと判定されなければ隠す）。
    # LLM未接続・応答不能な場合は安全側（＝隠す）に倒れる。こちらは行全体には
    # 拡張せず、検出された枠のまま隠す。
    llm_flagged = [t for t in other if not llm_classifier.is_public_text(t["text"])]

    texts = sensitive + llm_flagged

    return {
        "status": "success",
        "image_width": image_width,
        "image_height": image_height,
        "texts": [
            {
                "id": f"t_{i + 1:03d}",
                "x": t["x"],
                "y": t["y"],
                "w": t["w"],
                "h": t["h"],
                "text": t["text"],
            }
            for i, t in enumerate(texts)
        ],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
