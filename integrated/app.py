"""顔モザイク と 文字モザイク を1つの流れで同時に処理する統合アプリ。

【動作】
  写真を1枚アップロードして「解析を開始」を押すと、
    ・顔検出   … face/app.py の検出処理（ホワイトリスト照合込み）
    ・文字検出 … word/backend の /api/detect-text（OCR＋LLM判定）
  の両方を実行し、2つの結果を1つにまとめて1枚の画像にモザイクを掛ける。

【既存機能を壊さないための作り】
  face/app.py はコピーせず、ファイルそのものを実行して
    ・サイドバーのマスク設定UI
    ・ホワイトリスト登録UI
    ・検出／マスク処理の関数一式
  をそのまま使う。ただし face/app.py 自身の「写真アップロード〜プレビュー」部分
  （＝統合版では重複する部分）だけは実行しないように、その入口で実行を止めている。
  文字検出も word/backend のAPIをそのまま呼ぶだけで、判定ロジックには触れていない。

起動:
    python -m streamlit run app.py
"""

import inspect
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# -------------------------------------------------------------------
# 0. パスと接続先
# -------------------------------------------------------------------
HACKIT_DIR = Path(__file__).resolve().parent.parent
FACE_APP_PATH = HACKIT_DIR / "face" / "app.py"
FACE_MODEL_DIR = HACKIT_DIR / "face" / "mp_models"

TEXT_BACKEND_ORIGIN = "http://localhost:8000"
TEXT_DETECT_URL = f"{TEXT_BACKEND_ORIGIN}/api/detect-text"
TEXT_HEALTH_URL = f"{TEXT_BACKEND_ORIGIN}/api/health"
TEXT_UI_URL = f"{TEXT_BACKEND_ORIGIN}/ui/index.html"

# OCRは画像1枚に十数秒〜数十秒かかるため、余裕を持ったタイムアウトにする。
TEXT_DETECT_TIMEOUT_SEC = 300

# face/app.py 側の「加工したい写真をアップロードしてください」の入口。
# ここに到達したら face/app.py の実行を打ち切る（統合版が自前で用意するため）。
FACE_MAIN_UPLOADER_PREFIX = "加工したい写真"

st.set_page_config(
    page_title="AI Privacy Masking Suite", page_icon="🛡️", layout="wide"
)


# 画像を横幅いっぱいで表示する方法は Streamlit のバージョンで変わっている。
# 新しい版は width="stretch"、古い版は use_column_width=True。新しい版で
# 古い書き方をすると、画面に黄色い非推奨メッセージが出てしまうため出し分ける。
# 既定値が文字列（"content"）なら新しい書き方に対応している版と判断できる。
_WIDTH_DEFAULT = inspect.signature(st.image).parameters["width"].default
_SUPPORTS_WIDTH_STRETCH = isinstance(_WIDTH_DEFAULT, str)


def show_image(image) -> None:
    if _SUPPORTS_WIDTH_STRETCH:
        st.image(image, width="stretch")
    else:
        st.image(image, use_column_width=True)


# -------------------------------------------------------------------
# 1. face/app.py を「設定UIと関数だけ」実行する
# -------------------------------------------------------------------
class _StopBeforeMainUI(Exception):
    """face/app.py の写真アップロード部分に来たら実行を止めるための合図。"""


def run_face_module() -> dict:
    """face/app.py を無改変のまま実行し、その名前空間（関数・設定値）を返す。

    実行されるのは
      ・モデル読み込み（@st.cache_resource が効くので2回目以降は一瞬）
      ・サイドバーのマスク設定UI
      ・ホワイトリスト登録UI
      ・各種関数の定義
    まで。写真アップロード以降は統合版が担当するので実行しない。
    """
    source = FACE_APP_PATH.read_text(encoding="utf-8")
    code = compile(source, str(FACE_APP_PATH), "exec")
    ns = {"__file__": str(FACE_APP_PATH), "__name__": "hackit_face_app"}

    original_set_page_config = st.set_page_config
    original_file_uploader = st.file_uploader
    original_title = st.title
    original_caption = st.caption

    def _stop_at_main_uploader(label, *args, **kwargs):
        if isinstance(label, str) and label.startswith(FACE_MAIN_UPLOADER_PREFIX):
            raise _StopBeforeMainUI()
        return original_file_uploader(label, *args, **kwargs)

    # face/app.py 冒頭のタイトルとその直後の説明文だけを黙らせる（統合版の
    # ヘッダーと二重になるため）。ホワイトリスト等、後に出てくる説明文は残す。
    suppressed_captions = [0]

    def _caption_once(*args, **kwargs):
        if suppressed_captions[0] == 0:
            suppressed_captions[0] = 1
            return None
        return original_caption(*args, **kwargs)

    # ページ設定は統合版で1回だけ行う（検出やマスクの処理内容には影響しない）
    st.set_page_config = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.caption = _caption_once
    st.file_uploader = _stop_at_main_uploader

    reached_main_ui = False
    try:
        exec(code, ns)
    except _StopBeforeMainUI:
        reached_main_ui = True
    finally:
        st.set_page_config = original_set_page_config
        st.file_uploader = original_file_uploader
        st.title = original_title
        st.caption = original_caption

    if not reached_main_ui:
        st.warning(
            "face/app.py の写真アップロード欄が見つからなかったため、"
            "顔モザイク側の画面が二重に表示されている可能性があります。"
            f"（目印にしているラベル: 「{FACE_MAIN_UPLOADER_PREFIX}…」）"
        )
    return ns


# -------------------------------------------------------------------
# 2. 文字検出（word/backend のAPIをそのまま呼ぶ）
# -------------------------------------------------------------------
def is_text_backend_alive(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(TEXT_HEALTH_URL, timeout=timeout) as res:
            return res.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def detect_texts(image_bytes: bytes, filename: str) -> list[dict]:
    """OCR＋LLM判定の結果（文字ごとの矩形とラベル）を返す。"""
    response = requests.post(
        TEXT_DETECT_URL,
        files={"file": (filename, image_bytes, "application/octet-stream")},
        timeout=TEXT_DETECT_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json().get("texts", [])


def text_to_mask_item(text: dict) -> dict:
    """文字の矩形を、face/app.py の apply_masking が扱える形に変換する。

    多角形として矩形の4隅を渡すため、マスクの形状設定が「顔の形（楕円）」でも
    文字部分は元の文字モザイクアプリと同じく四角いまま塗られる。
    """
    x, y, w, h = int(text["x"]), int(text["y"]), int(text["w"]), int(text["h"])
    return {
        "box": (x, y, w, h),
        "polygon": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        "type": "text",
    }


# -------------------------------------------------------------------
# 3. 画面 - ヘッダーと顔側の設定UI
# -------------------------------------------------------------------
st.markdown("## 🛡️ AI Privacy Masking Suite")
st.caption(
    "写真を1枚アップロードすると、**顔と文字を同時に検出**して、"
    "1枚の画像にまとめてモザイクを掛けます。"
)

if not FACE_MODEL_DIR.exists():
    st.warning(
        f"モデルフォルダ {FACE_MODEL_DIR} がありません。"
        "起動.bat を使うと自動で作成・配置されます。"
    )

face_ns = run_face_module()

# -------------------------------------------------------------------
# 4. 画面 - 文字側の設定UI（サイドバー）
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔤 文字モザイク設定")

detect_text_enabled = st.sidebar.checkbox(
    "文字も検出してモザイクする", value=True, key="txt_enabled",
    help="外すと顔だけを処理します（文字認識サーバーが止まっていても使えます）。",
)

text_style_mode = st.sidebar.radio(
    "文字部分の塗りつぶし方",
    ["黒塗り（文字モザイクアプリと同じ）", "顔と同じマスキングスタイル"],
    index=0,
    key="txt_style_mode",
)

mask_public_texts = st.sidebar.checkbox(
    "public と判定された文字も隠す", value=False, key="txt_mask_public",
    help="LLMが「一般に公開されている情報」と判断した文字列は、既定では隠しません。",
)

show_manual_editor = st.sidebar.checkbox(
    "文字を手作業で編集する画面も開く", value=False, key="txt_manual",
    help="元の文字モザイクアプリをそのまま埋め込みます。"
         "ドラッグで枠を追加したい場合に使ってください。",
)

# -------------------------------------------------------------------
# 5. 画面 - 写真のアップロードと解析
# -------------------------------------------------------------------
st.markdown("---")

if "suite_file_id" not in st.session_state:
    st.session_state.suite_file_id = None
if "suite_image" not in st.session_state:
    st.session_state.suite_image = None
if "suite_face_items" not in st.session_state:
    st.session_state.suite_face_items = None
if "suite_texts" not in st.session_state:
    st.session_state.suite_texts = None
if "suite_report" not in st.session_state:
    st.session_state.suite_report = None
if "suite_timing" not in st.session_state:
    st.session_state.suite_timing = None
if "suite_confirmed" not in st.session_state:
    st.session_state.suite_confirmed = False

uploaded_file = st.file_uploader(
    "加工したい写真をアップロードしてください (JPEG, PNG, WEBP, BMP)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key="suite_uploader",
)

if uploaded_file is None:
    st.info("👆 写真をアップロードして「🚀 顔と文字をまとめて解析」を押してください。")
    st.stop()

file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.suite_file_id != file_id:
    st.session_state.suite_file_id = file_id
    st.session_state.suite_face_items = None
    st.session_state.suite_texts = None
    st.session_state.suite_report = None
    st.session_state.suite_timing = None
    st.session_state.suite_confirmed = False
    try:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        st.session_state.suite_image = image
        uploaded_file.seek(0)
        st.session_state.suite_image_bytes = uploaded_file.read()
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.session_state.suite_image = None

input_image = st.session_state.suite_image
if input_image is None:
    st.stop()

mask_targets = face_ns.get("mask_targets", [])
grids = face_ns.get("grids", [1])
whitelist_enabled = face_ns.get("whitelist_enabled", False)
whitelist_threshold = face_ns.get("whitelist_threshold", 0.70)

if st.button("🚀 顔と文字をまとめて解析", type="primary"):
    if not mask_targets and not detect_text_enabled:
        st.warning("⚠️ 「マスク対象」を選ぶか、文字の検出を有効にしてください。")
    else:
        timing = {}
        # --- 顔の検出（face/app.py の処理をそのまま使用）---
        face_items, detected_face_count, report = [], 0, None
        if mask_targets:
            with st.spinner("AIが顔を精密検出中..."):
                t0 = time.time()
                try:
                    face_items = face_ns["get_mask_boxes_locally"](
                        input_image, mask_targets, grids=grids
                    )
                    detected_face_count = len(face_items)
                    if whitelist_enabled and st.session_state.get("whitelist"):
                        exempt_boxes, report = face_ns["compute_whitelist_exemptions"](
                            input_image, grids, st.session_state.whitelist, whitelist_threshold
                        )
                        face_items = face_ns["filter_out_whitelisted"](face_items, exempt_boxes)
                except Exception as e:
                    st.error(f"顔の解析でエラーが発生しました: {e}")
                timing["顔検出"] = time.time() - t0

        # --- 文字の検出（word/backend のAPIをそのまま使用）---
        texts = []
        if detect_text_enabled:
            with st.spinner("AIが文字を検出中（OCR＋LLM判定）..."):
                t0 = time.time()
                try:
                    texts = detect_texts(
                        st.session_state.suite_image_bytes, uploaded_file.name
                    )
                except requests.RequestException as e:
                    st.error(
                        "文字認識サーバー（http://localhost:8000）に接続できませんでした。"
                        f"顔のモザイクだけ適用します。詳細: {e}"
                    )
                timing["文字検出"] = time.time() - t0

        st.session_state.suite_face_items = face_items
        st.session_state.suite_texts = texts
        st.session_state.suite_report = report
        st.session_state.suite_timing = timing
        st.session_state.suite_confirmed = False

        exempted = detected_face_count - len(face_items)
        msg = f"解析完了！ 顔 {len(face_items)} 箇所 ／ 文字 {len(texts)} 箇所を検出しました。"
        if exempted > 0:
            msg += f"（ホワイトリスト一致で顔 {exempted} 箇所を除外）"
        st.success(msg)

# -------------------------------------------------------------------
# 6. 画面 - 合成結果の表示
# -------------------------------------------------------------------
if st.session_state.suite_face_items is None and st.session_state.suite_texts is None:
    st.stop()

face_items = st.session_state.suite_face_items or []
texts = st.session_state.suite_texts or []

# 表示は「画像 → 文字一覧」の順だが、どの文字を隠すかはチェックボックスを
# 作ってからでないと分からない。そこで画像用の場所だけ先に確保しておき、
# 文字一覧を先に評価してから、あとで画像をこの場所に描き込む。
image_area = st.container()
summary_area = st.container()

# --- 検出された文字の一覧（ON/OFF切り替え）---
# キーにファイルIDと「publicも隠す」の状態を含めているので、
#   ・別の写真を読み込んだとき
#   ・「public と判定された文字も隠す」を切り替えたとき
# はチェック状態が既定値に戻り、それ以外では個別の指定が保持される。
enabled_map: dict[str, bool] = {}
if texts:
    with st.expander(f"🔤 検出された文字列（{len(texts)} 件）— 隠す／隠さないを個別に変更", expanded=True):
        st.caption(
            "private＝個人情報の可能性あり（既定でON）／ public＝一般公開情報とLLMが判断（既定でOFF）。"
        )
        for t in texts:
            label = t.get("label", "private")
            badge = "🔴 private" if label == "private" else "🟢 public"
            default_on = (label != "public") or mask_public_texts
            key = f"txtchk_{st.session_state.suite_file_id}_{int(mask_public_texts)}_{t['id']}"
            enabled_map[t["id"]] = st.checkbox(
                f"{badge}　`{t['id']}`　{t['text'] or '(認識できず)'}",
                value=default_on,
                key=key,
            )
elif detect_text_enabled:
    st.info("文字は検出されませんでした。")

# 文字はチェックが入っているものだけをマスク対象にする
active_texts = [t for t in texts if enabled_map.get(t["id"], True)]
text_items = [text_to_mask_item(t) for t in active_texts]

apply_masking = face_ns["apply_masking"]
mask_shape = face_ns.get("mask_shape", "顔の形（輪郭に沿う・楕円）")
mask_type = face_ns.get("mask_type", "ピクセルモザイク")


def render_masked_image(image: Image.Image) -> Image.Image:
    """顔と文字を1枚に合成してマスキングした画像を返す。"""
    common = dict(
        grid_size=face_ns.get("grid_size", 30),
        mosaic_size=face_ns.get("mosaic_size", 15),
        blur_radius=face_ns.get("blur_radius", 20),
        fill_color=face_ns.get("fill_color", "#000000"),
        offset_x=face_ns.get("offset_x", 0),
        offset_y=face_ns.get("offset_y", 0),
        emoji_char=face_ns.get("emoji_char", "🌸"),
        emoji_scale=face_ns.get("emoji_scale", 110),
        emoji_angle=face_ns.get("emoji_angle", 0),
    )

    if text_style_mode == "顔と同じマスキングスタイル":
        # 顔と文字を同じスタイルで一度に処理する
        return apply_masking(image, face_items + text_items, mask_shape, mask_type, **common)

    # 文字は黒塗り（元の文字モザイクアプリと同じ見た目）にするので2段階で処理する
    result = apply_masking(image, face_items, mask_shape, mask_type, **common)
    if text_items:
        black = dict(common)
        black["fill_color"] = "#000000"
        result = apply_masking(
            result, text_items, "四角（矩形）", "塗りつぶし（カラー指定）", **black
        )
    return result


processed_image = render_masked_image(input_image)

# 先に確保しておいた場所（文字一覧より前）に画像を描き込む
with image_area:
    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📷 元の画像")
        show_image(input_image)
    with col_right:
        st.subheader("🔍 マスキング結果（顔＋文字）")
        show_image(processed_image)

with summary_area:
    st.caption(
        f"顔 {len(face_items)} 箇所／文字 {len(active_texts)} 箇所"
        f"（検出 {len(texts)} 箇所中）をマスクしています。"
    )
    timing = st.session_state.suite_timing
    if timing:
        st.caption(
            "処理時間: "
            + " ／ ".join(f"{k} {v:.1f}秒" for k, v in timing.items())
            + "（設定を変えても再検出は走らないので、この時間はかかりません）"
        )

# --- ホワイトリスト照合レポート（face/app.py と同じ内容）---
report = st.session_state.suite_report
if report:
    with st.expander(f"👤 ホワイトリスト照合の詳細（検出した顔 {len(report)} 件）", expanded=False):
        for idx, r in enumerate(report, start=1):
            status = "✅ 除外（モザイクしない）" if r["exempt"] else "🔒 モザイク対象"
            st.write(
                f"顔{idx}: 最も近い登録者「{r['best_name']}」／ 類似度 {r['best_sim']:.3f} → {status}"
            )
        st.caption("本人なのにモザイクされる場合はしきい値を下げ、他人が除外される場合は上げてください。")

# --- 決定とダウンロード ---
st.markdown("---")
confirm_col1, confirm_col2 = st.columns([2, 3])
with confirm_col1:
    if not st.session_state.suite_confirmed:
        if st.button("✅ これで決定", type="primary", use_container_width=True):
            st.session_state.suite_confirmed = True
            st.rerun()
    else:
        st.success("✨ 位置とデザインが確定しました！")
        if st.button("🔄 もう一度微調整する", use_container_width=True):
            st.session_state.suite_confirmed = False
            st.rerun()
with confirm_col2:
    if st.session_state.suite_confirmed:
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

# --- 元の文字モザイクアプリ（ドラッグで枠を足したいとき用）---
if show_manual_editor:
    st.markdown("---")
    st.subheader("✏️ 文字モザイクの手動編集（元アプリをそのまま埋め込み）")
    st.caption(
        "こちらは独立した画面です。ドラッグでの枠追加や枠クリックでの解除ができます。"
        "上の合成結果とは連動しません。"
    )
    if is_text_backend_alive():
        components.iframe(TEXT_UI_URL, height=1000, scrolling=True)
    else:
        st.error("文字認識サーバーに接続できないため、手動編集画面を表示できません。")
