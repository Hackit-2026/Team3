"""顔モザイク と 文字モザイク を1つの画面にまとめた統合アプリ。

【設計方針：既存機能を絶対に変えない】
  - 顔モザイク : face/app.py を「そのままのファイル」として実行する（runpy）。
                 コードをコピーも改変もしていないため、単体起動したときと
                 まったく同じ検出・マスク・ホワイトリスト動作になる。
  - 文字モザイク: word/frontend/index.html を「そのままのページ」として
                 iframe で埋め込む。検出は従来どおり word/backend の
                 /api/detect-text が処理する（text_backend.py が配信）。

つまりこのファイルは、2つの既存アプリを1画面に並べる“入れ物”であり、
どちらの処理ロジックにも手を入れていない。

起動:
    python -m streamlit run app.py
"""

import runpy
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------------------------------------
# 0. パスと接続先の定義
# -------------------------------------------------------------------
HACKIT_DIR = Path(__file__).resolve().parent.parent
FACE_APP_PATH = HACKIT_DIR / "face" / "app.py"
FACE_MODEL_DIR = HACKIT_DIR / "face" / "mp_models"

TEXT_BACKEND_ORIGIN = "http://localhost:8000"
TEXT_UI_URL = f"{TEXT_BACKEND_ORIGIN}/ui/index.html"
TEXT_HEALTH_URL = f"{TEXT_BACKEND_ORIGIN}/api/health"

# face/app.py はモジュールとしてではなくスクリプトとして実行する。
# 実行のたびに毎回まっさらに評価させたいので import は使わない
# （import だとキャッシュされ、Streamlitの再実行時に中身が動かなくなる）。
FACE_RUN_NAME = "hackit_face_app"


# -------------------------------------------------------------------
# 1. ページ設定（統合画面として1回だけ行う）
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Privacy Masking Suite", page_icon="🛡️", layout="wide"
)

if "suite_mode" not in st.session_state:
    st.session_state.suite_mode = "🙂 顔モザイク"


# -------------------------------------------------------------------
# 2. 各機能の描画
# -------------------------------------------------------------------
def render_face_app() -> None:
    """face/app.py を無改変のまま実行して顔モザイク機能を描画する。"""
    if not FACE_APP_PATH.exists():
        st.error(f"顔モザイクのプログラムが見つかりません: {FACE_APP_PATH}")
        return

    if not FACE_MODEL_DIR.exists():
        st.warning(
            f"モデルフォルダ {FACE_MODEL_DIR} がありません。"
            "起動.bat を使うと自動で作成・配置されます"
            "（インターネットに繋がっていれば自動ダウンロードも行われます）。"
        )

    # face/app.py は自分専用のページ設定を行うが、統合画面ではページ設定は
    # 上で1回だけ済ませているため、その呼び出しだけ無効化して実行する。
    # （ページのタイトル以外、検出やマスクの動作には一切影響しない）
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        runpy.run_path(str(FACE_APP_PATH), run_name=FACE_RUN_NAME)
    finally:
        st.set_page_config = original_set_page_config


def is_text_backend_alive() -> bool:
    try:
        with urllib.request.urlopen(TEXT_HEALTH_URL, timeout=2) as res:
            return res.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def render_text_app() -> None:
    """word/frontend/index.html を無改変のまま埋め込んで文字モザイク機能を描画する。"""
    st.title("🔤 文字モザイク処理")
    st.caption(
        "写真の中の文字をAI（OCR＋LLM判定）で見つけて隠します。"
        "枠のクリックで個別に解除／再適用、ドラッグで手動追加ができます。"
    )

    alive = is_text_backend_alive()
    if not alive:
        st.error(
            "文字認識バックエンド（http://localhost:8000）に接続できません。\n\n"
            "起動.bat から起動した場合は、別ウィンドウで開いている "
            "「Text Backend (OCR API)」の初期化（PaddleOCRのモデル読み込み）が"
            "終わるのを待ってから、下の再確認ボタンを押してください。"
        )
        st.code(
            "cd " + str(Path(__file__).resolve().parent) + "\n"
            "python -m uvicorn text_backend:app --host 127.0.0.1 --port 8000",
            language="bat",
        )
        if st.button("🔄 接続を再確認する", type="primary"):
            st.rerun()
        return

    st.success("🟢 文字認識バックエンドに接続済みです。")

    st.sidebar.header("⚙️ 文字モザイク設定")
    st.sidebar.caption("細かい設定は下の画面内のツールバーで行います。")
    view_height = st.sidebar.slider(
        "表示エリアの高さ (px)", 600, 2400, 1200, 100,
        help="縦長の写真で下が見切れる場合は大きくしてください。",
    )

    # 既存の index.html を、バックエンドと同じオリジンからそのまま読み込む。
    components.iframe(TEXT_UI_URL, height=view_height, scrolling=True)


# -------------------------------------------------------------------
# 3. 画面本体（モード切替）
# -------------------------------------------------------------------
st.markdown("## 🛡️ AI Privacy Masking Suite")
st.caption("顔モザイクと文字モザイクを1つの画面から使えます。使いたい機能を選んでください。")

mode = st.radio(
    "機能の切り替え",
    ["🙂 顔モザイク", "🔤 文字モザイク"],
    horizontal=True,
    key="suite_mode",
    label_visibility="collapsed",
)

st.markdown("---")

if mode == "🙂 顔モザイク":
    render_face_app()
else:
    render_text_app()
