"""文字モザイク機能のサーバー（既存APIをそのまま利用し、画面配信だけを追加する）。

統合にあたって word/backend/main.py と word/frontend/index.html は一切書き換えない。
このファイルは、

  1. 既存の FastAPI アプリ（/api/detect-text, /api/health）をそのまま読み込み
  2. 既存の index.html を /ui として同じポートから配信する

という2つのことだけを行う薄いラッパーである。
統合画面（app.py）はこの /ui を iframe で埋め込むため、文字モザイク側の
UI・操作・検出処理は元のアプリと完全に同一のコードが動作する。

起動:
    python -m uvicorn text_backend:app --host 127.0.0.1 --port 8000
"""

import sys
from pathlib import Path

from fastapi.staticfiles import StaticFiles

HACKIT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = HACKIT_DIR / "word" / "backend"
FRONTEND_DIR = HACKIT_DIR / "word" / "frontend"

# 既存の main.py は同じフォルダにある llm_classifier を `import llm_classifier` で
# 読み込むため、そのフォルダを import パスの先頭に追加してから読み込む。
# （main.py 内の load_dotenv() は main.py の場所を基準に .env を探すので、
#   起動時のカレントフォルダがどこであっても word/backend/.env が読まれる）
sys.path.insert(0, str(BACKEND_DIR))

from main import app  # noqa: E402  既存APIをそのまま流用する

# 既存フロントエンドを、APIと同じ http://localhost:8000 から配信する。
# 同一オリジンになるため、統合画面のiframeに埋め込んでも
# fetch（検出リクエスト）も画像のダウンロードも制限を受けない。
app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="text-ui")
