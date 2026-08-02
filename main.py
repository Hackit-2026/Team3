from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json  # LLMの出力をパースするために追加

app = FastAPI()

# ハッカソン用設定: どこからでもアクセスできるようにCORSを全許可
# allow_credentials=True と allow_origins=["*"] の競合エラーを避けるため False に変更
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# フロントエンドから受け取るデータの型定義
class OCRResult(BaseModel):
    id: str
    text: str

class OCRRequest(BaseModel):
    extracted_texts: list[OCRResult]

# Ollamaのエンドポイントと使用モデル
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma-4-e4b-it-q5_k_m"  # 読み込ませたモデル名

@app.post("/api/check_mask")
def check_mask(request: OCRRequest):
    # LLMに渡すプロンプトを構築（出力形式を具体的に指定）
    prompt = (
        "以下のテキストリストの各項目について、個人情報や機密情報"
        "（名前、住所、電話番号、組織名など。辞書に載っているような一般知識は除く）を"
        "含むかどうかを判定してください。\n"
        "必ず、渡された全てのIDについて、次の形式のJSONで回答してください。"
        "他の形式や説明文は一切含めないでください。\n"
        '{"results": [{"id": "ID1", "is_sensitive": true}, {"id": "ID2", "is_sensitive": false}]}\n\n'
    )
    for item in request.extracted_texts:
        prompt += f"ID: {item.id}, Text: {item.text}\n"

    # Ollamaへのリクエストデータ
    ollama_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"  # LLMにJSONフォーマットでの出力を強制
    }

    try:
        # Ollamaに推論をリクエスト
        response = requests.post(OLLAMA_URL, json=ollama_payload)
        response.raise_for_status()
        
        # レスポンスをOllamaの生データから必要な文字列だけ取り出し、JSONにパース
        llm_output = response.json()["response"]
        return json.loads(llm_output)  # {"results": [...]} の形式で直接返す
        
    except json.JSONDecodeError:
        # LLMが指定したJSON形式を守らなかった場合の安全対策
        return {"error": "LLMが正しいJSONを返しませんでした", "raw_output": llm_output}
    except Exception as e:
        return {"error": str(e)}