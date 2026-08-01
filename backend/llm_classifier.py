import os

import requests

# 別端末（同一ネットワーク上）で動かすローカルLLM（Ollama互換API）のエンドポイント。
# 例: "http://192.168.1.50:11434"。未設定の間は常に判定不能（＝安全側でモザイク対象）扱いになる。
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.1")
LOCAL_LLM_TIMEOUT_SEC = float(os.environ.get("LOCAL_LLM_TIMEOUT_SEC", "8"))

# 「一般に広く知られている情報だけを公開扱いにする」という発想の反転を実現するための
# システムプロンプト。RAGで事前登録されていない社外秘語句も、これで一定拾えるようにする。
SYSTEM_PROMPT = """あなたは、カメラで撮影した写真からOCRで抽出された短いテキスト断片が、
モザイクで隠すべき情報かどうかを判定するプライバシー保護アシスタントです。

次のテキストが「一般に広く知られている公開情報」（例: 一般的な単語や慣用句、日付や曜日、
有名な地名・建物名・ブランド名、公共の看板文言、汎用的なラベル文言など）であると
確信できる場合にのみ「PUBLIC」と回答してください。

個人名、住所の一部、電話番号、メールアドレス、社内のプロジェクト名やコードネーム、
財務情報、その他特定の個人や組織に関わる可能性がある情報だと判断した場合、
あるいは短すぎる・文脈が不明で判断に迷う場合は、必ず「SENSITIVE」と回答してください。

回答は "PUBLIC" または "SENSITIVE" の1単語のみとし、他の文字は一切含めないでください。"""


def is_configured() -> bool:
    return bool(LOCAL_LLM_BASE_URL)


def is_public_text(text: str) -> bool:
    """一般公開情報だとLLMが確信した場合のみ True（モザイク不要）を返す。
    未接続・通信エラー・タイムアウト・応答不明など、判定できないケースは
    すべて False（＝安全側に倒してモザイク対象）にする。
    """
    if not is_configured():
        return False

    try:
        response = requests.post(
            f"{LOCAL_LLM_BASE_URL}/api/generate",
            json={
                "model": LOCAL_LLM_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": text,
                "stream": False,
            },
            timeout=LOCAL_LLM_TIMEOUT_SEC,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip().upper()
        return answer.startswith("PUBLIC")
    except (requests.RequestException, ValueError):
        return False
