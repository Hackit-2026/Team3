import os

import requests

# 別端末で動かすローカルLLMを、Cloudflare Tunnel（cloudflared）経由で公開された
# HTTPS URLで呼び出す。デスクトップ側は独自の /api/check_mask エンドポイントを
# 持っており、Ollama標準の /api/generate ではない点に注意（実接続確認で判明）。
# 未設定の間は常に判定不能（＝安全側でモザイク対象）扱いになる。
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.1")

# Cloudflare Tunnel はインターネットを経由するため、同一LAN内より応答が遅くなる
# ことを見込んでタイムアウトを長めに取る。
LOCAL_LLM_TIMEOUT_SEC = float(os.environ.get("LOCAL_LLM_TIMEOUT_SEC", "15"))

# Cloudflare Quick Tunnel のURLは推測困難だが認証なしで誰でも呼び出せてしまうため、
# 合言葉（共有トークン）を設定できるようにする。デスクトップ側のLLMサーバーの前段に
# このトークンを検証するプロキシ（例: 簡易authミドルウェア）を置く運用を想定。
# 未設定なら送信しない（LAN内で完結する接続方式に戻したい場合はこれで問題ない）。
LOCAL_LLM_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "")


def is_configured() -> bool:
    return bool(LOCAL_LLM_BASE_URL)


def _parse_sensitive_ids(data: dict, items: list[dict]) -> set[str]:
    """デスクトップ側は {"results": [{"id":..., "is_sensitive": bool}, ...]} を
    直接返す契約になっている。想定外の形（"error" キー、results 欠落など）は
    安全側（全件センシティブ扱い）に倒す。
    """
    ids = {item["id"] for item in items}

    if "error" in data:
        return set(ids)

    results = data.get("results")
    if not isinstance(results, list):
        return set(ids)

    return {
        entry["id"]
        for entry in results
        if isinstance(entry, dict) and entry.get("is_sensitive") and entry.get("id") in ids
    }


def classify_texts(items: list[dict]) -> dict[str, bool]:
    """items: [{"id": ..., "text": ...}, ...]
    戻り値: {id: is_public}（True＝一般公開情報だと確信できた＝モザイク不要）。
    未接続・通信エラー・タイムアウト・応答解釈不能など、判定できないケースは
    すべて False（＝安全側に倒してモザイク対象）にする。
    """
    if not items:
        return {}

    if not is_configured():
        return {item["id"]: False for item in items}

    headers = {}
    if LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LOCAL_LLM_API_KEY}"

    try:
        response = requests.post(
            f"{LOCAL_LLM_BASE_URL}/api/check_mask",
            json={
                "extracted_texts": [
                    {"id": item["id"], "text": item["text"]} for item in items
                ]
            },
            headers=headers,
            timeout=LOCAL_LLM_TIMEOUT_SEC,
        )
        response.raise_for_status()
        sensitive_ids = _parse_sensitive_ids(response.json(), items)
    except (requests.RequestException, ValueError):
        sensitive_ids = {item["id"] for item in items}

    return {item["id"]: item["id"] not in sensitive_ids for item in items}
