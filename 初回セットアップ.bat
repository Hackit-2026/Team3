@echo off
chcp 65001 > nul
title AI Smart Masking Pro - クリーンセットアップ (MediaPipe 1.0.0)

echo ===================================================
echo 🧹 古い環境・キャッシュの完全削除を実行しています...
echo ===================================================

:: 既存のライブラリを完全削除
pip uninstall -y mediapipe opencv-python protobuf numpy pillow streamlit > nul 2>&1

:: pipのキャッシュをクリア
pip cache purge > nul 2>&1

echo.
echo ===================================================
echo 📦 MediaPipe 1.0.0 および必要パッケージを新規インストール中...
echo ===================================================

:: 最新のpipに更新
python -m pip install --upgrade pip

:: キャッシュを使わずにクリーンインストール
pip install --no-cache-dir -r requirements.txt

if errorlevel 1 (
    echo.
    echo ===================================================
    echo [!] エラーが発生しました。
    echo 上記のエラーメッセージをご確認ください。
    echo ===================================================
) else (
    echo.
    echo ===================================================
    echo ✨ MediaPipe 1.0.0 のクリーンセットアップが完了しました！
    echo 起動スクリプトまたはショートカットからアプリを起動できます。
    echo ===================================================
)

pause