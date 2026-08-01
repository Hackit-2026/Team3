@echo off
chcp 65001 > nul
title AI Smart Masking Pro - 全自動クリーンセットアップ (MediaPipe 1.0.0 / Python 3.13対応)

echo ===================================================
echo 🧹 1/3: 既存パッケージの全削除・クリーン化を実行中...
echo ===================================================

:: OpenCV関連や旧パッケージの競合を完全にリセット
pip uninstall -y mediapipe opencv-python opencv-python-headless opencv-contrib-python protobuf numpy pillow streamlit > nul 2>&1

:: pipのキャッシュを完全に消去
pip cache purge > nul 2>&1

echo.
echo ===================================================
echo 🔄 2/3: pipの最新化を行っています...
echo ===================================================
python -m pip install --upgrade pip

echo.
echo ===================================================
echo 📦 3/3: パッケージの新規クリーンインストール中...
echo ===================================================
pip install --no-cache-dir -r requirements.txt

if errorlevel 1 (
    echo.
    echo ===================================================
    echo [!] セットアップ中にエラーが発生しました。
    echo 上記のエラーメッセージをご確認ください。
    echo ===================================================
) else (
    echo.
    echo ===================================================
    echo ✨ MediaPipe 1.0.0 クリーンセットアップが完了しました！
    echo アプリを起動できます。
    echo ===================================================
)

pause