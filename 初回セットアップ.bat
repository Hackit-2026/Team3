@echo off
chcp 65001 > nul

echo ==========================================
echo ライブラリを更新しています...
echo ==========================================

pip uninstall -y mediapipe opencv-python protobuf numpy pillow streamlit

pip install -r requirements.txt

echo.
echo ==========================================
echo セットアップ完了
echo ==========================================
pause