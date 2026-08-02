@echo off
title AI Privacy Masking Suite

cd /d "%~dp0"

if not exist app.py (
    echo app.py が見つかりません
    pause
    exit /b
)

rem --- 顔検出モデルが mp_models に無ければ face フォルダからコピーする ---
if not exist "..\face\mp_models" mkdir "..\face\mp_models"
if not exist "..\face\mp_models\face_landmarker.task" (
    if exist "..\face\face_landmarker.task" copy "..\face\face_landmarker.task" "..\face\mp_models\" > nul
)
if not exist "..\face\mp_models\blaze_face_full_range.tflite" (
    if exist "..\face\blaze_face_full_range.tflite" copy "..\face\blaze_face_full_range.tflite" "..\face\mp_models\" > nul
)

rem --- 文字認識バックエンドが未起動なら、ポート8000で別ウィンドウ起動する ---
netstat -ano | findstr ":8000" | findstr "LISTENING" > nul
if errorlevel 1 (
    echo 文字認識バックエンドを起動します。別ウィンドウが開きます...
    echo   初回はOCRモデルの自動ダウンロードで数分かかることがあります。
    start "Text Backend (OCR API)" cmd /k python -m uvicorn text_backend:app --host 127.0.0.1 --port 8000
) else (
    echo 文字認識バックエンドは既に起動しています。
)

echo.
echo 統合画面を起動しています...
python -m streamlit run app.py

pause
