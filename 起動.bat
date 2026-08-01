@echo off
chcp 65001 > nul

title AI Smart Masking

if not exist app.py (
    echo app.py が見つかりません
    pause
    exit /b
)

if not exist mp_models (
    echo mp_models フォルダが見つかりません
    pause
    exit /b
)

echo AI Smart Masking を起動中...

python -m streamlit run app.py

pause