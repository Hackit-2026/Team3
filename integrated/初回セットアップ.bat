@echo off
title AI Privacy Masking Suite - 初回セットアップ

cd /d "%~dp0"

echo ===================================================
echo 1/3: 顔検出モデルの配置を確認しています...
echo ===================================================

if not exist "..\face\mp_models" mkdir "..\face\mp_models"
if not exist "..\face\mp_models\face_landmarker.task" (
    if exist "..\face\face_landmarker.task" copy "..\face\face_landmarker.task" "..\face\mp_models\" > nul
)
if not exist "..\face\mp_models\blaze_face_full_range.tflite" (
    if exist "..\face\blaze_face_full_range.tflite" copy "..\face\blaze_face_full_range.tflite" "..\face\mp_models\" > nul
)
echo   配置の確認が完了しました。

echo.
echo ===================================================
echo 2/3: pip の最新化を行っています...
echo ===================================================
python -m pip install --upgrade pip

echo.
echo ===================================================
echo 3/3: パッケージをインストール中です。数分から十数分かかります...
echo ===================================================
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [!] セットアップ中にエラーが発生しました。上の内容をご確認ください。
) else (
    echo.
    echo セットアップが完了しました。起動.bat でアプリを起動できます。
)

pause
