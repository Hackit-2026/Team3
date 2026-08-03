# チーム名

DDM👾

# プロダクト名

機密情報保護カメラ「カクカク」

## 概要

SNSなどに画像を上げる際に個人情報である顔の画像と書類などに書かれている機密情報をモザイクにより保護するカメラシステムの開発を行った。
顔の認識では最大5名のホワイトリストを登録し、ホワイトリストに合致する人間はモザイクがかからないようにする機能を付けた。
文字の認識ではOCRを用いて文字を抽出したのちにローカルのLLMに渡し、世間一般的に知られている情報はモザイクを外すようにした。

## デモ

以下を掲載してください。

- 発表資料URL（必須）https://kitm365-my.sharepoint.com/:p:/g/personal/c1514542_st_kanazawa-it_ac_jp/IQDLbo9CmvO3Qb45oXDk_0KsAXRuKi8zoPWdCob1n7d3S-w?e=TLZZBI
- デモ動画（任意）
  
https://github.com/user-attachments/assets/60c0fd4e-17f5-4225-9f2f-383cc1c5c447  

- スクリーンショット（1枚以上推奨）
<img width="450" height="300" alt="スクリーンショット 2026-08-02 223855" src="https://github.com/user-attachments/assets/416c313c-7a1a-4cfd-b139-c59f1890159f" />
<img width="450" height="300" alt="スクリーンショット 2026-08-02 223919" src="https://github.com/user-attachments/assets/bb6c59c1-c097-4fe9-9b64-b03569120447" />
<img width="450" height="300" alt="スクリーンショット 2026-08-02 223940" src="https://github.com/user-attachments/assets/29caf609-afc7-403f-90eb-fb2722fc55d6" />
<br>
<img width="450" height="300" alt="スクリーンショット 2026-08-02 224008" src="https://github.com/user-attachments/assets/049c2aa0-374c-4a42-b67d-7974e5e62441" />
<img width="450" height="300" alt="スクリーンショット 2026-08-02 224142" src="https://github.com/user-attachments/assets/6541da3e-a098-40be-8a42-9bd27900b5ee" />
<img width="450" height="300" alt="スクリーンショット 2026-08-03 091715" src="https://github.com/user-attachments/assets/99195f72-f06c-4990-a5c2-9c8e2d67ac82" />
<img width="450" height="300" alt="スクリーンショット 2026-08-03 002835" src="https://github.com/user-attachments/assets/77eb94d2-d269-40f4-bbe5-fb467f0db156" />
<img width="450" height="300" alt="スクリーンショット 2026-08-03 092005" src="https://github.com/user-attachments/assets/524f37ee-143e-4a14-bb1e-9bc3830db4d9" />
<img width="450" height="300" alt="スクリーンショット 2026-08-03 092138" src="https://github.com/user-attachments/assets/859e171f-93b0-437d-96a2-4670318e0759" />

## システム構成

<img width="503" height="407" alt="image" src="https://github.com/user-attachments/assets/ea0fcc51-fae1-4102-ab9c-63d7944942b8" />


## 背景・課題

現在SNSの流行りとして加工を行わないBeRealやSetLogのようなものがz世代に人気になっています。
しかしそのSNSを巡って、個人情報の漏洩や顔写真の流出、機密情報が書かれていた書類が映り込み流出するという問題点がある。
そのような問題が発生する原因にインターネットリテラシ―の個人間での差がある点に注目した。
今後、いかに気を付けるように呼び掛けても一律の基準を設けて各人が実施することは難しいと考えたため今回のアプリケーションを制作した。

## 主な機能

- 機能1：画像に映った人間の顔の検知と、ホワイトリストによる除外処理
- 機能2：画像に映った文字の内容検知と、ローカルのLLMによる選択処理
- 機能3：画像に映ったモザイク対応箇所に対するモザイクデザインの選択

## 工夫した点・こだわった点

今回のシステム制作では機密情報の安全を何よりもの最優先にした。
外部APIやLLMAPIを用いて実装すれば今回制作した物よりも品質がいい物は制作できると考える。
しかし私たちは機密情報の安全を最優先事項に据えて処理のほとんどを実行するデバイスで行うことにした。
唯一の外部接続を行う点に関してはOCRを用いて抽出した単語を機密情報かどうなのかを判断する際にローカルLLMに送信すときのみである。
この時CloudflareとFastAPIを用いることで防御力とAPI通信を追加した。
またLLMによる機密情報かどうかの判断はローカルのLLMに知識があればそれは公共の情報であるとしてモザイクを外し、無い場合は機密情報として扱うことでモザイクの付与性を担保した。
また顔の検知に際しては画像を 1×1 / 2×2 / 3×3 に分割し、各タイルを30%オーバーラップさせながら個別に検出し小さく写った顔は画像全体では見逃されるため、切り出して拡大相当の状態で検出させている。
特にフェイルセーフを重要視して基本的に怪しい物はモザイクを付けるようにした。

## 使用技術

- フロントエンド：Streamlit 1.60,HTML / CSS / JavaScript,HTML Canvas API,File API / FormData / Fetch API,Streamlit Components
- バックエンド：Python 3.13, FastAPI 0.141, uvicorn 0.52, StaticFiles, CORS Middleware
- AI / API：MediaPipe Tasks API 1.0, BlazeFace Full Range, MobileNetV3-Small, PaddleOCR 3.7, PaddlePaddle 3.3, Ollama, gemma-4-e4b-it-q5_k_m, 
- データベース：利用していない
- インフラ：Cloudflare Tunne, Windows ローカル実行

## 今後の展望

今回のLLMによる言語判断ではシステムプロンプト的にモザイクを付与するのかどうかの判断を行っていたので今後はその判断処理の後に、モザイクを外すとした検知文字に対して、RAGに格納したキーワードに合致したものがあった場合には再度モザイクを付けるようにする機能を制作したいと考える。

## セットアップ方法

ローカルで実行する場合の手順を記載してください。

--------------------------------------------------------------------
 1. フォルダ構成
--------------------------------------------------------------------
  FinalCode\ <br>
&ensp;    ├ backend\             バックエンド（FastAPI） <br>
&ensp;    │   ├ main.py            APIエンドポイント一式 <br>
&ensp;    │   ├ face_engine.py      顔検出・ホワイトリスト照合ロジック <br>
&ensp;    │   ├ llm_classifier.py   文字列の公開/非公開判定（ローカルLLM連携） <br>
&ensp;    │   ├ requirements.txt    Python依存パッケージ一覧 <br>
&ensp;    │   ├ .env.example        環境変数のひな形（.envにコピーして使う） <br>
&ensp;    │   ├ mp_models\          （初回起動時に自動生成・ダウンロード） <br>
&ensp;    │   └ whitelist_store.pkl （初回登録時に自動生成） <br>
&ensp;    ├ src\                  フロントエンド（React）のソースコード <br>
&ensp;    ├ public\icon.png       アプリのアイコン <br>
&ensp;    ├ package.json / vite.config.ts / tsconfig*.json など <br>
&ensp;    └ 手順書.txt            （このファイル） <br>


--------------------------------------------------------------------
 2. 初回だけ行う準備
--------------------------------------------------------------------
  必要なもの: <br>
&emsp; ・Node.js 18以上 <br>
&emsp; ・Python 3.11〜3.13程度（mediapipe / paddleocr が対応するバージョン） <br>
&emsp; ・Git <br>

  (1) バックエンドの準備　<br>
&emsp;&emsp;  cd FinalCode\backend　<br>
&emsp;&emsp;  python -m venv venv　<br>
&emsp;&emsp;  venv\Scripts\activate　<br>
&emsp;&emsp;  pip install -r requirements.txt　<br>

      ローカルLLMによる文字列の公開/非公開判定を使う場合のみ、
      .env.example を .env にコピーして LOCAL_LLM_BASE_URL 等を設定します。
      未設定でも動きます。その場合は「判定できない＝安全側」に倒れ、
      検出した文字列はすべて private（モザイクON）扱いになります。

  (2) フロントエンドの準備 <br>
&emsp;&emsp;  cd FinalCode <br>
&emsp;&emsp;  npm install <br>

  (3) 顔検出モデルについて
      初回に顔検出処理を実行したタイミングで、以下のモデルファイルを
      自動でダウンロードします（backend\mp_models\ に保存）。 <br>
&ensp; ・ face_landmarker.task <br>
&ensp; ・ blaze_face_full_range.tflite <br>
&ensp; ・ mobilenet_v3_small.tflite <br>
&ensp;　このときだけインターネット接続が必要です。以降はキャッシュされます。
...

# 起動

--------------------------------------------------------------------
 3. 起動のしかた
--------------------------------------------------------------------
  ターミナルを2つ開いて、それぞれで実行します。

    1つ目（バックエンド）:
        cd FinalCode\backend
        venv\Scripts\activate
        python -m uvicorn main:app --host 127.0.0.1 --port 8000

    2つ目（フロントエンド）:
        cd FinalCode
        npm run dev

  フロントエンドは既定で http://localhost:8000 のバックエンドを見に
  行きます。バックエンドを別のホスト/ポートで動かす場合は、
  FinalCode 直下に .env.local を作り <br>
&emsp;&emsp;        VITE_API_BASE_URL=http://<ホスト>:<ポート> <br>
  を設定してください。

  npm run dev が表示するURL（通常 http://localhost:5173）をブラウザで
  開けば使用できます。
  

## メンバー

| 名前 | 担当 |
|------|------|
|久藤豊也|   PM   |
|石山雅治|   OCR   |
|谷内清吾|顔認識|
|朝井咲陽|顔認識|
|小川輪生|フロントエンド|
