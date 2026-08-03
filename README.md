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

- 発表資料URL（必須）
- デモURL（任意）
- デモ動画（任意）
https://github.com/user-attachments/assets/60c0fd4e-17f5-4225-9f2f-383cc1c5c447  

- スクリーンショット（1枚以上推奨）
<img width="450" height="300" alt="スクリーンショット 2026-08-02 223855" src="https://github.com/user-attachments/assets/416c313c-7a1a-4cfd-b139-c59f1890159f" />
<img width="450" height="300" alt="スクリーンショット 2026-08-02 223919" src="https://github.com/user-attachments/assets/bb6c59c1-c097-4fe9-9b64-b03569120447" />
<img width="540" height="360" alt="スクリーンショット 2026-08-02 223940" src="https://github.com/user-attachments/assets/29caf609-afc7-403f-90eb-fb2722fc55d6" />
<img width="360" height="240" alt="スクリーンショット 2026-08-02 224008" src="https://github.com/user-attachments/assets/049c2aa0-374c-4a42-b67d-7974e5e62441" />
<img width="360" height="240" alt="スクリーンショット 2026-08-02 224142" src="https://github.com/user-attachments/assets/6541da3e-a098-40be-8a42-9bd27900b5ee" />



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
- インフラ：Cloudflare Tunne, Windows ローカル実行, 
- その他：

## 今後の展望

今回のLLMによる言語判断ではシステムプロンプト的にモザイクを付与するのかどうかの判断を行っていたので今後はその判断処理の後に、モザイクを外すとした検知文字に対して、RAGに格納したキーワードに合致したものがあった場合には再度モザイクを付けるようにする機能を制作したいと考える。

## セットアップ方法

ローカルで実行する場合の手順を記載してください。

例

```bash
git clone <repository-url>
cd <repository-name>

# 必要なライブラリをインストール
...

# 起動
...
```

## メンバー

| 名前 | 担当 |
|------|------|
|久藤豊也|   PM   |
|石山雅治|   OCR   |
|谷内清吾|顔認識|
|朝井咲陽|顔認識|
|小川輪生|フロントエンド|
