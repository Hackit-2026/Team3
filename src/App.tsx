import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Camera as CameraIcon,
  Users,
  ChevronLeft,
  ChevronDown,
  X,
  Plus,
  Download,
  Phone,
  MapPin,
  FileText,
  EyeOff,
  SlidersHorizontal,
  ListChecks,
  Loader2,
  Pencil,
  Trash2,
} from 'lucide-react';

// ==========================================
// 型定義
// ==========================================
type ScreenType = 'home' | 'whitelist' | 'camera' | 'processing';
type MosaicId = 'pixel' | 'blur' | 'emoji' | 'fill' | 'tile';

interface FaceData {
  id: number;
  name: string;
  imgUrl: string; // BlobURLなどを想定
}

interface DetectedString {
  id: number;
  text: string;
  type: 'phone' | 'address' | 'other' | 'ignore';
  label: string;
}

// ==========================================
// 共通デザインパーツ（クラス定義）
// ==========================================
const card = 'bg-white rounded-2xl shadow-sm ring-1 ring-gray-200';
const primaryBtn =
  'inline-flex items-center justify-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-bold text-white shadow-sm shadow-emerald-500/30 transition hover:bg-emerald-600 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-gray-300 disabled:shadow-none';
const secondaryBtn =
  'inline-flex items-center justify-center gap-2 rounded-2xl border-2 border-emerald-500 bg-white px-4 py-3 text-sm font-bold text-emerald-600 shadow-sm transition hover:bg-emerald-50 active:scale-[0.98]';
const ghostBtn =
  'inline-flex items-center gap-1 text-sm font-semibold text-gray-600 transition hover:text-gray-900';
const inputClass =
  'w-full rounded-xl border border-gray-400 bg-white px-3 py-2.5 text-sm text-gray-800 shadow-sm outline-none transition focus:border-emerald-400 focus:ring-4 focus:ring-emerald-50';
const labelClass = 'mb-1.5 block text-xs font-semibold text-gray-600';

// ==========================================
// メインコンポーネント (App)
// ==========================================
export default function App() {
  const [currentScreen, setCurrentScreen] = useState<ScreenType>('home');
  const [selectedMosaic, setSelectedMosaic] = useState<MosaicId>('pixel');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  // グローバルな状態（ホワイトリスト）
  const [whitelistEnabled, setWhitelistEnabled] = useState<boolean>(true);
  const [whitelistFaces, setWhitelistFaces] = useState<FaceData[]>([
    { id: 1, name: 'Taro', imgUrl: '👤' }, // モックデータ
  ]);

  const navigate = (screen: ScreenType) => setCurrentScreen(screen);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 font-sans lg:py-8">
      <div className="relative mx-auto flex min-h-screen w-full max-w-6xl flex-col bg-gray-50 lg:min-h-[calc(100vh-4rem)] lg:rounded-3xl lg:shadow-xl lg:ring-1 lg:ring-black/5">
        {currentScreen === 'home' && (
          <HomeScreen
            navigate={navigate}
            selectedMosaic={selectedMosaic}
            setSelectedMosaic={setSelectedMosaic}
            setUploadedImage={setUploadedImage}
          />
        )}
        {currentScreen === 'whitelist' && (
          <WhitelistScreen
            navigate={navigate}
            enabled={whitelistEnabled}
            setEnabled={setWhitelistEnabled}
            faces={whitelistFaces}
            setFaces={setWhitelistFaces}
          />
        )}
        {currentScreen === 'camera' && (
          <CameraScreen navigate={navigate} setUploadedImage={setUploadedImage} />
        )}
        {currentScreen === 'processing' && (
          <ProcessingScreen
            navigate={navigate}
            selectedMosaic={selectedMosaic}
            uploadedImage={uploadedImage}
          />
        )}
      </div>
    </div>
  );
}

// ==========================================
// 【画面1】ホーム画面
// ==========================================
const MOSAIC_TYPES: { id: MosaicId; label: string }[] = [
  { id: 'pixel', label: 'ピクセルモザイク' },
  { id: 'blur', label: 'ぼかし（ブラー）' },
  { id: 'emoji', label: '絵文字スタンプ' },
  { id: 'fill', label: '塗りつぶし' },
  { id: 'tile', label: 'タイル状モザイク（グリッド）' },
];

interface HomeScreenProps {
  navigate: (screen: ScreenType) => void;
  selectedMosaic: MosaicId;
  setSelectedMosaic: (mosaic: MosaicId) => void;
  setUploadedImage: (url: string | null) => void;
}

const HomeScreen: React.FC<HomeScreenProps> = ({
  navigate,
  selectedMosaic,
  setSelectedMosaic,
  setUploadedImage,
}) => {
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const imgUrl = URL.createObjectURL(e.target.files[0]);
      setUploadedImage(imgUrl);
      navigate('processing');
    }
  };

  return (
    <div className="flex flex-1 flex-col items-center justify-between p-6 lg:p-12">
      <div className="flex w-full max-w-md flex-col items-center pt-10 lg:max-w-lg lg:pt-20">
        <img
          src="/icon.png"
          alt="カクカク"
          className="mb-5 h-24 w-24 rounded-3xl object-cover shadow-lg shadow-emerald-500/30 lg:h-28 lg:w-28"
        />
        <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 lg:text-4xl">カクカク</h1>
        <p className="mt-2 mb-12 text-base text-gray-500">写真の個人情報を、その場でマスク</p>

        <div className="mb-5 w-full">
          <label className={`${labelClass} text-sm`}>使用するモザイク</label>
          <div className="relative">
            <select
              value={selectedMosaic}
              onChange={(e) => setSelectedMosaic(e.target.value as MosaicId)}
              className={`${inputClass} appearance-none py-3.5 pr-9 text-base font-medium`}
            >
              {MOSAIC_TYPES.map((type) => (
                <option key={type.id} value={type.id}>
                  {type.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-500" />
          </div>
        </div>

        <button onClick={() => navigate('whitelist')} className={`${secondaryBtn} mb-3 w-full py-3.5 text-base`}>
          <Users className="h-5 w-5" />
          ホワイトリストを管理
        </button>

        <label className={`${primaryBtn} w-full cursor-pointer py-3.5 text-base`}>
          <Upload className="h-5 w-5" />
          写真をアップロード
          <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
        </label>
      </div>

      <div className="flex justify-center pb-6 pt-10">
        <button
          onClick={() => navigate('camera')}
          className="flex h-20 w-20 items-center justify-center rounded-full bg-emerald-500 text-white shadow-xl shadow-emerald-500/40 transition hover:bg-emerald-600 active:scale-95"
          aria-label="カメラを起動"
        >
          <CameraIcon className="h-8 w-8" strokeWidth={2.2} />
        </button>
      </div>
    </div>
  );
};

// ==========================================
// 【画面2】ホワイトリスト画面
// ==========================================
interface WhitelistScreenProps {
  navigate: (screen: ScreenType) => void;
  enabled: boolean;
  setEnabled: (val: boolean) => void;
  faces: FaceData[];
  setFaces: (faces: FaceData[]) => void;
}

const WhitelistScreen: React.FC<WhitelistScreenProps> = ({
  navigate,
  enabled,
  setEnabled,
  faces,
  setFaces,
}) => {
  const [name, setName] = useState<string>('');
  const [previewImg, setPreviewImg] = useState<string | null>(null);
  const [nameError, setNameError] = useState<boolean>(false);
  const [editingId, setEditingId] = useState<number | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPreviewImg(URL.createObjectURL(e.target.files[0]));
    }
  };

  const resetForm = () => {
    setEditingId(null);
    setName('');
    setPreviewImg(null);
    setNameError(false);
  };

  const handleSaveFace = () => {
    if (!name) {
      setNameError(true);
      return;
    }
    // TODO: ここにコア処理へホワイトリスト登録・更新するロジックを追加
    if (editingId !== null) {
      setFaces(
        faces.map((f) => (f.id === editingId ? { ...f, name, imgUrl: previewImg || f.imgUrl } : f)),
      );
    } else {
      const newFace: FaceData = { id: Date.now(), name, imgUrl: previewImg || '👤' };
      setFaces([...faces, newFace]);
    }
    resetForm();
  };

  const handleEditStart = (face: FaceData) => {
    setEditingId(face.id);
    setName(face.name);
    setPreviewImg(face.imgUrl.startsWith('blob:') ? face.imgUrl : null);
    setNameError(false);
  };

  const handleDeleteFace = (id: number) => {
    // TODO: ここにコア処理へホワイトリスト削除を反映するロジックを追加
    if (!window.confirm('この顔をホワイトリストから削除しますか？')) return;
    setFaces(faces.filter((f) => f.id !== id));
    if (editingId === id) resetForm();
  };

  return (
    <div className="flex flex-1 flex-col items-center p-5 lg:p-10">
      <div className="w-full max-w-xl">
      <div className="mb-6 flex items-center gap-3">
        <button onClick={() => navigate('home')} className={ghostBtn}>
          <ChevronLeft className="h-4 w-4" />
          戻る
        </button>
        <h2 className="text-lg font-bold text-gray-900">ホワイトリスト</h2>
      </div>

      <div className={`${card} mb-5 flex items-center justify-between p-4`}>
        <div>
          <p className="text-sm font-bold text-gray-800">ホワイトリストを有効化</p>
          <p className="text-xs text-gray-400">登録した顔にはモザイクをかけません</p>
        </div>
        <button
          onClick={() => setEnabled(!enabled)}
          className={`flex h-7 w-12 shrink-0 items-center rounded-full p-1 transition-colors ${
            enabled ? 'bg-emerald-500' : 'bg-gray-200'
          }`}
          aria-pressed={enabled}
        >
          <div
            className={`h-5 w-5 transform rounded-full bg-white shadow-md transition-transform ${
              enabled ? 'translate-x-5' : ''
            }`}
          />
        </button>
      </div>

      <div className={`${card} mb-6 p-4`}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-bold text-gray-800">{editingId !== null ? '編集中' : '新規追加'}</h3>
          {editingId !== null && (
            <button onClick={resetForm} className={ghostBtn}>
              <X className="h-3.5 w-3.5" />
              キャンセル
            </button>
          )}
        </div>
        <label className="mb-3 flex w-full cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-gray-200 py-5 text-center transition hover:border-emerald-300 hover:bg-emerald-50/40">
          {previewImg ? (
            <img src={previewImg} alt="プレビュー" className="h-16 w-16 rounded-full border object-cover" />
          ) : (
            <Upload className="h-6 w-6 text-gray-400" />
          )}
          <span className="text-xs font-semibold text-gray-500">顔写真をアップロード</span>
          <input type="file" accept="image/*" className="hidden" onChange={handleImageSelect} />
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="名前を入力"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (nameError) setNameError(false);
            }}
            className={`${inputClass} flex-1 ${nameError ? 'border-red-300 focus:border-red-400 focus:ring-red-50' : ''}`}
          />
          <button onClick={handleSaveFace} className={`${primaryBtn} px-4`}>
            {editingId !== null ? <Pencil className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
            {editingId !== null ? '更新' : '追加'}
          </button>
        </div>
        {nameError && <p className="mt-1.5 text-xs font-medium text-red-500">名前を入力してください</p>}
      </div>

      <h3 className="mb-3 text-sm font-bold text-gray-800">登録済みの顔（{faces.length}）</h3>
      {faces.length > 0 ? (
        <div className="grid grid-cols-3 gap-3">
          {faces.map((face) => (
            <div
              key={face.id}
              className={`${card} group relative flex flex-col items-center gap-2 p-3 ${
                editingId === face.id ? 'ring-2 ring-emerald-400' : ''
              }`}
            >
              <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                <button
                  onClick={() => handleEditStart(face)}
                  aria-label={`${face.name}を編集`}
                  className="flex h-6 w-6 items-center justify-center rounded-full bg-white text-gray-500 shadow ring-1 ring-gray-200 hover:text-emerald-600"
                >
                  <Pencil className="h-3 w-3" />
                </button>
                <button
                  onClick={() => handleDeleteFace(face.id)}
                  aria-label={`${face.name}を削除`}
                  className="flex h-6 w-6 items-center justify-center rounded-full bg-white text-gray-500 shadow ring-1 ring-gray-200 hover:text-red-600"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded-full bg-gray-100 text-2xl">
                {face.imgUrl.startsWith('blob:') ? (
                  <img src={face.imgUrl} className="h-full w-full object-cover" alt={face.name} />
                ) : (
                  face.imgUrl
                )}
              </div>
              <span className="w-full truncate text-center text-xs font-bold text-gray-700">
                {face.name}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="py-6 text-center text-sm text-gray-400">登録済みの顔はまだありません</p>
      )}
      </div>
    </div>
  );
};

// ==========================================
// 【画面3】撮影画面
// ==========================================
interface CameraScreenProps {
  navigate: (screen: ScreenType) => void;
  setUploadedImage: (url: string | null) => void;
}

const CameraScreen: React.FC<CameraScreenProps> = ({ navigate, setUploadedImage }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error('カメラエラー', err);
      }
    };
    startCamera();
    return () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const handleCapture = () => {
    if (!videoRef.current) return;
    // ビデオ映像をCanvasに描画して画像化
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0);
      setUploadedImage(canvas.toDataURL('image/png'));
      navigate('processing');
    }
  };

  return (
    <div className="relative flex-1 bg-black">
      <button
        onClick={() => navigate('home')}
        className="absolute left-5 top-5 z-10 flex items-center gap-1.5 rounded-full bg-black/50 px-4 py-2 text-sm font-bold text-white backdrop-blur"
      >
        <X className="h-4 w-4" />
        キャンセル
      </button>
      <video ref={videoRef} autoPlay playsInline className="h-full w-full object-cover" />

      {/* ガイドグリッド */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="relative h-72 w-56 rounded-[2rem] border-4 border-emerald-400/70">
          <div className="absolute inset-0 flex flex-col justify-evenly">
            <div className="h-px w-full bg-emerald-400/30" />
            <div className="h-px w-full bg-emerald-400/30" />
          </div>
          <div className="absolute inset-0 flex justify-evenly">
            <div className="h-full w-px bg-emerald-400/30" />
            <div className="h-full w-px bg-emerald-400/30" />
          </div>
        </div>
      </div>

      <div className="absolute bottom-10 flex w-full justify-center">
        <button
          onClick={handleCapture}
          className="h-[72px] w-[72px] rounded-full border-[5px] border-emerald-400 bg-white shadow-lg transition active:scale-95"
          aria-label="撮影する"
        />
      </div>
    </div>
  );
};

// ==========================================
// 【画面4】モザイク処理画面
// ==========================================
interface ProcessingScreenProps {
  navigate: (screen: ScreenType) => void;
  selectedMosaic: MosaicId;
  uploadedImage: string | null;
}

const STRING_TYPE_STYLE: Record<DetectedString['type'], { badge: string; icon: React.ReactNode }> = {
  phone: { badge: 'bg-red-50 text-red-600', icon: <Phone className="h-3 w-3" /> },
  address: { badge: 'bg-amber-50 text-amber-700', icon: <MapPin className="h-3 w-3" /> },
  other: { badge: 'bg-gray-100 text-gray-600', icon: <FileText className="h-3 w-3" /> },
  ignore: { badge: 'bg-gray-50 text-gray-400', icon: <EyeOff className="h-3 w-3" /> },
};

const ProcessingScreen: React.FC<ProcessingScreenProps> = ({ navigate, selectedMosaic, uploadedImage }) => {
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processedImage, setProcessedImage] = useState<string | null>(uploadedImage);

  // 検出結果リストのモックデータ（コア処理完了後にこれを更新する想定）
  const [detectedStrings, setDetectedStrings] = useState<DetectedString[]>([
    { id: 1, text: '090-XXXX-XXXX', type: 'phone', label: '機密（電話番号）' },
    { id: 2, text: '東京都千代田区...', type: 'address', label: '個人情報（住所）' },
  ]);
  // setProcessedImage / setDetectedStrings は下記 handleApplyProcess のコア処理連携で
  // 使う想定のため、現時点では未使用でも残す（noUnusedLocals対策で明示的に参照）。
  void setProcessedImage;
  void setDetectedStrings;

  // モザイク設定のState群（これらをコア処理に渡す）
  const [detectMode, setDetectMode] = useState<string>('顔 + 文字列');
  const [maskTarget, setMaskTarget] = useState<string>('未登録の顔のみ');
  const [maskShape, setMaskShape] = useState<string>('長方形');

  const [intensity, setIntensity] = useState<number>(50);
  const [fillColor, setFillColor] = useState<string>('#16a34a');
  const [emojiConfig, setEmojiConfig] = useState({ text: '😎', scale: 100, rotate: 0, offsetX: 0, offsetY: 0 });

  // コア処理実行フック
  const handleApplyProcess = async () => {
    setIsProcessing(true);

    try {
      // TODO: ここに既存のコア処理（画像解析・モザイク適用）を呼び出します。
      // 引数として uploadedImage や 各種State(detectMode, emojiConfig等) を渡してください。
      // 例: const resultUrl = await coreProcess(uploadedImage, { ... });

      // モックとして1秒待機
      await new Promise((res) => setTimeout(res, 1000));

      // TODO: コア処理が終わったら結果URLを setProcessedImage にセットします
      // setProcessedImage(resultUrl);
    } catch (error) {
      console.error('処理エラー', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // 保存処理フック
  const handleSave = () => {
    if (!processedImage) return;
    // TODO: ここに保存ロジック（ダウンロード等）
    const link = document.createElement('a');
    link.href = processedImage;
    link.download = 'privacy_masked.png';
    link.click();
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-gray-100 bg-white/80 p-4 backdrop-blur">
        <button onClick={() => navigate('home')} className={ghostBtn}>
          <ChevronLeft className="h-4 w-4" />
          戻る
        </button>
        <button onClick={handleSave} className={`${primaryBtn} px-4 py-2`}>
          <Download className="h-4 w-4" />
          保存
        </button>
      </div>

      <div className="flex flex-1 flex-col overflow-hidden lg:flex-row">
        <div className="relative flex flex-1 items-center justify-center bg-gray-200 p-4 lg:p-8">
          {processedImage ? (
            <div className="relative max-h-[50vh] w-full max-w-sm overflow-hidden rounded-2xl bg-gray-300 shadow-inner lg:max-h-[70vh] lg:max-w-xl">
              <img src={processedImage} alt="処理結果" className="h-auto max-h-[50vh] w-full object-contain lg:max-h-[70vh]" />

              {isProcessing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/45 font-bold text-white backdrop-blur-sm">
                  <Loader2 className="h-8 w-8 animate-spin text-emerald-400" />
                  処理中...
                </div>
              )}
            </div>
          ) : (
            <span className="text-sm text-gray-400">画像がありません</span>
          )}
        </div>

        <div className="max-h-[45vh] overflow-y-auto rounded-t-3xl bg-white p-4 shadow-[0_-8px_24px_-8px_rgba(0,0,0,0.12)] lg:max-h-none lg:w-96 lg:shrink-0 lg:rounded-none lg:border-l lg:border-gray-100 lg:shadow-none">
        <div className="mx-auto mb-3 h-1.5 w-10 rounded-full bg-gray-200 lg:hidden" />

        {/* コア処理実行ボタン */}
        <button onClick={handleApplyProcess} disabled={isProcessing} className={`${primaryBtn} mb-4 w-full`}>
          {isProcessing ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              処理中...
            </>
          ) : (
            '設定を適用してプレビュー更新'
          )}
        </button>

        <details className="group mb-3 overflow-hidden rounded-2xl border border-gray-100" open>
          <summary className="flex cursor-pointer list-none items-center justify-between bg-gray-50 p-3 text-sm font-bold text-gray-800">
            <span className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-emerald-500" />
              マスク・検出設定
            </span>
            <ChevronDown className="h-4 w-4 text-gray-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="space-y-4 p-3.5 text-sm">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelClass}>検出モード</label>
                <select value={detectMode} onChange={(e) => setDetectMode(e.target.value)} className={inputClass}>
                  <option>顔 + 文字列</option>
                  <option>顔のみ</option>
                  <option>文字列のみ</option>
                </select>
              </div>
              <div>
                <label className={labelClass}>マスク対象</label>
                <select value={maskTarget} onChange={(e) => setMaskTarget(e.target.value)} className={inputClass}>
                  <option>すべて</option>
                  <option>未登録の顔のみ</option>
                </select>
              </div>
            </div>

            <div>
              <label className={labelClass}>マスクの形状</label>
              <select value={maskShape} onChange={(e) => setMaskShape(e.target.value)} className={inputClass}>
                <option>長方形</option>
                <option>円形</option>
                <option>顔の輪郭に合わせる</option>
              </select>
            </div>

            <div className="rounded-xl border border-gray-100 bg-gray-50 p-3.5">
              <p className="mb-3 border-b border-gray-200 pb-2 text-xs font-bold text-emerald-700">
                スタイル詳細設定（{MOSAIC_TYPES.find((m) => m.id === selectedMosaic)?.label}）
              </p>

              {selectedMosaic === 'emoji' ? (
                <div className="space-y-3">
                  <div>
                    <label className={labelClass}>絵文字の選択</label>
                    <input
                      type="text"
                      value={emojiConfig.text}
                      onChange={(e) => setEmojiConfig({ ...emojiConfig, text: e.target.value })}
                      className={`${inputClass} text-xl`}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className={labelClass}>倍率（{emojiConfig.scale}%）</label>
                      <input
                        type="range"
                        min="10"
                        max="200"
                        value={emojiConfig.scale}
                        onChange={(e) => setEmojiConfig({ ...emojiConfig, scale: Number(e.target.value) })}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                    <div>
                      <label className={labelClass}>回転角度（{emojiConfig.rotate}°）</label>
                      <input
                        type="range"
                        min="0"
                        max="360"
                        value={emojiConfig.rotate}
                        onChange={(e) => setEmojiConfig({ ...emojiConfig, rotate: Number(e.target.value) })}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                    <div>
                      <label className={labelClass}>上下位置調整（{emojiConfig.offsetY}）</label>
                      <input
                        type="range"
                        min="-50"
                        max="50"
                        value={emojiConfig.offsetY}
                        onChange={(e) => setEmojiConfig({ ...emojiConfig, offsetY: Number(e.target.value) })}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                    <div>
                      <label className={labelClass}>左右位置調整（{emojiConfig.offsetX}）</label>
                      <input
                        type="range"
                        min="-50"
                        max="50"
                        value={emojiConfig.offsetX}
                        onChange={(e) => setEmojiConfig({ ...emojiConfig, offsetX: Number(e.target.value) })}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label className={labelClass}>粗さ・強度（{intensity}）</label>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={intensity}
                      onChange={(e) => setIntensity(Number(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>
                  {selectedMosaic === 'fill' && (
                    <div>
                      <label className={labelClass}>塗りつぶし色</label>
                      <input
                        type="color"
                        value={fillColor}
                        onChange={(e) => setFillColor(e.target.value)}
                        className="h-9 w-full cursor-pointer rounded-lg border border-gray-200"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </details>

        <details className="group overflow-hidden rounded-2xl border border-gray-100">
          <summary className="flex cursor-pointer list-none items-center justify-between bg-gray-50 p-3 text-sm font-bold text-gray-800">
            <span className="flex items-center gap-2">
              <ListChecks className="h-4 w-4 text-emerald-500" />
              検出された文字列一覧（{detectedStrings.length}）
            </span>
            <ChevronDown className="h-4 w-4 text-gray-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="p-3.5">
            {detectedStrings.length > 0 ? (
              <ul className="space-y-2">
                {detectedStrings.map((str) => {
                  const style = STRING_TYPE_STYLE[str.type];
                  return (
                    <li
                      key={str.id}
                      className="flex items-center justify-between gap-2 rounded-xl border border-gray-100 px-3 py-2"
                    >
                      <span className="truncate text-sm text-gray-700">{str.text}</span>
                      <span
                        className={`flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-bold ${style.badge}`}
                      >
                        {style.icon}
                        {str.label}
                      </span>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="py-4 text-center text-sm text-gray-400">文字列は検出されませんでした</p>
            )}
          </div>
        </details>
        </div>
      </div>
    </div>
  );
};
