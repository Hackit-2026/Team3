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
import {
  fetchWhitelist,
  saveWhitelistFace,
  deleteWhitelistFace,
  blobUrlFromBase64,
  detectFaces,
  detectText,
  type ScanMode,
  type DetectedFace,
} from './api';
import { applyMosaicToRegion, type MosaicShape, type MosaicStyleConfig } from './mosaic';

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
  // モザイク描画用の座標(コア処理から取得。一覧表示では使わない)。
  box: [number, number, number, number];
  // ユーザーが一覧のチェックボックスで手動ON/OFFできるモザイク適用状態。
  masked: boolean;
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
  const [whitelistFaces, setWhitelistFaces] = useState<FaceData[]>([]);

  // 起動時にバックエンドの登録済みホワイトリストを読み込む。
  useEffect(() => {
    fetchWhitelist()
      .then((entries) => {
        setWhitelistFaces(
          entries.map((entry) => ({
            id: Number(entry.id),
            name: entry.name,
            imgUrl: blobUrlFromBase64(entry.thumb_base64),
          })),
        );
      })
      .catch((err) => console.error('ホワイトリストの読み込みに失敗しました', err));
  }, []);

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
            whitelistEnabled={whitelistEnabled}
            whitelistFaces={whitelistFaces}
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

// 顔のマスク対象パーツ(バックエンドのface_engine.pyと同じ文言をそのまま使う)。
const MASK_PART_OPTIONS = ['顔全体', '目元（両目）', '右目 (解剖学的)', '左目 (解剖学的)', '鼻', '口元'];

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

  const handleSaveFace = async () => {
    if (!name) {
      setNameError(true);
      return;
    }
    const id = editingId ?? Date.now();
    try {
      await saveWhitelistFace(String(id), name, previewImg);
    } catch (err) {
      console.error('ホワイトリストの保存に失敗しました', err);
      window.alert('ホワイトリストの保存に失敗しました。バックエンドの接続を確認してください。');
      return;
    }
    if (editingId !== null) {
      setFaces(
        faces.map((f) => (f.id === editingId ? { ...f, name, imgUrl: previewImg || f.imgUrl } : f)),
      );
    } else {
      const newFace: FaceData = { id, name, imgUrl: previewImg || '👤' };
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

  const handleDeleteFace = async (id: number) => {
    if (!window.confirm('この顔をホワイトリストから削除しますか？')) return;
    try {
      await deleteWhitelistFace(String(id));
    } catch (err) {
      console.error('ホワイトリストの削除に失敗しました', err);
      window.alert('ホワイトリストの削除に失敗しました。バックエンドの接続を確認してください。');
      return;
    }
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
  whitelistEnabled: boolean;
  whitelistFaces: FaceData[];
}

const STRING_TYPE_STYLE: Record<DetectedString['type'], { badge: string; icon: React.ReactNode }> = {
  phone: { badge: 'bg-red-50 text-red-600', icon: <Phone className="h-3 w-3" /> },
  address: { badge: 'bg-amber-50 text-amber-700', icon: <MapPin className="h-3 w-3" /> },
  other: { badge: 'bg-gray-100 text-gray-600', icon: <FileText className="h-3 w-3" /> },
  ignore: { badge: 'bg-gray-50 text-gray-400', icon: <EyeOff className="h-3 w-3" /> },
};

interface EmojiConfigState {
  text: string;
  scale: number;
  rotate: number;
  offsetX: number;
  offsetY: number;
}

interface MosaicStyleFieldsProps {
  mosaic: MosaicId;
  intensity: number;
  setIntensity: (v: number) => void;
  fillColor: string;
  setFillColor: (v: string) => void;
  emoji: EmojiConfigState;
  setEmoji: (v: EmojiConfigState) => void;
}

// 顔用・文字列用で共通の「スタイル詳細設定」欄（粗さ・色・絵文字など）。
const MosaicStyleFields: React.FC<MosaicStyleFieldsProps> = ({
  mosaic,
  intensity,
  setIntensity,
  fillColor,
  setFillColor,
  emoji,
  setEmoji,
}) => (
  <div className="rounded-xl border border-gray-100 bg-gray-50 p-3.5">
    <p className="mb-3 border-b border-gray-200 pb-2 text-xs font-bold text-emerald-700">
      スタイル詳細設定（{MOSAIC_TYPES.find((m) => m.id === mosaic)?.label}）
    </p>

    {mosaic === 'emoji' ? (
      <div className="space-y-3">
        <div>
          <label className={labelClass}>絵文字の選択</label>
          <input
            type="text"
            value={emoji.text}
            onChange={(e) => setEmoji({ ...emoji, text: e.target.value })}
            className={`${inputClass} text-xl`}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelClass}>倍率（{emoji.scale}%）</label>
            <input
              type="range"
              min="10"
              max="200"
              value={emoji.scale}
              onChange={(e) => setEmoji({ ...emoji, scale: Number(e.target.value) })}
              className="w-full accent-emerald-500"
            />
          </div>
          <div>
            <label className={labelClass}>回転角度（{emoji.rotate}°）</label>
            <input
              type="range"
              min="0"
              max="360"
              value={emoji.rotate}
              onChange={(e) => setEmoji({ ...emoji, rotate: Number(e.target.value) })}
              className="w-full accent-emerald-500"
            />
          </div>
          <div>
            <label className={labelClass}>上下位置調整（{emoji.offsetY}）</label>
            <input
              type="range"
              min="-50"
              max="50"
              value={emoji.offsetY}
              onChange={(e) => setEmoji({ ...emoji, offsetY: Number(e.target.value) })}
              className="w-full accent-emerald-500"
            />
          </div>
          <div>
            <label className={labelClass}>左右位置調整（{emoji.offsetX}）</label>
            <input
              type="range"
              min="-50"
              max="50"
              value={emoji.offsetX}
              onChange={(e) => setEmoji({ ...emoji, offsetX: Number(e.target.value) })}
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
        {mosaic === 'fill' && (
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
);

const ProcessingScreen: React.FC<ProcessingScreenProps> = ({
  navigate,
  selectedMosaic,
  uploadedImage,
  whitelistEnabled,
  whitelistFaces,
}) => {
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processedImage, setProcessedImage] = useState<string | null>(uploadedImage);
  const [detectedStrings, setDetectedStrings] = useState<DetectedString[]>([]);

  // この処理画面だけで有効にするホワイトリスト対象者(既定は全員ON)。
  const [enabledWhitelistIds, setEnabledWhitelistIds] = useState<Set<number>>(
    () => new Set(whitelistFaces.map((f) => f.id)),
  );
  const toggleWhitelistFace = (id: number) => {
    setEnabledWhitelistIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const setAllWhitelistFaces = (enabled: boolean) => {
    setEnabledWhitelistIds(enabled ? new Set(whitelistFaces.map((f) => f.id)) : new Set());
  };

  // モザイク設定のState群（これらをコア処理に渡す）
  const [detectMode, setDetectMode] = useState<string>('顔 + 文字列');
  const [maskShape, setMaskShape] = useState<string>('長方形');
  const [faceAccuracy, setFaceAccuracy] = useState<string>('標準');
  const [maskTargetParts, setMaskTargetParts] = useState<string[]>(['顔全体']);
  const toggleMaskPart = (part: string) => {
    setMaskTargetParts((prev) => (prev.includes(part) ? prev.filter((p) => p !== part) : [...prev, part]));
  };

  // 顔用・文字列用でモザイクの種類とスタイル詳細をそれぞれ独立に持つ。
  const [faceMosaic, setFaceMosaic] = useState<MosaicId>(selectedMosaic);
  const [faceIntensity, setFaceIntensity] = useState<number>(50);
  const [faceFillColor, setFaceFillColor] = useState<string>('#16a34a');
  const [faceEmoji, setFaceEmoji] = useState<EmojiConfigState>({
    text: '😎',
    scale: 100,
    rotate: 0,
    offsetX: 0,
    offsetY: 0,
  });

  const [textMosaic, setTextMosaic] = useState<MosaicId>(selectedMosaic);
  const [textIntensity, setTextIntensity] = useState<number>(50);
  const [textFillColor, setTextFillColor] = useState<string>('#16a34a');
  const [textEmoji, setTextEmoji] = useState<EmojiConfigState>({
    text: '😎',
    scale: 100,
    rotate: 0,
    offsetX: 0,
    offsetY: 0,
  });

  // マスクの形状ドロップダウンの表示文言 → mosaic.ts のshape値への対応。
  const MASK_SHAPE_MAP: Record<string, MosaicShape> = {
    長方形: 'rect',
    円形: 'circle',
    顔の輪郭に合わせる: 'contour',
  };

  // 顔検出の精度ドロップダウンの表示文言 → バックエンドのscan_mode値への対応。
  const FACE_ACCURACY_MAP: Record<string, ScanMode> = {
    標準: 'standard',
    高精度: 'high',
  };

  // 検出済みの顔一覧(再描画のために保持。顔は一覧のON/OFFトグル対象外)。
  const [faceRegions, setFaceRegions] = useState<DetectedFace[]>([]);

  // 元画像＋現在の検出結果＋現在のスタイル設定から、Canvasに描き直す。
  // 文字列一覧のチェックボックスでON/OFFを切り替えるたびに呼び直す。
  const redrawCanvas = async (faces: DetectedFace[], strings: DetectedString[]) => {
    if (!uploadedImage) return;

    const img = new Image();
    const imgLoaded = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('画像の読み込みに失敗しました'));
    });
    img.src = uploadedImage;
    await imgLoaded;

    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvasの初期化に失敗しました');
    ctx.drawImage(img, 0, 0);

    const faceStyleConfig: MosaicStyleConfig = {
      style: faceMosaic,
      shape: MASK_SHAPE_MAP[maskShape] ?? 'rect',
      intensity: faceIntensity,
      fillColor: faceFillColor,
      emoji: faceEmoji,
    };
    for (const face of faces) {
      applyMosaicToRegion(ctx, img, { box: face.box, polygon: face.polygon }, faceStyleConfig);
    }

    // 文字列は形状ドロップダウン(顔用)に関わらず常に長方形でマスクする
    // （円形/輪郭クリップだと隅の文字が隠れずに残ってしまうため）。
    const textStyleConfig: MosaicStyleConfig = {
      style: textMosaic,
      shape: 'rect',
      intensity: textIntensity,
      fillColor: textFillColor,
      emoji: textEmoji,
    };
    for (const str of strings) {
      if (str.masked) {
        applyMosaicToRegion(ctx, img, { box: str.box }, textStyleConfig);
      }
    }

    setProcessedImage(canvas.toDataURL('image/png'));
  };

  // コア処理実行フック
  const handleApplyProcess = async () => {
    if (!uploadedImage) return;
    setIsProcessing(true);

    try {
      const sourceBlob = await (await fetch(uploadedImage)).blob();

      const includeFaces = detectMode !== '文字列のみ';
      const includeText = detectMode !== '顔のみ';

      const [faceResult, textResult] = await Promise.all([
        includeFaces
          ? detectFaces(
              sourceBlob,
              whitelistEnabled,
              FACE_ACCURACY_MAP[faceAccuracy] ?? 'standard',
              maskTargetParts,
              [...enabledWhitelistIds].map(String),
            )
          : Promise.resolve(null),
        includeText ? detectText(sourceBlob) : Promise.resolve(null),
      ]);

      const faces = faceResult ? faceResult.faces : [];
      // 検出直後の初期状態は、LLM判定のprivate/publicをそのままON/OFFの初期値にする。
      const strings: DetectedString[] = textResult
        ? textResult.texts.map((t, i) => {
            const isPrivate = t.label === 'private';
            return {
              id: i + 1,
              text: t.text,
              type: isPrivate ? 'other' : 'ignore',
              label: isPrivate ? '非公開（マスク対象）' : '公開情報（マスクなし）',
              box: [t.x, t.y, t.w, t.h] as [number, number, number, number],
              masked: isPrivate,
            };
          })
        : [];

      setFaceRegions(faces);
      setDetectedStrings(strings);
      await redrawCanvas(faces, strings);
    } catch (error) {
      console.error('処理エラー', error);
      window.alert('モザイク処理に失敗しました。バックエンドの接続を確認してください。');
    } finally {
      setIsProcessing(false);
    }
  };

  // 検出された文字列一覧のチェックボックスでモザイクON/OFFを切り替える。
  const handleToggleStringMask = (id: number) => {
    const updated = detectedStrings.map((s) => (s.id === id ? { ...s, masked: !s.masked } : s));
    setDetectedStrings(updated);
    redrawCanvas(faceRegions, updated).catch((error) => {
      console.error('再描画エラー', error);
      window.alert('モザイクの再描画に失敗しました。');
    });
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
            <div>
              <label className={labelClass}>検出モード</label>
              <select value={detectMode} onChange={(e) => setDetectMode(e.target.value)} className={inputClass}>
                <option>顔 + 文字列</option>
                <option>顔のみ</option>
                <option>文字列のみ</option>
              </select>
            </div>

            <div>
              <label className={labelClass}>顔検出の精度</label>
              <select
                value={faceAccuracy}
                onChange={(e) => setFaceAccuracy(e.target.value)}
                className={inputClass}
              >
                <option>標準</option>
                <option>高精度</option>
              </select>
            </div>

            <div className="border-t border-gray-100 pt-4">
              <p className="mb-3 text-xs font-bold text-gray-500">顔のモザイク</p>
              <div className="space-y-3">
                <div>
                  <label className={labelClass}>モザイクの種類</label>
                  <select
                    value={faceMosaic}
                    onChange={(e) => setFaceMosaic(e.target.value as MosaicId)}
                    className={inputClass}
                  >
                    {MOSAIC_TYPES.map((type) => (
                      <option key={type.id} value={type.id}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className={labelClass}>マスク対象パーツ</label>
                  <div className="flex flex-wrap gap-x-3 gap-y-1.5">
                    {MASK_PART_OPTIONS.map((part) => (
                      <label key={part} className="flex items-center gap-1.5 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          checked={maskTargetParts.includes(part)}
                          onChange={() => toggleMaskPart(part)}
                          className="h-4 w-4 accent-emerald-500"
                        />
                        {part}
                      </label>
                    ))}
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

                <MosaicStyleFields
                  mosaic={faceMosaic}
                  intensity={faceIntensity}
                  setIntensity={setFaceIntensity}
                  fillColor={faceFillColor}
                  setFillColor={setFaceFillColor}
                  emoji={faceEmoji}
                  setEmoji={setFaceEmoji}
                />
              </div>
            </div>

            <div className="border-t border-gray-100 pt-4">
              <p className="mb-3 text-xs font-bold text-gray-500">文字列のモザイク</p>
              <div className="space-y-3">
                <div>
                  <label className={labelClass}>モザイクの種類</label>
                  <select
                    value={textMosaic}
                    onChange={(e) => setTextMosaic(e.target.value as MosaicId)}
                    className={inputClass}
                  >
                    {MOSAIC_TYPES.map((type) => (
                      <option key={type.id} value={type.id}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>

                <MosaicStyleFields
                  mosaic={textMosaic}
                  intensity={textIntensity}
                  setIntensity={setTextIntensity}
                  fillColor={textFillColor}
                  setFillColor={setTextFillColor}
                  emoji={textEmoji}
                  setEmoji={setTextEmoji}
                />
              </div>
            </div>
          </div>
        </details>

        <details className="group mb-3 overflow-hidden rounded-2xl border border-gray-100">
          <summary className="flex cursor-pointer list-none items-center justify-between bg-gray-50 p-3 text-sm font-bold text-gray-800">
            <span className="flex items-center gap-2">
              <Users className="h-4 w-4 text-emerald-500" />
              ホワイトリスト対象者（{enabledWhitelistIds.size}/{whitelistFaces.length}）
            </span>
            <ChevronDown className="h-4 w-4 text-gray-400 transition-transform group-open:rotate-180" />
          </summary>
          <div className="p-3.5">
            {whitelistFaces.length > 0 ? (
              <>
                <div className="mb-3 flex gap-2">
                  <button
                    type="button"
                    onClick={() => setAllWhitelistFaces(true)}
                    className={`${secondaryBtn} flex-1 py-2 text-xs`}
                  >
                    全員ON
                  </button>
                  <button
                    type="button"
                    onClick={() => setAllWhitelistFaces(false)}
                    className={`${secondaryBtn} flex-1 py-2 text-xs`}
                  >
                    全員OFF
                  </button>
                </div>
                <ul className="space-y-2">
                  {whitelistFaces.map((face) => (
                    <li key={face.id} className="flex items-center rounded-xl border border-gray-100 px-3 py-2">
                      <label className="flex w-full cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={enabledWhitelistIds.has(face.id)}
                          onChange={() => toggleWhitelistFace(face.id)}
                          className="h-4 w-4 accent-emerald-500"
                        />
                        <span className="truncate text-sm text-gray-700">{face.name}</span>
                      </label>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="py-4 text-center text-sm text-gray-400">登録済みのホワイトリストはありません</p>
            )}
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
                      <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={str.masked}
                          onChange={() => handleToggleStringMask(str.id)}
                          className="h-4 w-4 shrink-0 accent-emerald-500"
                          aria-label={`${str.text}のモザイクを切り替え`}
                        />
                        <span className="truncate text-sm text-gray-700">{str.text}</span>
                      </label>
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
