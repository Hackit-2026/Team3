// Hackit/backend (FastAPI) との通信をまとめたモジュール。
// App.tsx側のJSX/レイアウトには関与せず、fetch呼び出しのロジックだけを持つ。

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';

export interface WhitelistEntry {
  id: string;
  name: string;
  thumb_base64: string;
}

export async function fetchWhitelist(): Promise<WhitelistEntry[]> {
  const res = await fetch(`${API_BASE}/api/whitelist`);
  if (!res.ok) throw new Error('ホワイトリストの取得に失敗しました');
  const data = await res.json();
  return data.faces as WhitelistEntry[];
}

export async function saveWhitelistFace(
  id: string,
  name: string,
  imageBlobUrl: string | null,
): Promise<WhitelistEntry> {
  const form = new FormData();
  form.append('id', id);
  form.append('name', name);
  if (imageBlobUrl) {
    const blob = await (await fetch(imageBlobUrl)).blob();
    form.append('image', blob, 'face.png');
  }
  const res = await fetch(`${API_BASE}/api/whitelist/register`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('ホワイトリストの登録に失敗しました');
  return res.json();
}

export async function deleteWhitelistFace(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/whitelist/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('ホワイトリストの削除に失敗しました');
}

export function blobUrlFromBase64(base64: string, mime = 'image/png'): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

export interface DetectedFace {
  id: string;
  box: [number, number, number, number];
  polygon: [number, number][];
  type: string;
}

export interface DetectFacesResult {
  image_width: number;
  image_height: number;
  faces: DetectedFace[];
}

export type ScanMode = 'standard' | 'high';

export async function detectFaces(
  imageBlob: Blob,
  whitelistEnabled: boolean,
  scanMode: ScanMode,
  maskTargets: string[],
  enabledWhitelistIds: string[],
): Promise<DetectFacesResult> {
  const form = new FormData();
  form.append('file', imageBlob, 'photo.png');
  form.append('whitelist_enabled', String(whitelistEnabled));
  form.append('scan_mode', scanMode);
  form.append('mask_targets', maskTargets.join(','));
  form.append('whitelist_ids', enabledWhitelistIds.join(','));
  const res = await fetch(`${API_BASE}/api/detect-faces`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('顔検出に失敗しました');
  return res.json();
}

export interface DetectedText {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  label: 'public' | 'private';
}

export interface DetectTextResult {
  image_width: number;
  image_height: number;
  texts: DetectedText[];
}

export async function detectText(imageBlob: Blob): Promise<DetectTextResult> {
  const form = new FormData();
  form.append('file', imageBlob, 'photo.png');
  const res = await fetch(`${API_BASE}/api/detect-text`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('文字検出に失敗しました');
  return res.json();
}
