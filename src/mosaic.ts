// Canvas 2D でモザイクを描画する純粋関数群。
// Hackit/frontend/index.html の applyMosaic(black/blur/pixelate) と、
// safelist ブランチの apply_masking(顔モザイクの5スタイル) を
// ブラウザCanvas向けに移植したもの。UI(App.tsx)側の見た目には関与しない。

export type MosaicStyle = 'pixel' | 'blur' | 'emoji' | 'fill' | 'tile';
export type MosaicShape = 'rect' | 'circle' | 'contour';

export interface MosaicRegion {
  box: [number, number, number, number]; // [x, y, w, h] 画像ピクセル座標
  polygon?: [number, number][]; // 輪郭(あれば shape:'contour' で使用)
}

export interface EmojiConfig {
  text: string;
  scale: number; // %
  rotate: number; // deg
  offsetX: number;
  offsetY: number;
}

export interface MosaicStyleConfig {
  style: MosaicStyle;
  shape: MosaicShape;
  intensity: number; // 1-100（粗さ・強さ）
  fillColor: string;
  emoji: EmojiConfig;
}

function clipRegion(
  ctx: CanvasRenderingContext2D,
  box: [number, number, number, number],
  polygon: [number, number][] | undefined,
  shape: MosaicShape,
) {
  const [x, y, w, h] = box;
  ctx.beginPath();
  if (shape === 'contour' && polygon && polygon.length >= 3) {
    ctx.moveTo(polygon[0][0], polygon[0][1]);
    for (let i = 1; i < polygon.length; i++) ctx.lineTo(polygon[i][0], polygon[i][1]);
    ctx.closePath();
  } else if (shape === 'circle') {
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
  } else {
    ctx.rect(x, y, w, h);
  }
  ctx.clip();
}

function drawPixelate(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  box: [number, number, number, number],
  intensity: number,
) {
  const [x, y, w, h] = box;
  if (w <= 0 || h <= 0) return;
  const blockPx = 6 + (intensity / 100) * 54;
  const smallW = Math.max(1, Math.round(w / blockPx));
  const smallH = Math.max(1, Math.round(h / blockPx));

  const off = document.createElement('canvas');
  off.width = smallW;
  off.height = smallH;
  const offCtx = off.getContext('2d');
  if (!offCtx) return;
  offCtx.drawImage(source, x, y, w, h, 0, 0, smallW, smallH);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, smallW, smallH, x, y, w, h);
}

function drawBlur(ctx: CanvasRenderingContext2D, source: CanvasImageSource, intensity: number) {
  const blurRadius = 2 + (intensity / 100) * 48;
  ctx.filter = `blur(${blurRadius}px)`;
  ctx.drawImage(source, 0, 0);
  ctx.filter = 'none';
}

function drawFill(ctx: CanvasRenderingContext2D, box: [number, number, number, number], color: string) {
  const [x, y, w, h] = box;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
}

function drawTile(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  box: [number, number, number, number],
  intensity: number,
) {
  const [x, y, w, h] = box;
  if (w <= 0 || h <= 0) return;
  const tileSize = 8 + (intensity / 100) * 40;

  for (let gx = x; gx < x + w; gx += tileSize) {
    for (let gy = y; gy < y + h; gy += tileSize) {
      const tw = Math.min(tileSize, x + w - gx);
      const th = Math.min(tileSize, y + h - gy);
      if (tw <= 0 || th <= 0) continue;
      const off = document.createElement('canvas');
      off.width = 1;
      off.height = 1;
      const offCtx = off.getContext('2d');
      if (!offCtx) continue;
      offCtx.drawImage(source, gx, gy, tw, th, 0, 0, 1, 1);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, 1, 1, gx, gy, tw, th);
    }
  }
}

function drawEmoji(ctx: CanvasRenderingContext2D, box: [number, number, number, number], emoji: EmojiConfig) {
  const [x, y, w, h] = box;
  const cx = x + w / 2 + emoji.offsetX;
  const cy = y + h / 2 + emoji.offsetY;
  const fontSize = Math.max(w, h) * (emoji.scale / 100);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate((emoji.rotate * Math.PI) / 180);
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(emoji.text, 0, 0);
  ctx.restore();
}

export function applyMosaicToRegion(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  region: MosaicRegion,
  config: MosaicStyleConfig,
) {
  const { box, polygon } = region;
  const [, , w, h] = box;
  if (w <= 0 || h <= 0) return;

  if (config.style === 'emoji') {
    // 絵文字は枠の外にはみ出す前提の演出なのでクリップしない(Python版と同じ挙動)。
    drawEmoji(ctx, box, config.emoji);
    return;
  }

  ctx.save();
  clipRegion(ctx, box, polygon, config.shape);

  switch (config.style) {
    case 'pixel':
      drawPixelate(ctx, source, box, config.intensity);
      break;
    case 'blur':
      drawBlur(ctx, source, config.intensity);
      break;
    case 'fill':
      drawFill(ctx, box, config.fillColor);
      break;
    case 'tile':
      drawTile(ctx, source, box, config.intensity);
      break;
  }

  ctx.restore();
}
