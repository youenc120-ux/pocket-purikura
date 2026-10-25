// 在 Node 裡載入編好的 wasm 做基本檢查（這台電腦的安全原則不允許執行 cargo test 產生的 .exe）。
// 用法：node px/test.mjs
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const wasm = readFileSync(new URL("../px.wasm", import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports;

function call(bufs, fn) {
  const ps = bufs.map(([ta]) => {
    const n = ta.byteLength, p = e.px_alloc(n);
    new Uint8Array(e.memory.buffer, p, n).set(new Uint8Array(ta.buffer, ta.byteOffset, n));
    return [p, n];
  });
  try { return fn(...ps.map(x => x[0])); }
  finally {
    bufs.forEach(([ta, mut], k) => {
      const [p, n] = ps[k];
      if (mut) new Uint8Array(ta.buffer, ta.byteOffset, n).set(new Uint8Array(e.memory.buffer, p, n));
      e.px_free(p, n);
    });
  }
}
const ramp = (w, h) => Uint8ClampedArray.from({ length: w * h * 4 }, (_, i) => (i % 4 === 3 ? 255 : (i * 7) % 256));
const results = [];
const test = (name, fn) => { try { fn(); results.push(`ok   ${name}`); } catch (err) { results.push(`FAIL ${name}: ${err.message}`); process.exitCode = 1; } };

test("強度 0 的大眼不改變影像", () => {
  const d = ramp(16, 16), before = d.slice();
  call([[d, 1]], p => e.bulge(p, 16, 16, 8, 8, 5, 0));
  assert.deepEqual(d, before);
});
test("大眼會改變眼睛範圍內、不動範圍外", () => {
  const d = ramp(32, 32), before = d.slice();
  call([[d, 1]], p => e.bulge(p, 32, 32, 16, 16, 6, .2));
  assert.notDeepEqual(d, before);
  assert.equal(d[(2 * 32 + 2) * 4], before[(2 * 32 + 2) * 4]);
});
test("均勻影像銳利化後不變", () => {
  const d = new Uint8ClampedArray(12 * 12 * 4).fill(100);
  call([[d, 1], [new Float32Array(0), 0]], (p, m) => e.sharpen(p, 12, 12, m, 0, .6));
  assert.ok(d.every((v, i) => i % 4 === 3 || v === 100));
});
test("濾鏡代碼 0 不改變影像", () => {
  const d = ramp(8, 8), before = d.slice();
  call([[d, 1]], p => e.tone(p, d.length, 0, 1));
  assert.deepEqual(d, before);
});
test("黑白濾鏡讓 RGB 三色相等（顆粒前）", () => {
  const d = ramp(8, 8);
  call([[d, 1]], p => e.tone(p, d.length, 4, 1));
  for (let i = 0; i < d.length; i += 4) assert.ok(Math.abs(d[i] - d[i + 1]) <= 1 && Math.abs(d[i] - d[i + 2]) <= 1);
});
test("磨皮保留硬邊（左黑右白）", () => {
  const w = 40, h = 20, d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { const v = i % w < w / 2 ? 0 : 255; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255; }
  const mask = new Float32Array(w * h).fill(1);
  call([[d, 1], [mask, 0]], (p, m) => e.smooth_skin(p, w, h, m, mask.length, 0, 0, w, h, 4, .002, .9, 0));
  assert.ok(d[(10 * w + 5) * 4] < 15 && d[(10 * w + 34) * 4] > 240);
});
test("瘦臉在下巴以下淡出", () => {
  const w = 64, h = 64, d = ramp(w, h), before = d.slice();
  const pts = new Float32Array([20, 30, 3, 44, 30, -3]);
  call([[d, 1], [pts, 0]], (p, q) => e.slim(p, w, h, q, pts.length, 34, 14, 4));
  const row = (y) => d.slice(y * w * 4, (y + 1) * w * 4);
  assert.notDeepEqual(row(30), before.slice(30 * w * 4, 31 * w * 4));
  assert.deepEqual(row(45), before.slice(45 * w * 4, 46 * w * 4));
});
test("遮罩放大後保持 0–1", () => {
  const small = Float32Array.from({ length: 16 }, (_, i) => (i % 4) / 3), out = new Float32Array(16 * 16);
  call([[small, 0], [out, 1]], (a, b) => e.upsample_mask(a, 4, 4, b, 16, 16, 4));
  assert.ok(out.every(v => v >= 0 && v <= 1));
});

console.log(results.join("\n"));
