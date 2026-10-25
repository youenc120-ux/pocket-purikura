//! 口袋拍貼機的影像運算核心，編成 WebAssembly 給網頁呼叫。
//!
//! 不使用 wasm-bindgen（它需要在本機執行 build script），改成零依賴、直接匯出 C 介面，
//! 網頁那邊的銜接程式是手寫的（見 booth.html 的 loadPx）。
//!
//! 所有影像都是 RGBA、每個像素 4 個 byte、逐列排列（跟 canvas 的 ImageData 一樣）。
//! 函式直接改寫傳進來的陣列，網頁那邊不用再複製一次。

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 255.0) + 0.5) as u8
}

/// 方框模糊：先橫向再縱向的累加和，跟半徑無關，O(n)。
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    let mut out = vec![0f32; w * h];
    let k = 1.0 / (2 * r + 1) as f32;
    let (ri, wi, hi) = (r as isize, w as isize, h as isize);
    for y in 0..h {
        let o = y * w;
        let mut s = 0f32;
        for x in -ri..=ri {
            s += src[o + x.clamp(0, wi - 1) as usize];
        }
        for x in 0..w {
            tmp[o + x] = s * k;
            let add = (x as isize + ri + 1).min(wi - 1) as usize;
            let sub = (x as isize - ri).max(0) as usize;
            s += src[o + add] - src[o + sub];
        }
    }
    for x in 0..w {
        let mut s = 0f32;
        for y in -ri..=ri {
            s += tmp[y.clamp(0, hi - 1) as usize * w + x];
        }
        for y in 0..h {
            out[y * w + x] = s * k;
            let add = (y as isize + ri + 1).min(hi - 1) as usize;
            let sub = (y as isize - ri).max(0) as usize;
            s += tmp[add * w + x] - tmp[sub * w + x];
        }
    }
    out
}

/// 導向濾波（He et al.）：用 guide 的邊緣來平滑 p。
fn guided(guide: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let ip: Vec<f32> = (0..n).map(|i| guide[i] * p[i]).collect();
    let ii: Vec<f32> = guide.iter().map(|v| v * v).collect();
    let m_i = box_blur(guide, w, h, r);
    let m_p = box_blur(p, w, h, r);
    let m_ip = box_blur(&ip, w, h, r);
    let m_ii = box_blur(&ii, w, h, r);
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for i in 0..n {
        let var = m_ii[i] - m_i[i] * m_i[i];
        a[i] = (m_ip[i] - m_i[i] * m_p[i]) / (var + eps);
        b[i] = m_p[i] - a[i] * m_i[i];
    }
    let m_a = box_blur(&a, w, h, r);
    let m_b = box_blur(&b, w, h, r);
    (0..n).map(|i| m_a[i] * guide[i] + m_b[i]).collect()
}

/// 自己當導引的導向濾波：平滑但保留邊緣，少算兩次模糊。
fn guided_self(p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let pp: Vec<f32> = p.iter().map(|v| v * v).collect();
    let m_p = box_blur(p, w, h, r);
    let m_pp = box_blur(&pp, w, h, r);
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for i in 0..n {
        let var = m_pp[i] - m_p[i] * m_p[i];
        a[i] = var / (var + eps);
        b[i] = (1.0 - a[i]) * m_p[i];
    }
    let m_a = box_blur(&a, w, h, r);
    let m_b = box_blur(&b, w, h, r);
    (0..n).map(|i| m_a[i] * p[i] + m_b[i]).collect()
}

#[inline]
fn bilinear(src: &[u8], w: usize, h: usize, x: f32, y: f32, out: &mut [u8], o: usize) {
    let x = x.clamp(0.0, w as f32 - 1.001);
    let y = y.clamp(0.0, h as f32 - 1.001);
    let (x0, y0) = (x as usize, y as usize);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let i00 = (y0 * w + x0) * 4;
    let (i10, i01) = (i00 + 4, i00 + w * 4);
    let i11 = i01 + 4;
    for c in 0..3 {
        let top = src[i00 + c] as f32 * (1.0 - fx) + src[i10 + c] as f32 * fx;
        let bot = src[i01 + c] as f32 * (1.0 - fx) + src[i11 + c] as f32 * fx;
        out[o + c] = to_u8(top * (1.0 - fy) + bot * fy);
    }
}

/// 大眼：以 (cx, cy) 為中心、半徑 r 局部放大，k 是強度，邊緣平滑衰減。
pub fn bulge(data: &mut [u8], w: u32, h: u32, cx: f32, cy: f32, r: f32, k: f32) {
    let (w, h) = (w as usize, h as usize);
    let src = data.to_vec();
    let r2 = r * r;
    let x0 = (cx - r).floor().max(0.0) as usize;
    let x1 = ((cx + r).ceil() as usize).min(w - 1);
    let y0 = (cy - r).floor().max(0.0) as usize;
    let y1 = ((cy + r).ceil() as usize).min(h - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let d2 = dx * dx + dy * dy;
            if d2 >= r2 {
                continue;
            }
            let t = 1.0 - d2 / r2;
            let s = 1.0 - k * t * t;
            bilinear(&src, w, h, cx + dx * s, cy + dy * s, data, (y * w + x) * 4);
        }
    }
}

/// 瘦臉：pts 是 (x, y, 水平位移) 一組三個的下顎控制點。
/// 整張只算一次平滑位移，下巴（chin_y）以下 fade 像素內淡出，不動到脖子和衣領。
pub fn slim(data: &mut [u8], w: u32, h: u32, pts: &[f32], chin_y: f32, r: f32, fade: f32) {
    let (w, h) = (w as usize, h as usize);
    if pts.len() < 3 {
        return;
    }
    let src = data.to_vec();
    let r2 = r * r;
    let pts: Vec<(f32, f32, f32)> = pts.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect();
    let min_x = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let max_x = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let min_y = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let x0 = (min_x - r).floor().max(0.0) as usize;
    let x1 = ((max_x + r).ceil().max(0.0) as usize).min(w - 1);
    let y0 = (min_y - r).floor().max(0.0) as usize;
    let y1 = ((chin_y + fade).ceil().max(0.0) as usize).min(h - 1);
    for y in y0..=y1 {
        let yf = y as f32;
        let below = if yf > chin_y { (1.0 - (yf - chin_y) / fade).max(0.0) } else { 1.0 };
        if below <= 0.0 {
            continue;
        }
        for x in x0..=x1 {
            let xf = x as f32;
            let (mut sw, mut sd) = (0f32, 0f32);
            for &(px, py, s) in &pts {
                let d2 = (xf - px) * (xf - px) + (yf - py) * (yf - py);
                if d2 < r2 {
                    let t = 1.0 - d2 / r2;
                    let wt = t * t;
                    sw += wt;
                    sd += wt * s;
                }
            }
            if sw == 0.0 {
                continue;
            }
            let dx = sd / sw * sw.min(1.0) * below;
            if dx.abs() < 0.05 {
                continue;
            }
            bilinear(&src, w, h, xf - dx, yf, data, (y * w + x) * 4);
        }
    }
}

/// 美肌：在 (x0, y0)–(x1, y1) 範圍內用導向濾波磨皮，mask 是整張的皮膚遮罩（0–1）。
/// amt 是磨皮比例，lift 是皮膚提亮量。
#[allow(clippy::too_many_arguments)]
pub fn smooth_skin(
    data: &mut [u8], w: u32, h: u32, mask: &[f32],
    x0: u32, y0: u32, x1: u32, y1: u32,
    r: u32, eps: f32, amt: f32, lift: f32,
) {
    let (w, x0, y0) = (w as usize, x0 as usize, y0 as usize);
    let (x1, y1) = ((x1 as usize).min(w), (y1 as usize).min(h as usize));
    if x1 <= x0 + 8 || y1 <= y0 + 8 || mask.len() < w * h as usize {
        return;
    }
    let (rw, rh) = (x1 - x0, y1 - y0);
    for c in 0..3 {
        let mut p = vec![0f32; rw * rh];
        for y in 0..rh {
            for x in 0..rw {
                p[y * rw + x] = data[((y + y0) * w + x + x0) * 4 + c] as f32 / 255.0;
            }
        }
        let q = guided_self(&p, rw, rh, r as usize, eps);
        for y in 0..rh {
            for x in 0..rw {
                let gi = (y + y0) * w + x + x0;
                let m = mask[gi];
                if m < 0.01 {
                    continue;
                }
                let mut v = p[y * rw + x];
                v += (q[y * rw + x] - v) * amt * m;
                v += lift * m * v * (1.0 - v) * 2.0;
                data[gi * 4 + c] = to_u8(v * 255.0);
            }
        }
    }
}

/// 銳利化（unsharp mask）：mask 為空時整張一樣強，否則皮膚的地方減弱。
pub fn sharpen(data: &mut [u8], w: u32, h: u32, mask: &[f32], amount: f32) {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    let has_mask = mask.len() >= n;
    for c in 0..3 {
        let p: Vec<f32> = (0..n).map(|i| data[i * 4 + c] as f32).collect();
        let b = box_blur(&p, w, h, 1);
        for i in 0..n {
            let k = if has_mask { amount * (1.0 - mask[i] * 0.85) } else { amount };
            data[i * 4 + c] = to_u8(p[i] + (p[i] - b[i]) * k);
        }
    }
}

/// 去背修邊：用照片亮度當導引，修整遮罩（RGBA 的 alpha）邊緣，頭髮會比較自然。
pub fn refine_alpha(img: &[u8], mask: &mut [u8], w: u32, h: u32, r: u32, eps: f32) {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    let guide: Vec<f32> = (0..n)
        .map(|i| (0.299 * img[i * 4] as f32 + 0.587 * img[i * 4 + 1] as f32 + 0.114 * img[i * 4 + 2] as f32) / 255.0)
        .collect();
    let p: Vec<f32> = (0..n).map(|i| mask[i * 4 + 3] as f32 / 255.0).collect();
    let q = guided(&guide, &p, w, h, r as usize, eps);
    for i in 0..n {
        mask[i * 4 + 3] = to_u8(q[i].clamp(0.0, 1.0) * 255.0);
    }
}

/// 把縮小算好的皮膚遮罩用雙線性放大回原尺寸。
pub fn upsample_mask(small: &[f32], sw: u32, sh: u32, w: u32, h: u32, scale: f32) -> Vec<f32> {
    let (sw, sh, w, h) = (sw as usize, sh as usize, w as usize, h as usize);
    let mut out = vec![0f32; w * h];
    if sw < 2 || sh < 2 {
        return out;
    }
    for y in 0..h {
        let gy = (y as f32 / scale).min(sh as f32 - 1.001);
        let (y0, fy) = (gy as usize, gy - (gy as usize) as f32);
        for x in 0..w {
            let gx = (x as f32 / scale).min(sw as f32 - 1.001);
            let (x0, fx) = (gx as usize, gx - (gx as usize) as f32);
            let i = y0 * sw + x0;
            out[y * w + x] = (small[i] * (1.0 - fx) + small[i + 1] * fx) * (1.0 - fy)
                + (small[i + sw] * (1.0 - fx) + small[i + sw + 1] * fx) * fy;
        }
    }
    out
}

/// 色調。filter：1 牛奶肌、2 粉嫩、3 復古、4 黑白、5 冷光；其他值不動。
/// 復古和黑白會加上底片顆粒，seed 讓同一張照片每次顆粒都一樣。
pub fn tone(data: &mut [u8], filter: u8, seed: u32) {
    if !(1..=5).contains(&filter) {
        return;
    }
    let grain = filter == 3 || filter == 4;
    let mut rng = seed.wrapping_mul(2_654_435_761) | 1;
    let lift = |v: f32| 0.025 + 0.975 * (v + 0.1 * v * (1.0 - v));
    for px in data.chunks_exact_mut(4) {
        let (mut r, mut g, mut b) = (px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0);
        let l = 0.299 * r + 0.587 * g + 0.114 * b;
        match filter {
            1 | 2 => {
                r = lift(r);
                g = lift(g);
                b = lift(b);
                let ll = 0.299 * r + 0.587 * g + 0.114 * b;
                r = ll + (r - ll) * 0.93;
                g = ll + (g - ll) * 0.93;
                b = ll + (b - ll) * 0.93;
                if filter == 2 {
                    r += 0.035 * l;
                    b += 0.025 * l;
                    g -= 0.01 * l;
                } else {
                    r += 0.01;
                    b -= 0.005;
                }
            }
            3 => {
                let sr = 0.393 * r + 0.769 * g + 0.189 * b;
                let sg = 0.349 * r + 0.686 * g + 0.168 * b;
                let sb = 0.272 * r + 0.534 * g + 0.131 * b;
                r = 0.07 + (r * 0.7 + sr * 0.3) * 0.86;
                g = 0.06 + (g * 0.7 + sg * 0.3) * 0.86;
                b = 0.05 + (b * 0.7 + sb * 0.3) * 0.82;
            }
            4 => {
                let s = if l < 0.5 { 2.0 * l * l } else { 1.0 - 2.0 * (1.0 - l) * (1.0 - l) };
                let v = 0.035 + 0.955 * (l * 0.45 + s * 0.55);
                r = v;
                g = v;
                b = v;
            }
            _ => {
                r *= 0.97;
                g += 0.01;
                b = b * 1.03 + 0.02;
            }
        }
        if grain {
            // xorshift32
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let n = (rng as f32 / u32::MAX as f32 - 0.5) * 0.045;
            r += n;
            g += n;
            b += n;
        }
        px[0] = to_u8(r * 255.0);
        px[1] = to_u8(g * 255.0);
        px[2] = to_u8(b * 255.0);
    }
}

/// 給網頁呼叫的 C 介面。網頁先用 px_alloc 在 WebAssembly 記憶體裡開空間、把資料複製進來，
/// 呼叫完再複製回去並 px_free。長度單位：u8 陣列是 byte，f32 陣列是元素個數。
#[cfg(target_arch = "wasm32")]
mod ffi {
    use std::alloc::{alloc, dealloc, Layout};
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len.max(1), 8).unwrap()
    }

    #[no_mangle]
    pub extern "C" fn px_alloc(len: usize) -> *mut u8 {
        unsafe { alloc(layout(len)) }
    }

    #[no_mangle]
    pub extern "C" fn px_free(ptr: *mut u8, len: usize) {
        unsafe { dealloc(ptr, layout(len)) }
    }

    unsafe fn rgba<'a>(p: *mut u8, w: u32, h: u32) -> &'a mut [u8] {
        from_raw_parts_mut(p, w as usize * h as usize * 4)
    }

    #[no_mangle]
    pub unsafe extern "C" fn bulge(d: *mut u8, w: u32, h: u32, cx: f32, cy: f32, r: f32, k: f32) {
        super::bulge(rgba(d, w, h), w, h, cx, cy, r, k)
    }

    #[no_mangle]
    pub unsafe extern "C" fn slim(d: *mut u8, w: u32, h: u32, pts: *const f32, n: usize, chin_y: f32, r: f32, fade: f32) {
        super::slim(rgba(d, w, h), w, h, from_raw_parts(pts, n), chin_y, r, fade)
    }

    #[no_mangle]
    pub unsafe extern "C" fn smooth_skin(
        d: *mut u8, w: u32, h: u32, mask: *const f32, n: usize,
        x0: u32, y0: u32, x1: u32, y1: u32, r: u32, eps: f32, amt: f32, lift: f32,
    ) {
        super::smooth_skin(rgba(d, w, h), w, h, from_raw_parts(mask, n), x0, y0, x1, y1, r, eps, amt, lift)
    }

    #[no_mangle]
    pub unsafe extern "C" fn sharpen(d: *mut u8, w: u32, h: u32, mask: *const f32, n: usize, amount: f32) {
        super::sharpen(rgba(d, w, h), w, h, from_raw_parts(mask, n), amount)
    }

    #[no_mangle]
    pub unsafe extern "C" fn refine_alpha(img: *const u8, mask: *mut u8, w: u32, h: u32, r: u32, eps: f32) {
        let n = w as usize * h as usize * 4;
        super::refine_alpha(from_raw_parts(img, n), from_raw_parts_mut(mask, n), w, h, r, eps)
    }

    #[no_mangle]
    pub unsafe extern "C" fn upsample_mask(small: *const f32, sw: u32, sh: u32, out: *mut f32, w: u32, h: u32, scale: f32) {
        let v = super::upsample_mask(from_raw_parts(small, sw as usize * sh as usize), sw, sh, w, h, scale);
        from_raw_parts_mut(out, v.len()).copy_from_slice(&v);
    }

    #[no_mangle]
    pub unsafe extern "C" fn tone(d: *mut u8, len: usize, filter: u8, seed: u32) {
        super::tone(from_raw_parts_mut(d, len), filter, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_keeps_flat_image() {
        let src = vec![0.5f32; 20 * 10];
        let out = box_blur(&src, 20, 10, 3);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-5));
    }

    #[test]
    fn guided_self_keeps_hard_edge() {
        // 左黑右白的硬邊，磨皮後邊緣兩側仍要分得很開
        let (w, h) = (40, 8);
        let p: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.0 } else { 1.0 }).collect();
        let q = guided_self(&p, w, h, 4, 0.002);
        assert!(q[3 * w + 5] < 0.05 && q[3 * w + 34] > 0.95);
    }

    #[test]
    fn tone_none_is_identity() {
        let mut d = vec![10, 120, 250, 255, 0, 0, 0, 255];
        let before = d.clone();
        tone(&mut d, 0, 1);
        assert_eq!(d, before);
    }

    #[test]
    fn bulge_zero_strength_is_identity() {
        let mut d: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 7 % 256) as u8).collect();
        let before = d.clone();
        bulge(&mut d, 16, 16, 8.0, 8.0, 5.0, 0.0);
        assert_eq!(d, before);
    }

    #[test]
    fn sharpen_flat_image_unchanged() {
        let mut d = vec![100u8; 12 * 12 * 4];
        sharpen(&mut d, 12, 12, &[], 0.6);
        assert!(d.chunks(4).all(|p| p[0] == 100 && p[1] == 100 && p[2] == 100));
    }
}
