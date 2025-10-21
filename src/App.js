import React, { useRef, useState, useEffect } from "react";

// Image Processing Web App
// Single-file React component. Uses canvas for image display + pixel-level processing.
// Features implemented (reference: Gonzalez & Woods):
// - Intensity transformations: negative, log, gamma, contrast stretching, intensity level slicing, bit-plane slicing
// - Histogram processing: histogram, equalization, matching (to a provided target image)
// - Linear filtering: low-pass (Gaussian blur), high-pass (sharpen), band-pass (DoG), custom convolution
// - Spatial enhancement combos: unsharp masking (sharpen via subtracting low-pass), combine filters
// - UI: upload original image, optional upload target for histogram matching, sliders & buttons for parameters

export default function ImageProcessingWebApp() {
  const originalCanvasRef = useRef(null);
  const processedCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const targetInputRef = useRef(null);

  const [imgLoaded, setImgLoaded] = useState(false);
  const [sourceBitmap, setSourceBitmap] = useState(null);
  const [width, setWidth] = useState(600);
  const [height, setHeight] = useState(400);
  const [status, setStatus] = useState("Drop or upload an image to begin.");

  // Parameters
  const [gamma, setGamma] = useState(1.0);
  const [logC, setLogC] = useState(1.0);
  const [stretchLow, setStretchLow] = useState(0);
  const [stretchHigh, setStretchHigh] = useState(255);
  const [sliceMin, setSliceMin] = useState(100);
  const [sliceMax, setSliceMax] = useState(200);
  const [bitPlane, setBitPlane] = useState(7);
  const [gaussSigma, setGaussSigma] = useState(1.4);
  const [sharpenAmount, setSharpenAmount] = useState(1.0);
  const [unsharpAmount, setUnsharpAmount] = useState(1.0);
  const [dogLow, setDogLow] = useState(1.0);
  const [dogHigh, setDogHigh] = useState(2.0);

  // Utility helpers
  function ctxForCanvas(ref) {
    const c = ref.current;
    if (!c) return null;
    return c.getContext("2d");
  }

  // drawing is handled from the `sourceBitmap` useEffect below

  function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Prefer createImageBitmap for blob decoding (better support & performance)
    if (window.createImageBitmap) {
      setStatus('Decoding image...');
      createImageBitmap(file).then((bitmap) => {
        // store bitmap in state and draw within effect
        setSourceBitmap(bitmap);
      }).catch((err) => {
        console.warn('createImageBitmap failed, falling back to FileReader', err);
        // fallback to FileReader
        const reader = new FileReader();
        reader.onload = (ev) => {
          const img = new Image();
          img.onload = () => setSourceBitmap(img);
          img.onerror = () => setStatus('Unable to load image. The file may be an unsupported format (e.g. multi-page TIFF).');
          img.src = ev.target.result;
        };
        reader.onerror = () => setStatus('Failed to read file');
        reader.readAsDataURL(file);
      });
    } else {
      // older browsers
      setStatus('Decoding image (fallback)...');
      const reader = new FileReader();
      reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => setSourceBitmap(img);
        img.onerror = () => setStatus('Unable to load image. The file may be an unsupported format (e.g. multi-page TIFF).');
        img.src = ev.target.result;
      };
      reader.onerror = () => setStatus('Failed to read file');
      reader.readAsDataURL(file);
    }
  }

  // draw into canvases whenever a source bitmap/image is available
  useEffect(() => {
    if (!sourceBitmap) return;
    // use same draw logic as drawImageToCanvas but ensure it runs after render
    const img = sourceBitmap;
    const canvas = originalCanvasRef.current;
    if (!canvas) { setStatus('Original canvas not available'); return; }
    const ctx = canvas.getContext('2d');
    const maxW = 800;
    const scale = Math.min(maxW / img.width, 1);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    canvas.width = w; canvas.height = h;
    if (processedCanvasRef.current) { processedCanvasRef.current.width = w; processedCanvasRef.current.height = h; }
    // keep React state in sync so CSS sizing matches internal buffer size
    setWidth(w); setHeight(h);
    try {
      console.debug('Drawing image to canvases', { w, h, img });
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      const pctx = processedCanvasRef.current && processedCanvasRef.current.getContext('2d');
      if (pctx) { pctx.clearRect(0, 0, w, h); pctx.drawImage(img, 0, 0, w, h); }
      setImgLoaded(true);
      setStatus(`Image loaded (${w}x${h}) — choose an operation.`);
    } catch (err) {
      console.error('Failed to draw sourceBitmap to canvas', err);
      setStatus('Failed to draw image to canvas.');
    }
  }, [sourceBitmap]);

  function getImageData(canvasRef) {
    const ctx = ctxForCanvas(canvasRef);
    if (!ctx || !canvasRef.current) return null;
    try {
      return ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
    } catch (err) {
      setStatus('Unable to read image data from canvas.');
      return null;
    }
  }

  function putImageData(canvasRef, imageData) {
    const ctx = ctxForCanvas(canvasRef);
    if (!ctx || !imageData) return;
    try {
      ctx.putImageData(imageData, 0, 0);
    } catch (err) {
      // fallback: try drawing via a temporary canvas
      try {
        const temp = document.createElement('canvas');
        temp.width = imageData.width;
        temp.height = imageData.height;
        const tctx = temp.getContext('2d');
        tctx.putImageData(imageData, 0, 0);
        ctx.drawImage(temp, 0, 0);
      } catch (e) {
        setStatus('Unable to write image data to canvas.');
      }
    }
  }

  // Pixel helpers: operate on Uint8ClampedArray in RGBA order
  function cloneImageData(imgData) {
    return new ImageData(new Uint8ClampedArray(imgData.data), imgData.width, imgData.height);
  }

  function applyPerPixel(imgData, fn) {
    const out = cloneImageData(imgData);
    const d = out.data;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2], a = d[i + 3];
      const [nr, ng, nb, na] = fn(r, g, b, a, i);
      d[i] = nr; d[i + 1] = ng; d[i + 2] = nb; d[i + 3] = na;
    }
    return out;
  }

  // Intensity transformations
  function negative() {
    if (!imgLoaded) return;
    setStatus("Applying negative...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const out = applyPerPixel(src, (r, g, b, a) => [255 - r, 255 - g, 255 - b, a]);
    putImageData(processedCanvasRef, out);
    setStatus("Negative applied.");
  }

  function logTransform(c = logC) {
    if (!imgLoaded) return;
    setStatus("Applying log transform...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    // log transform: s = c * log(1 + r)
    const out = applyPerPixel(src, (r, g, b, a) => {
      const nr = Math.min(255, c * Math.log(1 + r) * 45); // scale factor for visibility
      const ng = Math.min(255, c * Math.log(1 + g) * 45);
      const nb = Math.min(255, c * Math.log(1 + b) * 45);
      return [nr, ng, nb, a];
    });
    putImageData(processedCanvasRef, out);
    setStatus("Log transform applied.");
  }

  function gammaTransform(g = gamma) {
    if (!imgLoaded) return;
    setStatus("Applying gamma transform...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const inv = 1 / g;
    const lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) lut[i] = Math.min(255, Math.round(255 * Math.pow(i / 255, inv)));
    const out = applyPerPixel(src, (r, g_, b, a) => [lut[r], lut[g_], lut[b], a]);
    putImageData(processedCanvasRef, out);
    setStatus("Gamma transform applied.");
  }

  function contrastStretch(low = stretchLow, high = stretchHigh) {
    if (!imgLoaded) return;
    setStatus("Applying contrast stretching...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const out = applyPerPixel(src, (r, g, b, a) => {
      const map = (v) => {
        if (v <= low) return 0;
        if (v >= high) return 255;
        return Math.round(((v - low) / (high - low)) * 255);
      };
      return [map(r), map(g), map(b), a];
    });
    putImageData(processedCanvasRef, out);
    setStatus("Contrast stretching applied.");
  }

  function levelSlicing(min = sliceMin, max = sliceMax) {
    if (!imgLoaded) return;
    setStatus("Applying intensity level slicing...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const out = applyPerPixel(src, (r, g, b, a) => {
      const to = (v) => (v >= min && v <= max ? 255 : 0);
      return [to(r), to(g), to(b), a];
    });
    putImageData(processedCanvasRef, out);
    setStatus("Level slicing applied.");
  }

  function bitPlaneSlice(plane = bitPlane) {
    if (!imgLoaded) return;
    setStatus("Applying bit-plane slicing...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const mask = 1 << plane;
    const out = applyPerPixel(src, (r, g, b, a) => {
      const rbit = (r & mask) ? 255 : 0;
      const gbit = (g & mask) ? 255 : 0;
      const bbit = (b & mask) ? 255 : 0;
      return [rbit, gbit, bbit, a];
    });
    putImageData(processedCanvasRef, out);
    setStatus(`Bit plane ${plane} shown.`);
  }

  // Histogram helpers
  function computeHistogram(imgData) {
    const hist = {
      r: new Array(256).fill(0),
      g: new Array(256).fill(0),
      b: new Array(256).fill(0),
    };
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      hist.r[d[i]]++;
      hist.g[d[i + 1]]++;
      hist.b[d[i + 2]]++;
    }
    return hist;
  }

  function histogramEqualization() {
    if (!imgLoaded) return;
    setStatus("Applying histogram equalization...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const N = src.width * src.height;
    const hist = computeHistogram(src);
    const cdf = { r: [], g: [], b: [] };
    // cumulative
    let sumR = 0, sumG = 0, sumB = 0;
    for (let i = 0; i < 256; i++) {
      sumR += hist.r[i]; sumG += hist.g[i]; sumB += hist.b[i];
      cdf.r[i] = sumR / N;
      cdf.g[i] = sumG / N;
      cdf.b[i] = sumB / N;
    }
    const lutR = new Uint8ClampedArray(256);
    const lutG = new Uint8ClampedArray(256);
    const lutB = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) {
      lutR[i] = Math.round(255 * cdf.r[i]);
      lutG[i] = Math.round(255 * cdf.g[i]);
      lutB[i] = Math.round(255 * cdf.b[i]);
    }
    const out = applyPerPixel(src, (r, g, b, a) => [lutR[r], lutG[g], lutB[b], a]);
    putImageData(processedCanvasRef, out);
    setStatus("Histogram equalization done.");
  }

  // Histogram matching to a target image
  function histogramMatchingToTarget() {
    if (!imgLoaded) return;
    const targetCanvas = document.createElement("canvas");
    const tctx = targetCanvas.getContext("2d");
    const file = targetInputRef.current.files?.[0];
    if (!file) {
      setStatus("Please upload a target image for histogram matching.");
      return;
    }
    const img = new Image();
    img.onload = () => {
      targetCanvas.width = img.width;
      targetCanvas.height = img.height;
      tctx.drawImage(img, 0, 0);
      const src = getImageData(originalCanvasRef);
      if (!src) { setStatus('No source image loaded for histogram matching.'); return; }
      const targetData = tctx.getImageData(0, 0, targetCanvas.width, targetCanvas.height);
      setStatus("Applying histogram matching to target...");
      const out = matchHistogram(src, targetData);
      putImageData(processedCanvasRef, out);
      setStatus("Histogram matching done.");
    };
    img.src = URL.createObjectURL(file);
  }

  function computeCdfFromHist(hist, N) {
    const cdf = new Array(256);
    let s = 0;
    for (let i = 0; i < 256; i++) {
      s += hist[i];
      cdf[i] = s / N;
    }
    return cdf;
  }

  function matchHistogram(srcImgData, targetImgData) {
    const Nsrc = srcImgData.width * srcImgData.height;
    const Ntar = targetImgData.width * targetImgData.height;
    const srcHist = computeHistogram(srcImgData);
    const tarHist = computeHistogram(targetImgData);
    const cdfRsrc = computeCdfFromHist(srcHist.r, Nsrc);
    const cdfGsrc = computeCdfFromHist(srcHist.g, Nsrc);
    const cdfBsrc = computeCdfFromHist(srcHist.b, Nsrc);
    const cdfRtar = computeCdfFromHist(tarHist.r, Ntar);
    const cdfGtar = computeCdfFromHist(tarHist.g, Ntar);
    const cdfBtar = computeCdfFromHist(tarHist.b, Ntar);
    // build mapping from src intensity to target intensity by matching CDF values
    function buildMap(cdfSrc, cdfTar) {
      const map = new Uint8ClampedArray(256);
      for (let i = 0; i < 256; i++) {
        const val = cdfSrc[i];
        // find j such that cdfTar[j] >= val
        let j = 0;
        while (j < 255 && cdfTar[j] < val) j++;
        map[i] = j;
      }
      return map;
    }
    const mapR = buildMap(cdfRsrc, cdfRtar);
    const mapG = buildMap(cdfGsrc, cdfGtar);
    const mapB = buildMap(cdfBsrc, cdfBtar);
    const out = applyPerPixel(srcImgData, (r, g, b, a) => [mapR[r], mapG[g], mapB[b], a]);
    return out;
  }

  // Convolution (spatial filtering)
  function convolve(srcImgData, kernel, divisor = 1, offset = 0) {
    const w = srcImgData.width;
    const h = srcImgData.height;
    const out = cloneImageData(srcImgData);
    const sd = srcImgData.data;
    const od = out.data;
    const kSize = Math.sqrt(kernel.length);
    const kHalf = Math.floor(kSize / 2);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0;
        for (let ky = -kHalf; ky <= kHalf; ky++) {
          for (let kx = -kHalf; kx <= kHalf; kx++) {
            const ix = Math.min(w - 1, Math.max(0, x + kx));
            const iy = Math.min(h - 1, Math.max(0, y + ky));
            const idx = (iy * w + ix) * 4;
            const kval = kernel[(ky + kHalf) * kSize + (kx + kHalf)];
            r += sd[idx] * kval;
            g += sd[idx + 1] * kval;
            b += sd[idx + 2] * kval;
          }
        }
        const iout = (y * w + x) * 4;
        od[iout] = Math.min(255, Math.max(0, Math.round(r / divisor + offset)));
        od[iout + 1] = Math.min(255, Math.max(0, Math.round(g / divisor + offset)));
        od[iout + 2] = Math.min(255, Math.max(0, Math.round(b / divisor + offset)));
        // alpha unchanged
      }
    }
    return out;
  }

  function lowPassGaussian(sigma = gaussSigma) {
    if (!imgLoaded) return;
    setStatus("Applying Gaussian blur (low-pass)...");
    const k = makeGaussianKernel(sigma);
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const out = convolve(src, k.kernel, k.divisor, 0);
    putImageData(processedCanvasRef, out);
    setStatus("Gaussian blur applied.");
  }

  function highPassSharpen(amount = sharpenAmount) {
    if (!imgLoaded) return;
    setStatus("Applying high-pass sharpening...");
    // Simple sharpening kernel (unsharp-like)
    const kernel = [
      0, -1, 0,
      -1, 5, -1,
      0, -1, 0,
    ];
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const out = convolve(src, kernel, 1, 0);
    putImageData(processedCanvasRef, out);
    setStatus("Sharpen applied.");
  }

  function unsharpMask(amount = unsharpAmount, sigma = gaussSigma) {
    if (!imgLoaded) return;
    setStatus("Applying unsharp masking...");
    const k = makeGaussianKernel(sigma);
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const blurred = convolve(src, k.kernel, k.divisor, 0);
    // USM: result = original + amount * (original - blurred)
    const out = cloneImageData(src);
    const sd = src.data;
    const bd = blurred.data;
    const od = out.data;
    for (let i = 0; i < sd.length; i += 4) {
      od[i] = clamp(sd[i] + amount * (sd[i] - bd[i]));
      od[i + 1] = clamp(sd[i + 1] + amount * (sd[i + 1] - bd[i + 1]));
      od[i + 2] = clamp(sd[i + 2] + amount * (sd[i + 2] - bd[i + 2]));
      od[i + 3] = sd[i + 3];
    }
    putImageData(processedCanvasRef, out);
    setStatus("Unsharp mask applied.");
  }

  function dogBandpass(low = dogLow, high = dogHigh) {
    if (!imgLoaded) return;
    setStatus("Applying band-pass (DoG)...");
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    const k1 = makeGaussianKernel(low);
    const k2 = makeGaussianKernel(high);
    const blur1 = convolve(src, k1.kernel, k1.divisor, 0);
    const blur2 = convolve(src, k2.kernel, k2.divisor, 0);
    // DoG = blur1 - blur2 (selective bandpass)
    const out = cloneImageData(src);
    const od = out.data;
    const d1 = blur1.data;
    const d2 = blur2.data;
    for (let i = 0; i < od.length; i += 4) {
      od[i] = clamp(d1[i] - d2[i] + 128); // +128 to recentre for visibility
      od[i + 1] = clamp(d1[i + 1] - d2[i + 1] + 128);
      od[i + 2] = clamp(d1[i + 2] - d2[i + 2] + 128);
      od[i + 3] = d1[i + 3];
    }
    putImageData(processedCanvasRef, out);
    setStatus("Band-pass (DoG) applied.");
  }

  // Utility: make Gaussian kernel (square) - small radius only
  function makeGaussianKernel(sigma) {
    // kernel size = 2*ceil(3*sigma)+1
    const radius = Math.ceil(3 * sigma);
    const size = radius * 2 + 1;
    const kernel = new Array(size * size).fill(0);
    let sum = 0;
    for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
        const val = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
        kernel[(y + radius) * size + (x + radius)] = val;
        sum += val;
      }
    }
    return { kernel, divisor: sum };
  }

  function clamp(v) {
    return Math.max(0, Math.min(255, Math.round(v)));
  }

  // Combined spatial enhancement: example pipeline
  function enhancePipeline() {
    if (!imgLoaded) return;
    setStatus("Running combined enhancement pipeline (stretch -> unsharp -> equalize)...");
    // 1) contrast stretch using percentiles
    const src = getImageData(originalCanvasRef);
    if (!src) { setStatus('No image data available. Load an image first.'); return; }
    // compute percentiles (2% and 98%) to avoid outliers
    const flat = [];
    for (let i = 0; i < src.data.length; i += 4) {
      // use luminance
      const lum = Math.round(0.2989 * src.data[i] + 0.587 * src.data[i + 1] + 0.114 * src.data[i + 2]);
      flat.push(lum);
    }
    flat.sort((a, b) => a - b);
    const p2 = flat[Math.floor(flat.length * 0.02)];
    const p98 = flat[Math.floor(flat.length * 0.98)];
    // stretch
    const stretched = applyPerPixel(src, (r, g, b, a) => {
      const map = (v) => {
        if (v <= p2) return 0;
        if (v >= p98) return 255;
        return Math.round(((v - p2) / (p98 - p2)) * 255);
      };
      return [map(r), map(g), map(b), a];
    });
    // unsharp mask on stretched
    const k = makeGaussianKernel(1.0);
    const blurred = convolve(stretched, k.kernel, k.divisor, 0);
    const out = cloneImageData(stretched);
    for (let i = 0; i < out.data.length; i += 4) {
      out.data[i] = clamp(stretched.data[i] + 1.2 * (stretched.data[i] - blurred.data[i]));
      out.data[i + 1] = clamp(stretched.data[i + 1] + 1.2 * (stretched.data[i + 1] - blurred.data[i + 1]));
      out.data[i + 2] = clamp(stretched.data[i + 2] + 1.2 * (stretched.data[i + 2] - blurred.data[i + 2]));
    }
    // equalize final
    const final = histogramEqualizeImageData(out);
    putImageData(processedCanvasRef, final);
    setStatus("Enhancement pipeline complete.");
  }

  function histogramEqualizeImageData(imgData) {
    const N = imgData.width * imgData.height;
    const hist = { r: new Array(256).fill(0), g: new Array(256).fill(0), b: new Array(256).fill(0) };
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      hist.r[d[i]]++;
      hist.g[d[i + 1]]++;
      hist.b[d[i + 2]]++;
    }
    const cdf = { r: [], g: [], b: [] };
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0; i < 256; i++) {
      sr += hist.r[i]; sg += hist.g[i]; sb += hist.b[i];
      cdf.r[i] = sr / N; cdf.g[i] = sg / N; cdf.b[i] = sb / N;
    }
    const lutR = new Uint8ClampedArray(256);
    const lutG = new Uint8ClampedArray(256);
    const lutB = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) {
      lutR[i] = Math.round(255 * cdf.r[i]);
      lutG[i] = Math.round(255 * cdf.g[i]);
      lutB[i] = Math.round(255 * cdf.b[i]);
    }
    const out = cloneImageData(imgData);
    for (let i = 0; i < d.length; i += 4) {
      out.data[i] = lutR[d[i]];
      out.data[i + 1] = lutG[d[i + 1]];
      out.data[i + 2] = lutB[d[i + 2]];
      out.data[i + 3] = d[i + 3];
    }
    return out;
  }

  // Export processed image
  function downloadProcessed() {
    const link = document.createElement("a");
    link.download = "processed.png";
    link.href = processedCanvasRef.current.toDataURL();
    link.click();
  }

  // Reset processed to original
  function resetToOriginal() {
    if (!imgLoaded) return;
    const src = getImageData(originalCanvasRef);
    putImageData(processedCanvasRef, src);
    setStatus("Reset to original.");
  }

  return (
    <div className="p-4 max-w-6xl mx-auto font-sans">
      <h1 className="text-2xl font-bold mb-3">Image Processing Playground — Gonzalez & Woods inspired</h1>
      <p className="mb-2 text-sm text-gray-600">Upload an image, then try intensity transforms, histogram ops, filters and enhancement pipelines.</p>

      <div className="flex gap-4 mb-4">
        <div>
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFile} />
        </div>
        <div>
          <input ref={targetInputRef} type="file" accept="image/*" />
          <div className="text-xs text-gray-500">(optional) target image for histogram matching</div>
        </div>
        <div>
          <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={() => { resetToOriginal(); }}>Reset</button>
        </div>
        <div>
          <button className="px-3 py-1 bg-green-600 text-white rounded" onClick={downloadProcessed}>Download</button>
        </div>
      </div>

      <div style={{display: 'flex', gap: 12, alignItems: 'flex-start'}}>
        <div style={{flex: '1 1 auto'}}>
          <div className="border p-2" style={{boxSizing: 'border-box'}}>
            <div className="text-sm font-medium mb-1">Original</div>
            <canvas ref={originalCanvasRef} className="border" style={{display: 'block', boxSizing: 'border-box', width: width + 'px', height: height + 'px'}}></canvas>
          </div>
        </div>

        {/* visual divider */}
        <div style={{width: 2, background: '#e5e7eb', borderRadius: 2}} aria-hidden="true"></div>

        <div style={{flex: '1 1 auto'}}>
          <div className="border p-2" style={{boxSizing: 'border-box'}}>
            <div className="text-sm font-medium mb-1">Processed</div>
            <canvas ref={processedCanvasRef} className="border" style={{display: 'block', boxSizing: 'border-box', width: width + 'px', height: height + 'px'}}></canvas>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="p-2 border rounded">
          <h3 className="font-semibold mb-2">Intensity transforms</h3>
          <div className="flex flex-col gap-2">
            <button className="btn" onClick={negative}>Negative</button>
            <div>
              <label className="text-xs">Log c: {logC}</label>
              <input type="range" min="0.1" max="5" step="0.1" value={logC} onChange={(e) => setLogC(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => logTransform(logC)}>Apply Log</button>
            </div>
            <div>
              <label className="text-xs">Gamma: {gamma}</label>
              <input type="range" min="0.1" max="3" step="0.1" value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => gammaTransform(gamma)}>Apply Gamma</button>
            </div>
            <div>
              <label className="text-xs">Stretch low:{stretchLow} high:{stretchHigh}</label>
              <input type="range" min="0" max="255" value={stretchLow} onChange={(e) => setStretchLow(parseInt(e.target.value))} />
              <input type="range" min="0" max="255" value={stretchHigh} onChange={(e) => setStretchHigh(parseInt(e.target.value))} />
              <button className="btn" onClick={() => contrastStretch(stretchLow, stretchHigh)}>Contrast Stretch</button>
            </div>
            <div>
              <label className="text-xs">Slice min:{sliceMin} max:{sliceMax}</label>
              <input type="range" min="0" max="255" value={sliceMin} onChange={(e) => setSliceMin(parseInt(e.target.value))} />
              <input type="range" min="0" max="255" value={sliceMax} onChange={(e) => setSliceMax(parseInt(e.target.value))} />
              <button className="btn" onClick={() => levelSlicing(sliceMin, sliceMax)}>Level Slicing</button>
            </div>
            <div>
              <label className="text-xs">Bit plane: {bitPlane}</label>
              <input type="range" min="0" max="7" value={bitPlane} onChange={(e) => setBitPlane(parseInt(e.target.value))} />
              <button className="btn" onClick={() => bitPlaneSlice(bitPlane)}>Bit-plane Slice</button>
            </div>
          </div>
        </div>

        <div className="p-2 border rounded">
          <h3 className="font-semibold mb-2">Histogram</h3>
          <div className="flex flex-col gap-2">
            <button className="btn" onClick={histogramEqualization}>Equalize Histogram</button>
            <button className="btn" onClick={histogramMatchingToTarget}>Match to Target (upload target)</button>
          </div>
        </div>

        <div className="p-2 border rounded">
          <h3 className="font-semibold mb-2">Filters & Enhancement</h3>
          <div className="flex flex-col gap-2">
            <div>
              <label className="text-xs">Gaussian sigma: {gaussSigma}</label>
              <input type="range" min="0.5" max="5" step="0.1" value={gaussSigma} onChange={(e) => setGaussSigma(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => lowPassGaussian(gaussSigma)}>Gaussian (Low-pass)</button>
            </div>
            <div>
              <label className="text-xs">Sharpen amount: {sharpenAmount}</label>
              <input type="range" min="0.5" max="3" step="0.1" value={sharpenAmount} onChange={(e) => setSharpenAmount(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => highPassSharpen(sharpenAmount)}>High-pass Sharpen</button>
            </div>
            <div>
              <label className="text-xs">Unsharp amount: {unsharpAmount}</label>
              <input type="range" min="0.1" max="3" step="0.1" value={unsharpAmount} onChange={(e) => setUnsharpAmount(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => unsharpMask(unsharpAmount, gaussSigma)}>Unsharp Mask</button>
            </div>
            <div>
              <label className="text-xs">DoG low:{dogLow} high:{dogHigh}</label>
              <input type="range" min="0.5" max="4" step="0.1" value={dogLow} onChange={(e) => setDogLow(parseFloat(e.target.value))} />
              <input type="range" min="0.5" max="6" step="0.1" value={dogHigh} onChange={(e) => setDogHigh(parseFloat(e.target.value))} />
              <button className="btn" onClick={() => dogBandpass(dogLow, dogHigh)}>Band-pass (DoG)</button>
            </div>
            <button className="btn" onClick={enhancePipeline}>Combined enhancement pipeline</button>
          </div>
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-700">
        <strong>Status:</strong> {status}
      </div>

      <style>{`
        .btn{ display:inline-block;padding:6px 10px;background:#111827;color:white;border-radius:6px;font-size:13px }
        input[type=range]{width:100%}
      `}</style>

    </div>
  );
}
