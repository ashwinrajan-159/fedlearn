importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let session = null;
let ws = null;
let alpha = 0.1; // Learning rate
let computeThreads = 1;

// ═══════════════════════════════════════════════════════════════════════════════
//  Enhancement Configuration (set via INIT message from main thread)
// ═══════════════════════════════════════════════════════════════════════════════
let DP_CLIP_C   = 0;       // L2 clipping bound (0 = disabled)
let DP_SIGMA    = 0;       // Gaussian noise σ (0 = disabled)
let DP_DELTA    = 1e-5;    // δ for (ε,δ)-DP
let TOP_K_FRAC  = 0;       // Top-K sparsification fraction (0 = disabled)
let QUANTIZE    = "none";  // "none" or "int8"
let FEDPROX_MU  = 0;       // μ for proximal term (0 = FedAvg)
let LOCAL_STEPS = 1;       // number of local SGD steps

const NUM_FEATURES = 4;
const NUM_CLASSES  = 3;
const W_SIZE       = NUM_FEATURES * NUM_CLASSES;  // 12
const B_SIZE       = NUM_CLASSES;                  // 3
const GRAD_DIM     = W_SIZE + B_SIZE;              // 15

// Compression mode constants (match server protocol)
const COMPRESS_NONE      = 0;
const COMPRESS_INT8      = 1;
const COMPRESS_TOPK      = 2;
const COMPRESS_TOPK_INT8 = 3;

// Error feedback buffer (persists across rounds)
let errorBuffer = new Float32Array(GRAD_DIM);

// ═══════════════════════════════════════════════════════════════════════════════
//  RDP Privacy Accountant
// ═══════════════════════════════════════════════════════════════════════════════
let rdpSteps = 0;

function computePrivacyBudget() {
    if (DP_CLIP_C <= 0 || DP_SIGMA <= 0 || rdpSteps === 0) {
        return { epsilon: 0, delta: DP_DELTA, steps: rdpSteps };
    }
    const alphas = [];
    for (let a = 1.01; a <= 100; a += (a < 10 ? 0.1 : 1.0)) {
        alphas.push(a);
    }
    let bestEps = Infinity;
    for (const a of alphas) {
        const rdpPerStep = (a * DP_CLIP_C * DP_CLIP_C) / (2 * DP_SIGMA * DP_SIGMA);
        const rdpTotal = rdpSteps * rdpPerStep;
        const eps = rdpTotal + Math.log(1.0 / DP_DELTA) / (a - 1);
        if (eps < bestEps) bestEps = eps;
    }
    return { epsilon: bestEps, delta: DP_DELTA, steps: rdpSteps };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Gradient Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/** L2-clip a flat gradient in-place. Returns original norm. */
function clipGradientL2(grad, C) {
    let norm = 0;
    for (let i = 0; i < grad.length; i++) norm += grad[i] * grad[i];
    norm = Math.sqrt(norm);
    const scale = Math.max(1.0, norm / C);
    if (scale > 1.0) {
        for (let i = 0; i < grad.length; i++) grad[i] /= scale;
    }
    return norm;
}

/** Add i.i.d. Gaussian noise N(0, σ²) via Box-Muller. */
function addGaussianNoise(grad, sigma) {
    for (let i = 0; i < grad.length; i += 2) {
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        grad[i] += sigma * r * Math.cos(theta);
        if (i + 1 < grad.length) {
            grad[i + 1] += sigma * r * Math.sin(theta);
        }
    }
}

/** Top-K sparsification. */
function topKSparsify(grad, fraction) {
    const k = Math.max(1, Math.round(grad.length * fraction));
    const indexed = [];
    for (let i = 0; i < grad.length; i++) {
        indexed.push({ idx: i, absVal: Math.abs(grad[i]) });
    }
    indexed.sort((a, b) => b.absVal - a.absVal);
    const topIndices = new Uint16Array(k);
    const topValues = new Float32Array(k);
    for (let i = 0; i < k; i++) {
        topIndices[i] = indexed[i].idx;
        topValues[i] = grad[indexed[i].idx];
    }
    return { indices: topIndices, values: topValues, k };
}

/** INT8 symmetric quantization. */
function quantizeInt8(values) {
    let maxAbs = 0;
    for (let i = 0; i < values.length; i++) {
        const a = Math.abs(values[i]);
        if (a > maxAbs) maxAbs = a;
    }
    const scale = maxAbs || 1e-10;
    const quantized = new Int8Array(values.length);
    for (let i = 0; i < values.length; i++) {
        let q = Math.round(127 * values[i] / scale);
        q = Math.max(-128, Math.min(127, q));
        quantized[i] = q;
    }
    return { quantized, scale };
}

/**
 * Compress gradient with error feedback, returning an ArrayBuffer
 * suitable for sending over WebSocket.
 */
function compressAndPack(grad) {
    // p_t = g_t + e_t
    const corrected = new Float32Array(grad.length);
    for (let i = 0; i < grad.length; i++) {
        corrected[i] = grad[i] + errorBuffer[i];
    }

    const useTopK = TOP_K_FRAC > 0 && TOP_K_FRAC < 1;
    const useQuant = QUANTIZE === "int8";

    if (useTopK && useQuant) {
        // Mode 3: Top-K + INT8
        const { indices, values, k } = topKSparsify(corrected, TOP_K_FRAC);
        const { quantized, scale } = quantizeInt8(values);
        const reconstructed = new Float32Array(grad.length);
        for (let i = 0; i < k; i++) {
            reconstructed[indices[i]] = quantized[i] * (scale / 127);
        }
        for (let i = 0; i < grad.length; i++) {
            errorBuffer[i] = corrected[i] - reconstructed[i];
        }
        const totalBytes = 4 + 4 + 4 + 2 * k + k;
        const buf = new ArrayBuffer(totalBytes);
        const view = new DataView(buf);
        view.setUint32(0, COMPRESS_TOPK_INT8, true);
        view.setUint32(4, k, true);
        view.setFloat32(8, scale, true);
        let offset = 12;
        for (let i = 0; i < k; i++) { view.setUint16(offset, indices[i], true); offset += 2; }
        for (let i = 0; i < k; i++) { view.setInt8(offset, quantized[i]); offset += 1; }
        return buf;

    } else if (useTopK) {
        // Mode 2: Top-K Float32
        const { indices, values, k } = topKSparsify(corrected, TOP_K_FRAC);
        const reconstructed = new Float32Array(grad.length);
        for (let i = 0; i < k; i++) reconstructed[indices[i]] = values[i];
        for (let i = 0; i < grad.length; i++) errorBuffer[i] = corrected[i] - reconstructed[i];
        const totalBytes = 4 + 4 + 2 * k + 4 * k;
        const buf = new ArrayBuffer(totalBytes);
        const view = new DataView(buf);
        view.setUint32(0, COMPRESS_TOPK, true);
        view.setUint32(4, k, true);
        let offset = 8;
        for (let i = 0; i < k; i++) { view.setUint16(offset, indices[i], true); offset += 2; }
        for (let i = 0; i < k; i++) { view.setFloat32(offset, values[i], true); offset += 4; }
        return buf;

    } else if (useQuant) {
        // Mode 1: INT8 full vector
        const { quantized, scale } = quantizeInt8(corrected);
        for (let i = 0; i < grad.length; i++) {
            errorBuffer[i] = corrected[i] - quantized[i] * (scale / 127);
        }
        const totalBytes = 4 + 4 + grad.length;
        const buf = new ArrayBuffer(totalBytes);
        const view = new DataView(buf);
        view.setUint32(0, COMPRESS_INT8, true);
        view.setFloat32(4, scale, true);
        for (let i = 0; i < grad.length; i++) view.setInt8(8 + i, quantized[i]);
        return buf;

    } else {
        // Mode 0: Uncompressed
        for (let i = 0; i < grad.length; i++) errorBuffer[i] = 0;
        const buf = new ArrayBuffer(4 + grad.length * 4);
        const view = new DataView(buf);
        view.setUint32(0, COMPRESS_NONE, true);
        const floats = new Float32Array(buf, 4);
        floats.set(corrected);
        return buf;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Worker Initialization
// ═══════════════════════════════════════════════════════════════════════════════

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === "INIT") {
        computeThreads = payload.threads || 1;
        ort.env.wasm.numThreads = computeThreads;

        // Read enhancement config from main thread
        if (payload.dpClipC !== undefined)   DP_CLIP_C   = payload.dpClipC;
        if (payload.dpSigma !== undefined)   DP_SIGMA    = payload.dpSigma;
        if (payload.dpDelta !== undefined)   DP_DELTA    = payload.dpDelta;
        if (payload.topKFrac !== undefined)  TOP_K_FRAC  = payload.topKFrac;
        if (payload.quantize !== undefined)  QUANTIZE    = payload.quantize;
        if (payload.fedproxMu !== undefined) FEDPROX_MU  = payload.fedproxMu;
        if (payload.localSteps !== undefined) LOCAL_STEPS = payload.localSteps;

        // Initialize ONNX session
        try {
            session = await ort.InferenceSession.create('http://localhost:8000/host/model.onnx');
            self.postMessage({ type: "STATUS", msg: "ONNX model loaded." });

            // Report active features
            const features = [];
            if (DP_CLIP_C > 0)   features.push(`DP(C=${DP_CLIP_C},σ=${DP_SIGMA})`);
            if (TOP_K_FRAC > 0)  features.push(`Top-K(${(TOP_K_FRAC*100).toFixed(1)}%)`);
            if (QUANTIZE !== "none") features.push(`Quant(${QUANTIZE})`);
            if (FEDPROX_MU > 0)  features.push(`FedProx(μ=${FEDPROX_MU})`);
            if (LOCAL_STEPS > 1) features.push(`Steps=${LOCAL_STEPS}`);
            if (features.length > 0) {
                self.postMessage({ type: "STATUS", msg: `Enhancements: ${features.join(", ")}` });
            }

            connectWebSocket();
        } catch (err) {
            self.postMessage({ type: "ERROR", msg: `Failed to load model: ${err.message}` });
        }
    }
};

function connectWebSocket() {
    ws = new WebSocket("ws://localhost:8080");
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        self.postMessage({ type: "STATUS", msg: "Connected to Coordinator." });
    };

    ws.onmessage = async (e) => {
        if (e.data instanceof ArrayBuffer) {
            await processBatch(e.data);
        }
    };

    ws.onclose = () => {
        self.postMessage({ type: "STATUS", msg: "Disconnected from Coordinator." });
        setTimeout(connectWebSocket, 5000);
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Batch Processing — FedProx + DP + Compression
// ═══════════════════════════════════════════════════════════════════════════════

async function processBatch(buffer) {
    self.postMessage({ type: "STATUS", msg: "Processing batch..." });
    const t0 = performance.now();

    // Unpack Binary Payload
    const dataView = new DataView(buffer);
    const B = dataView.getUint32(0, true);

    const floats = new Float32Array(buffer, 4);
    const W_global = floats.slice(0, 12);
    const bias_global = floats.slice(12, 15);
    const X = floats.slice(15, 15 + B * 4);
    const Y_true = floats.slice(15 + B * 4, 15 + B * 4 + B * 3);

    // ═══════════════════════════════════════════════════════════════════════
    //  FedProx Multi-Step Local Training
    // ═══════════════════════════════════════════════════════════════════════
    const W_local = new Float32Array(W_global);
    const bias_local = new Float32Array(bias_global);

    let loss = 0;

    for (let step = 0; step < LOCAL_STEPS; step++) {
        // Create ONNX tensors with current local weights
        const tensorX = new ort.Tensor('float32', X, [B, 4]);
        const tensorW = new ort.Tensor('float32', new Float32Array(W_local), [4, 3]);
        const tensorB = new ort.Tensor('float32', new Float32Array(bias_local), [3]);

        // Forward pass
        const feeds = { X: tensorX, W: tensorW, B: tensorB };
        const results = await session.run(feeds);
        const probs = results.probs.data;

        // Compute gradients: dZ = P - Y
        const dZ = new Float32Array(B * 3);
        loss = 0;

        for (let i = 0; i < B * 3; i++) {
            dZ[i] = probs[i] - Y_true[i];
            if (Y_true[i] > 0) {
                loss -= Math.log(Math.max(probs[i], 1e-7));
            }
        }
        loss /= B;

        // dW = X^T * dZ
        const dW = new Float32Array(12);
        const db = new Float32Array(3);

        for (let b = 0; b < B; b++) {
            for (let j = 0; j < 3; j++) {
                const dz_val = dZ[b * 3 + j];
                db[j] += dz_val;
                for (let i = 0; i < 4; i++) {
                    dW[i * 3 + j] += X[b * 4 + i] * dz_val;
                }
            }
        }

        // Apply gradient with FedProx proximal term
        for (let i = 0; i < 12; i++) {
            let grad = (alpha * dW[i]) / B;
            if (FEDPROX_MU > 0) {
                grad += FEDPROX_MU * (W_local[i] - W_global[i]);
            }
            W_local[i] -= grad;
        }
        for (let j = 0; j < 3; j++) {
            let grad = (alpha * db[j]) / B;
            if (FEDPROX_MU > 0) {
                grad += FEDPROX_MU * (bias_local[j] - bias_global[j]);
            }
            bias_local[j] -= grad;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Compute gradient delta: ΔW = W_global - W_local
    // ═══════════════════════════════════════════════════════════════════════
    const flatGrad = new Float32Array(GRAD_DIM);
    for (let i = 0; i < 12; i++) flatGrad[i] = W_global[i] - W_local[i];
    for (let j = 0; j < 3; j++) flatGrad[12 + j] = bias_global[j] - bias_local[j];

    // ═══════════════════════════════════════════════════════════════════════
    //  Differential Privacy: L2 Clipping + Gaussian Noise
    // ═══════════════════════════════════════════════════════════════════════
    if (DP_CLIP_C > 0) clipGradientL2(flatGrad, DP_CLIP_C);
    if (DP_SIGMA > 0) {
        addGaussianNoise(flatGrad, DP_SIGMA);
        rdpSteps++;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Compression + Error Feedback
    // ═══════════════════════════════════════════════════════════════════════
    const useCompression = (TOP_K_FRAC > 0 && TOP_K_FRAC < 1) || QUANTIZE === "int8";
    let responseBuffer;

    if (useCompression) {
        responseBuffer = compressAndPack(flatGrad);
    } else {
        // Legacy 60-byte uncompressed format (backward compatible)
        responseBuffer = new ArrayBuffer(60);
        const responseFloats = new Float32Array(responseBuffer);
        responseFloats.set(flatGrad.subarray(0, 12), 0);
        responseFloats.set(flatGrad.subarray(12), 12);
    }

    // Send back to Host
    ws.send(responseBuffer);

    const t1 = performance.now();

    // Build result message
    let resultMsg = `Processed batch (loss: ${loss.toFixed(4)}) in ${(t1-t0).toFixed(1)}ms`;
    resultMsg += ` [${responseBuffer.byteLength}B]`;
    if (LOCAL_STEPS > 1) resultMsg += ` [${LOCAL_STEPS} steps]`;

    // Report privacy budget periodically
    if (DP_SIGMA > 0 && rdpSteps % 10 === 0) {
        const budget = computePrivacyBudget();
        resultMsg += ` [ε=${budget.epsilon.toFixed(2)}]`;
    }

    self.postMessage({ type: "RESULT", msg: resultMsg });
}
