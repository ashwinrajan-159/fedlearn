"use strict";

const WebSocket = require("ws");
const os = require("os");

// ===========================================================================
//  CLI Configuration
// ===========================================================================
const args = process.argv.slice(2);

function flag(name, fallback) {
  const idx = args.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return args[idx + 1] ?? fallback;
}

const TARGET_URL   = flag("url", "ws://localhost:8080");
const NUM_WORKERS  = parseInt(flag("workers", "3"), 10);
const MIN_DELAY_MS = parseInt(flag("min-delay", "150"), 10);
const MAX_DELAY_MS = parseInt(flag("max-delay", "500"), 10);
const FAULT_RATE   = parseFloat(flag("fault-rate", "0"));     // 0–1, chance a worker drops a response
const POISON_RATE  = parseFloat(flag("poison-rate", "0"));    // 0–1, chance of sending malicious grads
const JOB_ID       = flag("job-id", null);

// --- Differential Privacy ---
const DP_CLIP_C    = parseFloat(flag("dp-c", "0"));           // L2 clipping bound C (0 = disabled)
const DP_SIGMA     = parseFloat(flag("dp-sigma", "0"));       // Gaussian noise σ (0 = disabled)
const DP_DELTA     = parseFloat(flag("dp-delta", "1e-5"));    // δ for (ε,δ)-DP

// --- Gradient Compression ---
const TOP_K_FRAC   = parseFloat(flag("top-k", "0"));          // fraction (0–1), e.g. 0.1 = 10%
const QUANTIZE     = flag("quantize", "none");                 // "none", "int8"

// --- FedProx ---
const FEDPROX_MU   = parseFloat(flag("fedprox-mu", "0"));     // μ for proximal term (0 = standard FedAvg)
const LOCAL_STEPS  = parseInt(flag("local-steps", "1"), 10);   // number of local SGD steps

// ===========================================================================
//  Constants
// ===========================================================================
const NUM_FEATURES = 4;
const NUM_CLASSES  = 3;
const W_SIZE       = NUM_FEATURES * NUM_CLASSES;  // 12
const B_SIZE       = NUM_CLASSES;                  // 3
const GRAD_DIM     = W_SIZE + B_SIZE;              // 15

// Compression mode constants (must match server)
const COMPRESS_NONE      = 0;
const COMPRESS_INT8      = 1;
const COMPRESS_TOPK      = 2;
const COMPRESS_TOPK_INT8 = 3;

// ===========================================================================
//  ANSI Helpers
// ===========================================================================
const C = {
  reset:   "\x1b[0m",
  bold:    "\x1b[1m",
  dim:     "\x1b[2m",
  red:     "\x1b[31m",
  green:   "\x1b[32m",
  yellow:  "\x1b[33m",
  blue:    "\x1b[34m",
  magenta: "\x1b[35m",
  cyan:    "\x1b[36m",
  white:   "\x1b[37m",
  gray:    "\x1b[90m",
  bgBlue:  "\x1b[44m",
  bgGreen: "\x1b[42m",
  bgRed:   "\x1b[41m",
};

const WORKER_COLORS = [C.cyan, C.magenta, C.yellow, C.green, C.blue, C.red];

// ===========================================================================
//  Global Statistics
// ===========================================================================
const startTime = Date.now();
let totalBatches = 0;
let totalGradientsSent = 0;
let totalFaults = 0;
let totalPoisoned = 0;
let totalReconnections = 0;

// ===========================================================================
//  Differential Privacy — Rényi DP (RDP) Budget Accountant
//
//  For the Gaussian mechanism with L2 sensitivity C and noise σ:
//    RDP at order α:  ε(α) = α C² / (2σ²)
//
//  Composition over T rounds:  ε_total(α) = T · ε(α)
//
//  Conversion to (ε, δ)-DP:
//    ε = min over α of [ ε_total(α) + log(1/δ) / (α - 1) ]
// ===========================================================================

class RDPAccountant {
  constructor(clipC, sigma, delta) {
    this.clipC = clipC;
    this.sigma = sigma;
    this.delta = delta;
    this.steps = 0;

    // Pre-compute Rényi orders to search over
    this.alphas = [];
    for (let a = 1.01; a <= 100; a += (a < 10 ? 0.1 : 1.0)) {
      this.alphas.push(a);
    }
  }

  /** Record one DP mechanism application. */
  step() {
    this.steps++;
  }

  /**
   * Compute the current (ε, δ)-DP guarantee via RDP → (ε, δ) conversion.
   * @returns {{ epsilon: number, delta: number, steps: number }}
   */
  getPrivacySpent() {
    if (this.sigma <= 0 || this.clipC <= 0 || this.steps === 0) {
      return { epsilon: 0, delta: this.delta, steps: this.steps };
    }

    let bestEps = Infinity;
    for (const alpha of this.alphas) {
      // Per-step RDP at order alpha for Gaussian mechanism
      const rdpPerStep = (alpha * this.clipC * this.clipC) / (2 * this.sigma * this.sigma);
      // Composition over T steps
      const rdpTotal = this.steps * rdpPerStep;
      // Convert to (ε, δ)-DP
      const eps = rdpTotal + Math.log(1.0 / this.delta) / (alpha - 1);
      if (eps < bestEps) {
        bestEps = eps;
      }
    }

    return { epsilon: bestEps, delta: this.delta, steps: this.steps };
  }
}

// ===========================================================================
//  Gradient Utilities
// ===========================================================================

/**
 * L2-clip a flat gradient vector in-place.
 *   g_clipped = g / max(1, ||g|| / C)
 *
 * @param {Float32Array} grad  — flat gradient (mutated)
 * @param {number} C           — clipping bound
 * @returns {number}           — original L2 norm before clipping
 */
function clipGradientL2(grad, C) {
  let norm = 0;
  for (let i = 0; i < grad.length; i++) {
    norm += grad[i] * grad[i];
  }
  norm = Math.sqrt(norm);
  const scale = Math.max(1.0, norm / C);
  if (scale > 1.0) {
    for (let i = 0; i < grad.length; i++) {
      grad[i] /= scale;
    }
  }
  return norm;
}

/**
 * Add i.i.d. Gaussian noise N(0, σ²) to each element of the gradient.
 * Uses the Box-Muller transform for generating Gaussian samples.
 *
 * @param {Float32Array} grad  — flat gradient (mutated)
 * @param {number} sigma       — noise standard deviation
 */
function addGaussianNoise(grad, sigma) {
  for (let i = 0; i < grad.length; i += 2) {
    // Box-Muller transform: generate 2 independent N(0,1) samples
    const u1 = Math.random() || 1e-10;  // avoid log(0)
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    grad[i] += sigma * r * Math.cos(theta);
    if (i + 1 < grad.length) {
      grad[i + 1] += sigma * r * Math.sin(theta);
    }
  }
}

/**
 * Top-K sparsification: keep only the top k elements by absolute value.
 *
 * @param {Float32Array} grad — flat gradient
 * @param {number} fraction   — fraction of elements to keep (0–1)
 * @returns {{ indices: Uint16Array, values: Float32Array }}
 */
function topKSparsify(grad, fraction) {
  const k = Math.max(1, Math.round(grad.length * fraction));

  // Build (index, |value|) pairs and partially sort to find top-k
  const indexed = [];
  for (let i = 0; i < grad.length; i++) {
    indexed.push({ idx: i, absVal: Math.abs(grad[i]) });
  }
  indexed.sort((a, b) => b.absVal - a.absVal);

  const topIndices = new Uint32Array(k);
  const topValues = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    topIndices[i] = indexed[i].idx;
    topValues[i] = grad[indexed[i].idx];
  }

  return { indices: topIndices, values: topValues, k };
}

/**
 * INT8 symmetric quantization.
 *   scale = max(|X|)
 *   q = round(127 * X / scale), clipped to [-128, 127]
 *
 * @param {Float32Array} values
 * @returns {{ quantized: Int8Array, scale: number }}
 */
function quantizeInt8(values) {
  let maxAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs || 1e-10;  // avoid div-by-zero

  const quantized = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    let q = Math.round(127 * values[i] / scale);
    q = Math.max(-128, Math.min(127, q));
    quantized[i] = q;
  }

  return { quantized, scale };
}

/**
 * Apply error feedback: add accumulated error to current gradient,
 * compress, and compute new error for next round.
 *
 * @param {Float32Array} grad       — current raw gradient
 * @param {Float32Array} errorBuf   — accumulated error from previous rounds (mutated)
 * @param {number} topKFrac         — Top-K fraction (0 = no sparsification)
 * @param {string} quantMode        — "none" or "int8"
 * @returns {Buffer}                — packed binary payload with compression header
 */
function compressWithErrorFeedback(grad, errorBuf, topKFrac, quantMode) {
  // p_t = g_t + e_t
  const corrected = new Float32Array(grad.length);
  for (let i = 0; i < grad.length; i++) {
    corrected[i] = grad[i] + errorBuf[i];
  }

  let packedBuffer;
  const useTopK = topKFrac > 0 && topKFrac < 1;
  const useQuant = quantMode === "int8";

  if (useTopK && useQuant) {
    // ── Mode 3: Top-K + INT8 ──────────────────────────────────────────
    const { indices, values, k } = topKSparsify(corrected, topKFrac);
    const { quantized, scale } = quantizeInt8(values);

    // Compute reconstructed for error feedback
    const reconstructed = new Float32Array(grad.length);  // zeros
    for (let i = 0; i < k; i++) {
      reconstructed[indices[i]] = quantized[i] * (scale / 127);
    }

    // e_{t+1} = p_t - c_t
    for (let i = 0; i < grad.length; i++) {
      errorBuf[i] = corrected[i] - reconstructed[i];
    }

    // Pack: [mode(4)] [nnz(4)] [scale(4)] [indices(4*k)] [qvals(k)]
    const totalBytes = 4 + 4 + 4 + 4 * k + k;
    packedBuffer = Buffer.alloc(totalBytes);
    packedBuffer.writeUInt32LE(COMPRESS_TOPK_INT8, 0);
    packedBuffer.writeUInt32LE(k, 4);
    packedBuffer.writeFloatLE(scale, 8);
    let offset = 12;
    for (let i = 0; i < k; i++) {
      packedBuffer.writeUInt32LE(indices[i], offset);
      offset += 4;
    }
    for (let i = 0; i < k; i++) {
      packedBuffer.writeInt8(quantized[i], offset);
      offset += 1;
    }

  } else if (useTopK) {
    // ── Mode 2: Top-K (Float32 values) ────────────────────────────────
    const { indices, values, k } = topKSparsify(corrected, topKFrac);

    const reconstructed = new Float32Array(grad.length);
    for (let i = 0; i < k; i++) {
      reconstructed[indices[i]] = values[i];
    }
    for (let i = 0; i < grad.length; i++) {
      errorBuf[i] = corrected[i] - reconstructed[i];
    }

    // Pack: [mode(4)] [nnz(4)] [indices(4*k)] [values(4*k)]
    const totalBytes = 4 + 4 + 4 * k + 4 * k;
    packedBuffer = Buffer.alloc(totalBytes);
    packedBuffer.writeUInt32LE(COMPRESS_TOPK, 0);
    packedBuffer.writeUInt32LE(k, 4);
    let offset = 8;
    for (let i = 0; i < k; i++) {
      packedBuffer.writeUInt32LE(indices[i], offset);
      offset += 4;
    }
    for (let i = 0; i < k; i++) {
      packedBuffer.writeFloatLE(values[i], offset);
      offset += 4;
    }

  } else if (useQuant) {
    // ── Mode 1: INT8 quantization (full vector) ──────────────────────
    const { quantized, scale } = quantizeInt8(corrected);

    // Reconstructed = q * (scale / 127)
    for (let i = 0; i < grad.length; i++) {
      const reconstructed = quantized[i] * (scale / 127);
      errorBuf[i] = corrected[i] - reconstructed;
    }

    // Pack: [mode(4)] [scale(4)] [qvals(15)]
    const totalBytes = 4 + 4 + grad.length;
    packedBuffer = Buffer.alloc(totalBytes);
    packedBuffer.writeUInt32LE(COMPRESS_INT8, 0);
    packedBuffer.writeFloatLE(scale, 4);
    for (let i = 0; i < grad.length; i++) {
      packedBuffer.writeInt8(quantized[i], 8 + i);
    }

  } else {
    // ── Mode 0: No compression ───────────────────────────────────────
    // No error feedback needed for uncompressed
    for (let i = 0; i < grad.length; i++) {
      errorBuf[i] = 0;
    }

    // Pack: [mode(4)] [15 x Float32]
    const totalBytes = 4 + grad.length * 4;
    packedBuffer = Buffer.alloc(totalBytes);
    packedBuffer.writeUInt32LE(COMPRESS_NONE, 0);
    for (let i = 0; i < grad.length; i++) {
      packedBuffer.writeFloatLE(corrected[i], 4 + i * 4);
    }
  }

  return packedBuffer;
}


// ===========================================================================
//  SimulatedWorker
// ===========================================================================
class SimulatedWorker {
  constructor(id, name, color) {
    this.id = id;
    this.name = name;
    this.color = color;
    this.ws = null;
    this.alive = true;

    // Per-worker stats
    this.stats = {
      connected: false,
      batchesReceived: 0,
      gradientsSent: 0,
      faultsInjected: 0,
      poisonedSent: 0,
      reconnections: 0,
      latencies: [],           // last N processing times in ms
      totalProcessingMs: 0,
      lastBatchSize: 0,
      lastLatencyMs: 0,
      lastGradNorm: 0,
      bytesSent: 0,
    };

    // Error feedback buffer (persists across rounds)
    this.errorBuffer = new Float32Array(GRAD_DIM);  // initialized to 0

    // RDP Accountant (one per worker — local DP)
    this.rdpAccountant = (DP_CLIP_C > 0 && DP_SIGMA > 0)
      ? new RDPAccountant(DP_CLIP_C, DP_SIGMA, parseFloat(DP_DELTA))
      : null;

    // Reconnection state
    this._backoff = 1000;
    this._maxBackoff = 15000;
    this._reconnectTimer = null;

    this.connect();
  }

  // --- Logging -----------------------------------------------------------
  log(msg) {
    const ts = new Date().toISOString().slice(11, 23);
    console.log(`${C.gray}${ts}${C.reset}  ${this.color}[${this.name}]${C.reset}  ${msg}`);
  }

  // --- Connection --------------------------------------------------------
  connect() {
    if (!this.alive) return;

    try {
      this.ws = new WebSocket(TARGET_URL);
    } catch (err) {
      this.log(`${C.red}Connection init error: ${err.message}${C.reset}`);
      this._scheduleReconnect();
      return;
    }

    this.ws.on("open", () => {
      this.stats.connected = true;
      this._backoff = 1000; // reset backoff
      this.log(`${C.green}Connected${C.reset}`);
      if (JOB_ID) {
        this.ws.send(JSON.stringify({
          type: "JOIN_JOB",
          jobId: JOB_ID,
          workerName: this.name
        }));
      }
    });

    this.ws.on("message", (data) => {
      if (Buffer.isBuffer(data)) {
        this._handleBinaryPayload(data);
      }
    });

    this.ws.on("close", () => {
      this.stats.connected = false;
      if (this.alive) {
        this.log(`${C.yellow}Disconnected${C.reset}`);
        this._scheduleReconnect();
      }
    });

    this.ws.on("error", () => {
      // swallow — close will fire next
    });
  }

  _scheduleReconnect() {
    if (!this.alive) return;
    this.stats.reconnections++;
    totalReconnections++;
    const jitter = Math.floor(Math.random() * 500);
    const delay = Math.min(this._backoff + jitter, this._maxBackoff);
    this._backoff = Math.min(this._backoff * 1.5, this._maxBackoff);
    this.log(`${C.dim}Reconnecting in ${(delay / 1000).toFixed(1)}s...${C.reset}`);
    this._reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  // --- Binary payload processing -----------------------------------------
  _handleBinaryPayload(buffer) {
    this.stats.batchesReceived++;
    totalBatches++;

    // Read batch size + weights from the coordinator's packed format
    const batchSize = buffer.readUInt32LE(0);
    this.stats.lastBatchSize = batchSize;

    // Extract W (12 floats) and B (3 floats) starting at byte 4
    const W_global = new Float32Array(buffer.buffer, buffer.byteOffset + 4, 12);
    const B_global = new Float32Array(buffer.buffer, buffer.byteOffset + 52, 3);

    // Extract X (B*4 floats) and Y (B*3 floats)
    const X = new Float32Array(buffer.buffer, buffer.byteOffset + 64, batchSize * 4);
    const Y = new Float32Array(buffer.buffer, buffer.byteOffset + 64 + batchSize * 16, batchSize * 3);

    // --- Fault injection: random drop ---
    if (FAULT_RATE > 0 && Math.random() < FAULT_RATE) {
      this.stats.faultsInjected++;
      totalFaults++;
      this.log(`${C.red}FAULT: Dropping response (simulated failure)${C.reset}`);
      return;
    }

    // Simulate variable compute latency
    const delay = MIN_DELAY_MS + Math.floor(Math.random() * (MAX_DELAY_MS - MIN_DELAY_MS));

    setTimeout(() => {
      const t0 = performance.now();

      // ════════════════════════════════════════════════════════════════════
      //  FedProx: Multi-Step Local Training with Proximal Term
      //
      //  For each local step:
      //    L = L_local + (μ/2) * ||W_local - W_global||²
      //    ∇L = ∇L_local + μ * (W_local - W_global)
      //    W_local ← W_local - α * ∇L
      //
      //  Final gradient sent = W_global - W_local  (the total weight delta)
      // ════════════════════════════════════════════════════════════════════

      // Make mutable copies for local training
      const W_local = new Float32Array(W_global);
      const B_local = new Float32Array(B_global);

      let loss = 0;

      for (let step = 0; step < LOCAL_STEPS; step++) {
        const dW = new Float32Array(12);
        const dB = new Float32Array(3);
        loss = 0;

        for (let b = 0; b < batchSize; b++) {
          // Forward: logits = X[b] @ W_local + B_local
          const logits = new Float32Array(3);
          for (let j = 0; j < 3; j++) {
            logits[j] = B_local[j];
            for (let i = 0; i < 4; i++) {
              logits[j] += X[b * 4 + i] * W_local[i * 3 + j];
            }
          }

          // Softmax
          let maxLogit = -Infinity;
          for (let j = 0; j < 3; j++) maxLogit = Math.max(maxLogit, logits[j]);
          let sumExp = 0;
          const probs = new Float32Array(3);
          for (let j = 0; j < 3; j++) {
            probs[j] = Math.exp(logits[j] - maxLogit);
            sumExp += probs[j];
          }
          for (let j = 0; j < 3; j++) probs[j] /= sumExp;

          // Cross-entropy loss
          for (let j = 0; j < 3; j++) {
            if (Y[b * 3 + j] > 0) {
              loss -= Math.log(Math.max(probs[j], 1e-7));
            }
          }

          // dZ = probs - Y_true
          for (let j = 0; j < 3; j++) {
            const dz = probs[j] - Y[b * 3 + j];
            dB[j] += dz;
            for (let i = 0; i < 4; i++) {
              dW[i * 3 + j] += X[b * 4 + i] * dz;
            }
          }
        }

        loss /= batchSize;

        // Normalize by batch size and apply learning rate
        const alpha = 0.1;
        for (let i = 0; i < 12; i++) {
          let grad = (alpha * dW[i]) / batchSize;

          // FedProx proximal term: + μ * (W_local[i] - W_global[i])
          if (FEDPROX_MU > 0) {
            grad += FEDPROX_MU * (W_local[i] - W_global[i]);
          }

          W_local[i] -= grad;
        }
        for (let j = 0; j < 3; j++) {
          let grad = (alpha * dB[j]) / batchSize;

          if (FEDPROX_MU > 0) {
            grad += FEDPROX_MU * (B_local[j] - B_global[j]);
          }

          B_local[j] -= grad;
        }
      }

      // ════════════════════════════════════════════════════════════════════
      //  Compute the gradient to send: ΔW = W_global - W_local
      //  (positive delta means "subtract this from W_global to get W_local")
      // ════════════════════════════════════════════════════════════════════
      const flatGrad = new Float32Array(GRAD_DIM);
      for (let i = 0; i < 12; i++) {
        flatGrad[i] = W_global[i] - W_local[i];
      }
      for (let j = 0; j < 3; j++) {
        flatGrad[12 + j] = B_global[j] - B_local[j];
      }

      // ════════════════════════════════════════════════════════════════════
      //  Poison injection: send out-of-range gradients
      // ════════════════════════════════════════════════════════════════════
      let poisoned = false;
      if (POISON_RATE > 0 && Math.random() < POISON_RATE) {
        this.stats.poisonedSent++;
        totalPoisoned++;
        poisoned = true;
        for (let i = 0; i < GRAD_DIM; i++) {
          flatGrad[i] = (Math.random() - 0.5) * 10;
        }
        this.log(`${C.red}${C.bold}POISON: Sending malicious gradients${C.reset}`);
      }

      // ════════════════════════════════════════════════════════════════════
      //  Differential Privacy: L2 Clipping + Gaussian Noise
      // ════════════════════════════════════════════════════════════════════
      let gradNorm = 0;
      for (let i = 0; i < GRAD_DIM; i++) gradNorm += flatGrad[i] * flatGrad[i];
      gradNorm = Math.sqrt(gradNorm);

      if (DP_CLIP_C > 0) {
        clipGradientL2(flatGrad, DP_CLIP_C);
      }
      if (DP_SIGMA > 0) {
        addGaussianNoise(flatGrad, DP_SIGMA);
        if (this.rdpAccountant) {
          this.rdpAccountant.step();
        }
      }

      // ════════════════════════════════════════════════════════════════════
      //  Compression + Error Feedback → Binary Packing
      // ════════════════════════════════════════════════════════════════════
      let responseBuffer;
      const useCompression = (TOP_K_FRAC > 0 && TOP_K_FRAC < 1) || QUANTIZE === "int8";

      if (useCompression) {
        responseBuffer = compressWithErrorFeedback(
          flatGrad, this.errorBuffer, TOP_K_FRAC, QUANTIZE
        );
      } else {
        // Legacy 60-byte uncompressed format (no header, backward compat)
        responseBuffer = Buffer.alloc(60);
        for (let i = 0; i < 12; i++) responseBuffer.writeFloatLE(flatGrad[i], i * 4);
        for (let j = 0; j < 3; j++) responseBuffer.writeFloatLE(flatGrad[12 + j], 48 + j * 4);
      }

      const elapsed = performance.now() - t0;
      const totalLatency = delay + elapsed;

      try {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(responseBuffer);
          this.stats.gradientsSent++;
          totalGradientsSent++;
          this.stats.bytesSent += responseBuffer.length;
          this.stats.totalProcessingMs += totalLatency;
          this.stats.lastLatencyMs = totalLatency;
          this.stats.lastGradNorm = gradNorm;
          this.stats.latencies.push(totalLatency);
          if (this.stats.latencies.length > 50) this.stats.latencies.shift();

          // Build log line
          let logParts = [
            `${C.green}OK${C.reset}`,
            `batch=${batchSize}`,
            `loss=${C.bold}${loss.toFixed(4)}${C.reset}`,
            `|grad|=${gradNorm.toFixed(4)}`,
            `${C.dim}${totalLatency.toFixed(0)}ms${C.reset}`,
            `${responseBuffer.length}B`,
          ];

          if (LOCAL_STEPS > 1) {
            logParts.push(`steps=${LOCAL_STEPS}`);
          }
          if (FEDPROX_MU > 0) {
            logParts.push(`μ=${FEDPROX_MU}`);
          }
          if (poisoned) {
            logParts.push(`${C.red}POISONED${C.reset}`);
          }
          if (this.rdpAccountant && this.rdpAccountant.steps % 10 === 0) {
            const privacy = this.rdpAccountant.getPrivacySpent();
            logParts.push(`${C.yellow}ε=${privacy.epsilon.toFixed(2)}${C.reset}`);
          }

          this.log(logParts.join("  "));
        }
      } catch (err) {
        this.log(`${C.red}Send failed: ${err.message}${C.reset}`);
      }
    }, delay);
  }

  // --- Shutdown -----------------------------------------------------------
  destroy() {
    this.alive = false;
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    if (this.ws) {
      try { this.ws.close(); } catch {}
    }
  }
}

// ===========================================================================
//  Live Summary Dashboard
// ===========================================================================
let summaryTimer = null;

function printSummary() {
  const uptime = ((Date.now() - startTime) / 1000).toFixed(0);
  const throughput = totalGradientsSent > 0
    ? (totalGradientsSent / ((Date.now() - startTime) / 1000)).toFixed(2)
    : "0.00";

  const lines = [
    "",
    `${C.bgBlue}${C.white}${C.bold} FEDLEARN SIMULATOR DASHBOARD ${C.reset}  ${C.dim}uptime ${uptime}s${C.reset}`,
    `${C.dim}${"─".repeat(70)}${C.reset}`,
  ];

  // Feature status line
  const features = [];
  if (DP_CLIP_C > 0)   features.push(`DP(C=${DP_CLIP_C},σ=${DP_SIGMA})`);
  if (TOP_K_FRAC > 0)  features.push(`Top-K(${(TOP_K_FRAC * 100).toFixed(1)}%)`);
  if (QUANTIZE !== "none") features.push(`Quant(${QUANTIZE})`);
  if (FEDPROX_MU > 0)  features.push(`FedProx(μ=${FEDPROX_MU})`);
  if (LOCAL_STEPS > 1) features.push(`LocalSteps=${LOCAL_STEPS}`);
  if (features.length > 0) {
    lines.push(`  ${C.cyan}Features: ${features.join(" | ")}${C.reset}`);
    lines.push(`${C.dim}${"─".repeat(70)}${C.reset}`);
  }

  for (const w of workers) {
    const s = w.stats;
    const status = s.connected
      ? `${C.green}ONLINE${C.reset} `
      : `${C.red}OFFLINE${C.reset}`;

    const avgLatency = s.latencies.length > 0
      ? (s.latencies.reduce((a, b) => a + b, 0) / s.latencies.length).toFixed(0)
      : "-";

    let workerLine =
      `  ${w.color}${w.name}${C.reset}  ${status}  ` +
      `batches: ${C.bold}${s.batchesReceived}${C.reset}  ` +
      `sent: ${C.bold}${s.gradientsSent}${C.reset}  ` +
      `avg: ${avgLatency}ms  ` +
      `|grad|: ${s.lastGradNorm.toFixed(3)}  ` +
      `bytes: ${s.bytesSent}`;

    // Show DP budget
    if (w.rdpAccountant) {
      const p = w.rdpAccountant.getPrivacySpent();
      workerLine += `  ${C.yellow}ε=${p.epsilon.toFixed(2)}${C.reset}`;
    }

    lines.push(workerLine);
  }

  lines.push(`${C.dim}${"─".repeat(70)}${C.reset}`);
  lines.push(
    `  ${C.bold}Total${C.reset}  ` +
    `batches: ${totalBatches}  ` +
    `grads: ${totalGradientsSent}  ` +
    `throughput: ${C.bold}${throughput}${C.reset} grads/s  ` +
    `faults: ${totalFaults}  ` +
    `poison: ${totalPoisoned}`
  );
  lines.push("");

  console.log(lines.join("\n"));
}

// ===========================================================================
//  Startup
// ===========================================================================
const featureList = [];
if (DP_CLIP_C > 0)          featureList.push(`Differential Privacy (C=${DP_CLIP_C}, σ=${DP_SIGMA}, δ=${DP_DELTA})`);
if (TOP_K_FRAC > 0)         featureList.push(`Top-K Sparsification (${(TOP_K_FRAC * 100).toFixed(1)}%)`);
if (QUANTIZE !== "none")     featureList.push(`Quantization (${QUANTIZE})`);
if (FEDPROX_MU > 0)         featureList.push(`FedProx (μ=${FEDPROX_MU})`);
if (LOCAL_STEPS > 1)        featureList.push(`Local Steps = ${LOCAL_STEPS}`);

console.log(`
${C.bold}${C.cyan}╔══════════════════════════════════════════════════════════════╗
║          FedLearn Worker Simulator v3.0 (Enhanced)           ║
╚══════════════════════════════════════════════════════════════╝${C.reset}

  ${C.dim}Target${C.reset}       ${TARGET_URL}
  ${C.dim}Workers${C.reset}      ${NUM_WORKERS}
  ${C.dim}Latency${C.reset}      ${MIN_DELAY_MS}–${MAX_DELAY_MS}ms  (simulated compute)
  ${C.dim}Fault rate${C.reset}   ${(FAULT_RATE * 100).toFixed(0)}%
  ${C.dim}Poison rate${C.reset}  ${(POISON_RATE * 100).toFixed(0)}%
  ${C.dim}CPU cores${C.reset}    ${os.cpus().length}
  ${C.dim}Platform${C.reset}     ${os.platform()} ${os.arch()}
${featureList.length > 0 ? `\n  ${C.bold}${C.green}Active Enhancements:${C.reset}\n${featureList.map(f => `    ✓ ${f}`).join("\n")}\n` : ""}
`);

const workers = [];
for (let i = 0; i < NUM_WORKERS; i++) {
  const color = WORKER_COLORS[i % WORKER_COLORS.length];
  workers.push(new SimulatedWorker(i, `Worker-${i + 1}`, color));
}

// Print summary every 10 seconds
summaryTimer = setInterval(printSummary, 10_000);

// ===========================================================================
//  Graceful Shutdown
// ===========================================================================
function shutdown() {
  console.log(`\n${C.yellow}${C.bold}Shutting down...${C.reset}`);
  if (summaryTimer) clearInterval(summaryTimer);

  for (const w of workers) w.destroy();

  // Print final report
  const uptime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`
${C.bold}${C.cyan}╔══════════════════════════════════════════════════════════════╗
║               Final Simulation Report                        ║
╚══════════════════════════════════════════════════════════════╝${C.reset}

  ${C.dim}Duration${C.reset}          ${uptime}s
  ${C.dim}Total Batches${C.reset}     ${totalBatches}
  ${C.dim}Gradients Sent${C.reset}    ${totalGradientsSent}
  ${C.dim}Throughput${C.reset}        ${(totalGradientsSent / (parseFloat(uptime) || 1)).toFixed(2)} grads/s
  ${C.dim}Faults Injected${C.reset}   ${totalFaults}
  ${C.dim}Poison Injected${C.reset}   ${totalPoisoned}
  ${C.dim}Reconnections${C.reset}     ${totalReconnections}
`);

  for (const w of workers) {
    const s = w.stats;
    const avg = s.latencies.length > 0
      ? (s.latencies.reduce((a, b) => a + b, 0) / s.latencies.length).toFixed(1)
      : "-";
    const p95 = s.latencies.length >= 2
      ? s.latencies.slice().sort((a, b) => a - b)[Math.floor(s.latencies.length * 0.95)].toFixed(1)
      : "-";

    let line =
      `  ${w.color}${w.name}${C.reset}  ` +
      `batches: ${s.batchesReceived}  sent: ${s.gradientsSent}  ` +
      `avg: ${avg}ms  p95: ${p95}ms  bytes: ${s.bytesSent}  reconn: ${s.reconnections}`;

    // Final DP budget
    if (w.rdpAccountant) {
      const p = w.rdpAccountant.getPrivacySpent();
      line += `  ${C.yellow}ε=${p.epsilon.toFixed(4)} (δ=${p.delta})${C.reset}`;
    }

    console.log(line);
  }

  console.log("");
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
