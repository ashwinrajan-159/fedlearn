import asyncio
import websockets
import numpy as np
import struct
import json
import uuid
import math

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
HOST = "localhost"
PORT = 8080
LR = 0.1               # Alpha (Learning Rate)
BATCH_SIZE = 16
NUM_FEATURES = 4
NUM_CLASSES = 3

# --- Aggregation Buffer ---
# Collect K gradients before performing robust aggregation.
# This enables Byzantine-robust aggregation (geometric median needs multiple inputs).
BUFFER_K = 5

# --- Byzantine Fault Tolerance ---
# Assumed attacker fraction: f < 0.5  (geometric median breakdown point).
# Weiszfeld iterations and convergence tolerance:
WEISZFELD_MAX_ITER = 100
WEISZFELD_TOL = 1e-6
WEISZFELD_SMOOTHING = 1e-7   # η to avoid division by zero

# --- Anti-Gaming (Incentive Verification) ---
# Minimum L2 norm for a gradient to be considered non-trivial:
MIN_GRAD_NORM = 1e-6
# Minimum cosine similarity between worker gradient and aggregated update:
MIN_COSINE_SIMILARITY = 0.0

# --- Evaluation / Early Stopping ---
EARLY_STOP_PATIENCE = 20      # rounds without improvement before stopping
EARLY_STOP_MIN_DELTA = 1e-4   # minimum loss decrease to count as improvement

# ═══════════════════════════════════════════════════════════════════════════════
#  Global Model State
# ═══════════════════════════════════════════════════════════════════════════════
global_W = np.zeros((NUM_FEATURES, NUM_CLASSES), dtype=np.float32)
global_B = np.zeros(NUM_CLASSES, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — Train/Validation Split (80/20)
# ═══════════════════════════════════════════════════════════════════════════════
NUM_SAMPLES = 100
_all_X = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
_all_y_indices = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
_all_Y = np.zeros((NUM_SAMPLES, NUM_CLASSES), dtype=np.float32)
_all_Y[np.arange(NUM_SAMPLES), _all_y_indices] = 1.0

TRAIN_SIZE = 80
VAL_SIZE = NUM_SAMPLES - TRAIN_SIZE

train_X = _all_X[:TRAIN_SIZE]
train_Y = _all_Y[:TRAIN_SIZE]
val_X   = _all_X[TRAIN_SIZE:]
val_Y   = _all_Y[TRAIN_SIZE:]
val_y_indices = _all_y_indices[TRAIN_SIZE:]

# ═══════════════════════════════════════════════════════════════════════════════
#  Server State
# ═══════════════════════════════════════════════════════════════════════════════
connected_workers = {}
dashboard_clients = set()
current_round = 0
round_in_progress = False

# Gradient buffer: list of (worker_id, delta_W, delta_B) tuples
gradient_buffer = []

# Evaluation history
eval_history = []          # list of { round, loss, accuracy }
best_val_loss = float("inf")
patience_counter = 0
training_stopped = False

# Per-worker utility tracking
worker_utilities = {}      # worker_id → list of recent cosine similarities

# ═══════════════════════════════════════════════════════════════════════════════
#  Compression Constants (must match worker protocol)
# ═══════════════════════════════════════════════════════════════════════════════
COMPRESS_NONE     = 0
COMPRESS_INT8     = 1
COMPRESS_TOPK     = 2
COMPRESS_TOPK_INT8 = 3

GRADIENT_DIM = NUM_FEATURES * NUM_CLASSES + NUM_CLASSES   # 12 + 3 = 15

# ═══════════════════════════════════════════════════════════════════════════════
#  Softmax Utility
# ═══════════════════════════════════════════════════════════════════════════════
def softmax_rows(logits):
    """Numerically stable softmax over last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FL Logic — Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_random_batch():
    indices = np.random.choice(TRAIN_SIZE, BATCH_SIZE, replace=False)
    return train_X[indices], train_Y[indices]

def pack_payload(batch_x, batch_y):
    """
    Packs the global weights and a batch of data into a binary buffer.
    Format:
    [0-4 bytes]: batch_size (UInt32)
    [4-52 bytes]: W (12 x Float32)
    [52-64 bytes]: B (3 x Float32)
    [64 - 64+B*16 bytes]: X (B*4 x Float32)
    [64+B*16 - end]: Y (B*3 x Float32)
    """
    buffer = bytearray()
    # Batch size
    buffer.extend(struct.pack('<I', BATCH_SIZE))
    # W (flattened)
    buffer.extend(global_W.flatten().tobytes())
    # B
    buffer.extend(global_B.tobytes())
    # X
    buffer.extend(batch_x.flatten().tobytes())
    # Y
    buffer.extend(batch_y.flatten().tobytes())
    return buffer

# ═══════════════════════════════════════════════════════════════════════════════
#  Server-Side Decompression
# ═══════════════════════════════════════════════════════════════════════════════

def unpack_gradients(data):
    """
    Unpacks gradients from a worker, supporting multiple compression modes.

    Wire format (first 4 bytes = header):
      [0-4]:  UInt32 LE  →  compression_mode

    Mode 0 (NONE):
      [4-64]:  15 × Float32 (12 dW + 3 dB)                  = 60 bytes payload

    Mode 1 (INT8):
      [4-8]:   Float32 scale factor
      [8-23]:  15 × Int8 quantized values                    = 19 bytes payload

    Mode 2 (TOP-K sparse, Float32 values):
      [4-8]:   UInt32 LE  nnz (number of non-zero entries)
      [8 .. 8+2*nnz]:  nnz × Uint16 LE indices
      [8+2*nnz .. 8+2*nnz+4*nnz]:  nnz × Float32 values

    Mode 3 (TOP-K sparse + INT8 quantized values):
      [4-8]:   UInt32 LE  nnz
      [8-12]:  Float32 scale factor
      [12 .. 12+2*nnz]:  nnz × Uint16 LE indices
      [12+2*nnz .. 12+2*nnz+nnz]:  nnz × Int8 values

    Legacy (no header — exactly 60 bytes):
      Same as Mode 0 without the 4-byte header.
    """
    length = len(data)

    # ── Legacy uncompressed format (backward compat) ──────────────────────
    if length == 60:
        flat = np.frombuffer(data, dtype=np.float32)   # 15 floats
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    # ── New format: read 4-byte header ────────────────────────────────────
    if length < 4:
        raise ValueError(f"Payload too short: {length} bytes")

    mode = struct.unpack_from('<I', data, 0)[0]

    if mode == COMPRESS_NONE:
        # 4 + 60 = 64 bytes
        if length != 64:
            raise ValueError(f"Mode 0 expects 64 bytes, got {length}")
        flat = np.frombuffer(data, dtype=np.float32, offset=4, count=15)
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_INT8:
        # 4 (header) + 4 (scale) + 15 (int8) = 23 bytes
        if length != 23:
            raise ValueError(f"Mode 1 (INT8) expects 23 bytes, got {length}")
        scale = struct.unpack_from('<f', data, 4)[0]
        quantized = np.frombuffer(data, dtype=np.int8, offset=8, count=15)
        flat = quantized.astype(np.float32) * (scale / 127.0)
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_TOPK:
        # 4 (header) + 4 (nnz) + 2*nnz (indices) + 4*nnz (f32 values)
        if length < 8:
            raise ValueError(f"Mode 2 (Top-K) too short: {length}")
        nnz = struct.unpack_from('<I', data, 4)[0]
        expected = 8 + 2 * nnz + 4 * nnz
        if length != expected:
            raise ValueError(f"Mode 2 expects {expected} bytes, got {length}")
        indices = np.frombuffer(data, dtype=np.uint16, offset=8, count=nnz)
        values  = np.frombuffer(data, dtype=np.float32, offset=8 + 2 * nnz, count=nnz)
        flat = np.zeros(GRADIENT_DIM, dtype=np.float32)
        for idx, val in zip(indices, values):
            if idx < GRADIENT_DIM:
                flat[idx] = val
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_TOPK_INT8:
        # 4 (header) + 4 (nnz) + 4 (scale) + 2*nnz (indices) + nnz (int8 values)
        if length < 12:
            raise ValueError(f"Mode 3 (Top-K+INT8) too short: {length}")
        nnz   = struct.unpack_from('<I', data, 4)[0]
        scale = struct.unpack_from('<f', data, 8)[0]
        expected = 12 + 2 * nnz + nnz
        if length != expected:
            raise ValueError(f"Mode 3 expects {expected} bytes, got {length}")
        indices = np.frombuffer(data, dtype=np.uint16, offset=12, count=nnz)
        qvals   = np.frombuffer(data, dtype=np.int8, offset=12 + 2 * nnz, count=nnz)
        flat = np.zeros(GRADIENT_DIM, dtype=np.float32)
        for idx, q in zip(indices, qvals):
            if idx < GRADIENT_DIM:
                flat[idx] = float(q) * (scale / 127.0)
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Byzantine Fault Tolerance — Weiszfeld Geometric Median
# ═══════════════════════════════════════════════════════════════════════════════

def geometric_median(vectors, max_iter=WEISZFELD_MAX_ITER, tol=WEISZFELD_TOL,
                     eta=WEISZFELD_SMOOTHING):
    """
    Compute the geometric median of a set of vectors using Weiszfeld's algorithm.

    Given m vectors {g_1, ..., g_m} in R^d, the geometric median is:
        argmin_y  Σ_i ||y - g_i||_2

    Weiszfeld update:
        y^{t+1} = (Σ_i  g_i / ||y^t - g_i||) / (Σ_i  1 / ||y^t - g_i||)

    We use a smoothing constant η to avoid division by zero.

    Breakdown point: 0.5 — tolerates up to 50% Byzantine inputs.

    Parameters
    ----------
    vectors : list of np.ndarray, each shape (d,)
    max_iter : int
    tol : float   — convergence threshold on ||y^{t+1} - y^t||
    eta : float   — smoothing constant

    Returns
    -------
    np.ndarray, shape (d,)
    """
    vectors = [v.astype(np.float64) for v in vectors]
    m = len(vectors)
    if m == 0:
        raise ValueError("Cannot compute geometric median of 0 vectors")
    if m == 1:
        return vectors[0].astype(np.float32)

    d = vectors[0].shape[0]

    # Initialize with the component-wise median (a good starting point)
    stacked = np.stack(vectors, axis=0)   # (m, d)
    y = np.median(stacked, axis=0)        # (d,)

    for iteration in range(max_iter):
        # Compute weights  w_i = 1 / max(||y - g_i||, eta)
        distances = np.array([np.linalg.norm(y - g) for g in vectors])
        weights = 1.0 / np.maximum(distances, eta)

        # Weighted average
        total_weight = weights.sum()
        y_new = np.zeros(d, dtype=np.float64)
        for i in range(m):
            y_new += weights[i] * vectors[i]
        y_new /= total_weight

        # Check convergence
        shift = np.linalg.norm(y_new - y)
        y = y_new
        if shift < tol:
            break

    return y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Anti-Gaming — Cosine Similarity Utility
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Cosine similarity between two flat vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════════
#  Federated Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_global_model():
    """
    Compute validation loss and accuracy on the held-out validation set.

    Returns
    -------
    (loss: float, accuracy: float)
    """
    # Forward pass: logits = X @ W + B
    logits = val_X @ global_W + global_B       # (VAL_SIZE, NUM_CLASSES)
    probs = softmax_rows(logits)

    # Cross-entropy loss
    # L = -1/N * Σ Σ y_ij * log(p_ij)
    loss = -np.sum(val_Y * np.log(np.maximum(probs, 1e-7))) / VAL_SIZE

    # Accuracy
    pred_indices = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred_indices == val_y_indices))

    return float(loss), accuracy


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradient Handling — Buffered + Robust Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def handle_gradient(worker_id, data):
    """
    Receive a gradient from a worker and add it to the buffer.
    When the buffer reaches K entries, perform robust aggregation.
    """
    global global_W, global_B, best_val_loss, patience_counter, training_stopped

    if training_stopped:
        return

    try:
        delta_W, delta_B = unpack_gradients(data)
    except Exception as e:
        print(f"[!] Error unpacking gradient from {worker_id}: {e}")
        return

    # ── Basic validation ──────────────────────────────────────────────────
    if np.isnan(delta_W).any() or np.isnan(delta_B).any():
        print(f"[!] NaN gradient from {worker_id} — discarded.")
        return
    if np.isinf(delta_W).any() or np.isinf(delta_B).any():
        print(f"[!] Inf gradient from {worker_id} — discarded.")
        return

    # ── Reject trivial/zero gradients (anti-gaming) ───────────────────────
    flat_grad = np.concatenate([delta_W.flatten(), delta_B.flatten()])
    grad_norm = np.linalg.norm(flat_grad)
    if grad_norm < MIN_GRAD_NORM:
        print(f"[!] Trivial gradient from {worker_id} (||g||={grad_norm:.2e}) — rejected.")
        return

    # ── Add to buffer ─────────────────────────────────────────────────────
    gradient_buffer.append((worker_id, delta_W.copy(), delta_B.copy()))
    print(f"[OK] Buffered gradient from {worker_id} (||g||={grad_norm:.4f}) "
          f"[{len(gradient_buffer)}/{BUFFER_K}]")

    # ── Check if buffer is full ───────────────────────────────────────────
    if len(gradient_buffer) < BUFFER_K:
        return

    # ══════════════════════════════════════════════════════════════════════
    #  AGGREGATION — Byzantine-Robust Geometric Median
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  AGGREGATION — {len(gradient_buffer)} gradients buffered")
    print(f"{'═'*60}")

    # Flatten each worker's gradient into a single vector for geometric median
    worker_ids = []
    flat_grads = []
    for wid, dw, db in gradient_buffer:
        worker_ids.append(wid)
        flat_grads.append(np.concatenate([dw.flatten(), db.flatten()]))

    # Compute geometric median (robust against <= 49% Byzantine workers)
    aggregated = geometric_median(flat_grads)

    # ── Anti-gaming: compute cosine similarity per worker ────────────────
    print(f"  Worker utility (cosine similarity with aggregated update):")
    accepted_count = 0
    rejected_workers = []
    for i, (wid, flat_g) in enumerate(zip(worker_ids, flat_grads)):
        utility = cosine_similarity(flat_g, aggregated)

        # Track utility history
        if wid not in worker_utilities:
            worker_utilities[wid] = []
        worker_utilities[wid].append(utility)
        if len(worker_utilities[wid]) > 50:
            worker_utilities[wid].pop(0)

        status = "OK" if utility >= MIN_COSINE_SIMILARITY else "REJECTED"
        print(f"    {wid[:8]}  cos={utility:+.4f}  {status}")

        if utility < MIN_COSINE_SIMILARITY:
            rejected_workers.append(wid)
        else:
            accepted_count += 1

    if rejected_workers:
        print(f"  [!] {len(rejected_workers)} worker(s) rejected (adversarial/gaming).")

    # ── Apply aggregated update ──────────────────────────────────────────
    agg_W = aggregated[:12].reshape((NUM_FEATURES, NUM_CLASSES))
    agg_B = aggregated[12:]

    global_W -= LR * agg_W
    global_B -= LR * agg_B

    print(f"  Applied robust aggregated update (||delta||={np.linalg.norm(aggregated):.4f})")

    # ── Clear buffer ─────────────────────────────────────────────────────
    gradient_buffer.clear()

    # ══════════════════════════════════════════════════════════════════════
    #  EVALUATION — Validation Loss + Accuracy
    # ══════════════════════════════════════════════════════════════════════
    val_loss, val_acc = evaluate_global_model()
    eval_history.append({
        "round": current_round,
        "loss": val_loss,
        "accuracy": val_acc,
    })

    print(f"  Validation -- loss: {val_loss:.4f}  accuracy: {val_acc:.2%}")

    # ── Early stopping ───────────────────────────────────────────────────
    if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if val_acc >= 1.0:
        print(f"  [*] Perfect validation accuracy -- stopping training.")
        training_stopped = True
    elif patience_counter >= EARLY_STOP_PATIENCE:
        print(f"  [STOP] Early stopping triggered (no improvement for "
              f"{EARLY_STOP_PATIENCE} aggregation rounds).")
        training_stopped = True

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Handlers
# ═══════════════════════════════════════════════════════════════════════════════

async def broadcast_dashboard():
    if not dashboard_clients:
        return

    last_eval = eval_history[-1] if eval_history else None
    msg = json.dumps({
        "type": "METRICS",
        "round": current_round,
        "workers": len(connected_workers),
        "status": "STOPPED" if training_stopped else ("RUNNING" if round_in_progress else "IDLE"),
        "bufferFill": len(gradient_buffer),
        "bufferK": BUFFER_K,
        "valLoss": last_eval["loss"] if last_eval else None,
        "valAccuracy": last_eval["accuracy"] if last_eval else None,
        "aggregations": len(eval_history),
    })
    for client in dashboard_clients:
        try:
            await client.send(msg)
        except:
            pass

async def handle_dashboard(websocket):
    dashboard_clients.add(websocket)
    try:
        await broadcast_dashboard()
        async for _ in websocket:
            pass  # Keep alive
    finally:
        dashboard_clients.remove(websocket)

async def handle_worker(websocket):
    worker_id = str(uuid.uuid4())
    connected_workers[worker_id] = {"websocket": websocket, "status": "IDLE"}
    print(f"[+] Worker connected: {worker_id}")
    await broadcast_dashboard()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Gradient payload received
                handle_gradient(worker_id, message)
                await broadcast_dashboard()
    finally:
        print(f"[-] Worker disconnected: {worker_id}")
        del connected_workers[worker_id]
        await broadcast_dashboard()

async def router(websocket, path=None):
    if path is None:
        path = getattr(websocket, "path", None)
        if path is None:
            request = getattr(websocket, "request", None)
            if request is not None:
                path = getattr(request, "path", "/")
            else:
                path = "/"

    if path == "/dashboard":
        await handle_dashboard(websocket)
    else:
        await handle_worker(websocket)

# ═══════════════════════════════════════════════════════════════════════════════
#  FL Coordinator Loop
# ═══════════════════════════════════════════════════════════════════════════════

async def fl_coordinator():
    global current_round, round_in_progress
    print("Coordinator started...")
    print(f"  Buffer K          : {BUFFER_K}")
    print(f"  BFT method        : Weiszfeld Geometric Median (f < 0.5)")
    print(f"  Anti-gaming       : cosine similarity >= {MIN_COSINE_SIMILARITY}")
    print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
    print(f"  Train/Val split   : {TRAIN_SIZE}/{VAL_SIZE}")
    print()

    while True:
        await asyncio.sleep(2)  # 2 seconds between rounds

        if training_stopped:
            await asyncio.sleep(10)
            continue

        idle_workers = [w for w in connected_workers.values() if w["status"] == "IDLE"]
        if not idle_workers:
            continue

        current_round += 1
        round_in_progress = True
        await broadcast_dashboard()

        print(f"\n--- Starting Round {current_round} ---")
        print(f"Selected {len(idle_workers)} workers.")

        # Dispatch tasks
        for worker in idle_workers:
            batch_x, batch_y = get_random_batch()
            payload = pack_payload(batch_x, batch_y)
            try:
                await worker["websocket"].send(payload)
                worker["status"] = "BUSY"
            except:
                pass

        # Wait a bit for them to process
        await asyncio.sleep(1)

        # Reset statuses
        for worker in idle_workers:
            worker["status"] = "IDLE"

        round_in_progress = False
        await broadcast_dashboard()

async def main():
    server = await websockets.serve(router, HOST, PORT)
    print(f"WebSocket Server listening on ws://{HOST}:{PORT}")

    # Run coordinator alongside server
    asyncio.create_task(fl_coordinator())

    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
