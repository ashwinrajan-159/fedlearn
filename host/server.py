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
# Jobs manager: job_id -> job_state
jobs = {}

# Per-worker utility tracking (global or per-job)
worker_utilities = {}      # worker_id -> list of recent cosine similarities

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

def pack_payload_for_job(job, batch_x, batch_y):
    """
    Packs the job's weights and a batch of data into a binary buffer.
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
    buffer.extend(job["W"].flatten().tobytes())
    # B
    buffer.extend(job["B"].tobytes())
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
        if length != 64:
            raise ValueError(f"Mode 0 expects 64 bytes, got {length}")
        flat = np.frombuffer(data, dtype=np.float32, offset=4, count=15)
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_INT8:
        if length != 23:
            raise ValueError(f"Mode 1 (INT8) expects 23 bytes, got {length}")
        scale = struct.unpack_from('<f', data, 4)[0]
        quantized = np.frombuffer(data, dtype=np.int8, offset=8, count=15)
        flat = quantized.astype(np.float32) * (scale / 127.0)
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_TOPK:
        if length < 8:
            raise ValueError(f"Mode 2 (Top-K) too short: {length}")
        nnz = struct.unpack_from('<I', data, 4)[0]
        expected = 8 + 4 * nnz + 4 * nnz
        if length != expected:
            raise ValueError(f"Mode 2 expects {expected} bytes, got {length}")
        indices = np.frombuffer(data, dtype=np.uint32, offset=8, count=nnz)
        values  = np.frombuffer(data, dtype=np.float32, offset=8 + 4 * nnz, count=nnz)
        flat = np.zeros(GRADIENT_DIM, dtype=np.float32)
        for idx, val in zip(indices, values):
            if idx < GRADIENT_DIM:
                flat[idx] = val
        delta_W = flat[:12].reshape((NUM_FEATURES, NUM_CLASSES)).copy()
        delta_B = flat[12:].copy()
        return delta_W, delta_B

    elif mode == COMPRESS_TOPK_INT8:
        if length < 12:
            raise ValueError(f"Mode 3 (Top-K+INT8) too short: {length}")
        nnz   = struct.unpack_from('<I', data, 4)[0]
        scale = struct.unpack_from('<f', data, 8)[0]
        expected = 12 + 4 * nnz + nnz
        if length != expected:
            raise ValueError(f"Mode 3 expects {expected} bytes, got {length}")
        indices = np.frombuffer(data, dtype=np.uint32, offset=12, count=nnz)
        qvals   = np.frombuffer(data, dtype=np.int8, offset=12 + 4 * nnz, count=nnz)
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
        distances = np.array([np.linalg.norm(y - g) for g in vectors])
        weights = 1.0 / np.maximum(distances, eta)

        total_weight = weights.sum()
        y_new = np.zeros(d, dtype=np.float64)
        for i in range(m):
            y_new += weights[i] * vectors[i]
        y_new /= total_weight

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
#  Federated Evaluation Loop per Job
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_global_model_for_job(job):
    """
    Compute validation loss and accuracy on the held-out validation set.
    """
    # Forward pass: logits = X @ W + B
    logits = val_X @ job["W"] + job["B"]       # (VAL_SIZE, NUM_CLASSES)
    probs = softmax_rows(logits)

    # Cross-entropy loss
    loss = -np.sum(val_Y * np.log(np.maximum(probs, 1e-7))) / VAL_SIZE

    # Accuracy
    pred_indices = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred_indices == val_y_indices))

    return float(loss), accuracy


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradient Handling — Buffered + Robust Aggregation per Job
# ═══════════════════════════════════════════════════════════════════════════════

def handle_gradient_for_job(job_id, worker_id, data):
    """
    Receive a gradient from a worker and add it to the job's buffer.
    When the buffer reaches the job's minWorkers count, perform robust aggregation.
    """
    if job_id not in jobs:
        return
    job = jobs[job_id]
    config = job["config"]
    model_config = config["modelConfig"]
    
    buffer_k = int(model_config.get("minWorkers", 3))

    if job["training_stopped"]:
        return

    try:
        delta_W, delta_B = unpack_gradients(data)
    except Exception as e:
        print(f"[!] [Job {job_id}] Error unpacking gradient from {worker_id}: {e}")
        return

    # ── Basic validation ──────────────────────────────────────────────────
    if np.isnan(delta_W).any() or np.isnan(delta_B).any():
        print(f"[!] [Job {job_id}] NaN gradient from {worker_id} — discarded.")
        return
    if np.isinf(delta_W).any() or np.isinf(delta_B).any():
        print(f"[!] [Job {job_id}] Inf gradient from {worker_id} — discarded.")
        return

    # ── Reject trivial/zero gradients (anti-gaming) ───────────────────────
    flat_grad = np.concatenate([delta_W.flatten(), delta_B.flatten()])
    grad_norm = np.linalg.norm(flat_grad)
    if grad_norm < MIN_GRAD_NORM:
        print(f"[!] [Job {job_id}] Trivial gradient from {worker_id} (||g||={grad_norm:.2e}) — rejected.")
        return

    # ── Add to buffer ─────────────────────────────────────────────────────
    job["gradient_buffer"].append((worker_id, delta_W.copy(), delta_B.copy()))
    print(f"[OK] [Job {job_id}] Buffered gradient from {worker_id} (||g||={grad_norm:.4f}) "
          f"[{len(job['gradient_buffer'])}/{buffer_k}]")

    # ── Check if buffer is full ───────────────────────────────────────────
    if len(job["gradient_buffer"]) < buffer_k:
        return

    # ══════════════════════════════════════════════════════════════════════
    #  AGGREGATION — Byzantine-Robust Geometric Median
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  [Job {job_id}] AGGREGATION - {len(job['gradient_buffer'])} gradients buffered")
    print(f"{'='*60}")

    worker_ids = []
    flat_grads = []
    for wid, dw, db in job["gradient_buffer"]:
        worker_ids.append(wid)
        flat_grads.append(np.concatenate([dw.flatten(), db.flatten()]))

    # Compute geometric median
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
        print(f"  [!] [Job {job_id}] {len(rejected_workers)} worker(s) rejected (adversarial/gaming).")

    # ── Apply aggregated update ──────────────────────────────────────────
    agg_W = aggregated[:12].reshape((NUM_FEATURES, NUM_CLASSES))
    agg_B = aggregated[12:]

    job["W"] -= LR * agg_W
    job["B"] -= LR * agg_B

    print(f"  [Job {job_id}] Applied robust aggregated update (||delta||={np.linalg.norm(aggregated):.4f})")

    # ── Clear buffer ─────────────────────────────────────────────────────
    job["gradient_buffer"].clear()

    # ══════════════════════════════════════════════════════════════════════
    #  EVALUATION — Validation Loss + Accuracy
    # ══════════════════════════════════════════════════════════════════════
    val_loss, val_acc = evaluate_global_model_for_job(job)
    job["eval_history"].append({
        "round": job["current_round"],
        "loss": val_loss,
        "accuracy": val_acc,
    })

    # Track metrics in config
    config["metrics"]["loss"].append(val_loss)
    config["metrics"]["accuracy"].append(val_acc)

    print(f"  [Job {job_id}] Validation -- loss: {val_loss:.4f}  accuracy: {val_acc:.2%}")

    # ── Early stopping ───────────────────────────────────────────────────
    if val_loss < job["best_val_loss"] - EARLY_STOP_MIN_DELTA:
        job["best_val_loss"] = val_loss
        job["patience_counter"] = 0
    else:
        job["patience_counter"] += 1

    if val_acc >= 1.0:
        print(f"  [*] [Job {job_id}] Perfect validation accuracy -- stopping training.")
        job["training_stopped"] = True
    elif job["patience_counter"] >= EARLY_STOP_PATIENCE:
        print(f"  [STOP] [Job {job_id}] Early stopping triggered (no improvement for "
              f"{EARLY_STOP_PATIENCE} aggregation rounds).")
        job["training_stopped"] = True

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Telemetry / Broadcast handlers
# ═══════════════════════════════════════════════════════════════════════════════

async def broadcast_dashboard(job_id):
    if job_id not in jobs:
        return
    job = jobs[job_id]
    config = job["config"]
    model_config = config["modelConfig"]
    min_workers = int(model_config.get("minWorkers", 3))

    if not job["dashboard_sockets"]:
        return

    last_eval = job["eval_history"][-1] if job["eval_history"] else None
    
    msg = json.dumps({
        "type": "METRICS",
        "round": job["current_round"],
        "workers": len(job["active_worker_sockets"]),
        "status": "STOPPED" if job["training_stopped"] else ("RUNNING" if config["status"] == "running" else "WAITING"),
        "bufferFill": len(job["gradient_buffer"]),
        "bufferK": min_workers,
        "valLoss": last_eval["loss"] if last_eval else None,
        "valAccuracy": last_eval["accuracy"] if last_eval else None,
        "aggregations": len(job["eval_history"]),
    })

    for client in job["dashboard_sockets"]:
        try:
            await client.send(msg)
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Connection Handlers
# ═══════════════════════════════════════════════════════════════════════════════

async def router(websocket, path=None):
    if path is None:
        path = getattr(websocket, "path", None)
        if path is None:
            request = getattr(websocket, "request", None)
            if request is not None:
                path = getattr(request, "path", "/")
            else:
                path = "/"

    print(f"[WebSocket] Connected with path: {path}")

    if path.startswith("/dashboard"):
        # Expect path format /dashboard/<job_id>
        parts = path.split("/")
        job_id = parts[-1] if len(parts) > 2 else None
        
        if job_id and job_id in jobs:
            jobs[job_id]["dashboard_sockets"].add(websocket)
            try:
                await broadcast_dashboard(job_id)
                async for _ in websocket:
                    pass  # Keep alive
            finally:
                if job_id in jobs:
                    jobs[job_id]["dashboard_sockets"].discard(websocket)
        else:
            print(f"[WebSocket] Dashboard tried to connect to non-existent job: {job_id}")
            try:
                await websocket.send(json.dumps({"type": "ERROR", "msg": f"Job {job_id} not found"}))
                await websocket.close(1011, f"Job {job_id} not found")
            except:
                pass
    else:
        # Worker connection - await JOIN_JOB handshake frame
        worker_id = str(uuid.uuid4())
        current_job_id = None
        print(f"[WebSocket] Worker connected, waiting for JOIN_JOB: {worker_id}")

        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if data.get("type") == "JOIN_JOB":
                            job_id = data.get("jobId")
                            worker_name = data.get("workerName", "User Device")

                            if job_id in jobs:
                                current_job_id = job_id
                                job = jobs[job_id]
                                
                                # Register socket
                                job["active_worker_sockets"][worker_id] = websocket

                                # Add worker to job list if not already present
                                worker_entry = {"id": worker_id, "name": worker_name, "status": "IDLE"}
                                if not any(w["id"] == worker_id for w in job["config"]["workers"]):
                                    job["config"]["workers"].append(worker_entry)

                                print(f"[WebSocket] Worker {worker_id} joined Job {job_id} as '{worker_name}'")
                                await broadcast_dashboard(job_id)

                                # Trigger training loop if connected count >= minWorkers and not running
                                min_workers = int(job["config"]["modelConfig"].get("minWorkers", 3))
                                if len(job["active_worker_sockets"]) >= min_workers and job["config"]["status"] == "waiting":
                                    job["config"]["status"] = "running"
                                    asyncio.create_task(job_coordinator(job_id))
                            else:
                                print(f"[WebSocket] Worker tried to join non-existent job: {job_id}")
                                await websocket.send(json.dumps({"type": "ERROR", "msg": f"Job {job_id} not found"}))
                                await websocket.close(1011, f"Job {job_id} not found")
                                return
                    except Exception as e:
                        print(f"[WebSocket] Error parsing JOIN_JOB JSON: {e}")
                elif isinstance(message, bytes):
                    # Gradient payload received
                    if current_job_id is not None:
                        handle_gradient_for_job(current_job_id, worker_id, message)
                        await broadcast_dashboard(current_job_id)
                    else:
                        print(f"[WebSocket] Binary payload received from {worker_id} before handshake")
        finally:
            if current_job_id is not None and current_job_id in jobs:
                job = jobs[current_job_id]
                if worker_id in job["active_worker_sockets"]:
                    del job["active_worker_sockets"][worker_id]
                for w in job["config"]["workers"]:
                    if w["id"] == worker_id:
                        w["status"] = "OFFLINE"
                print(f"[WebSocket] Worker {worker_id} disconnected from Job {current_job_id}")
                await broadcast_dashboard(current_job_id)

# ═══════════════════════════════════════════════════════════════════════════════
#  FL Coordinator Loop per Job
# ═══════════════════════════════════════════════════════════════════════════════

async def job_coordinator(job_id):
    print(f"\n--- [Job {job_id}] Dynamic Coordinator Started ---")
    
    if job_id not in jobs:
        return
    job = jobs[job_id]
    config = job["config"]
    model_config = config["modelConfig"]

    print(f"  Min Workers       : {model_config.get('minWorkers', 3)}")
    print(f"  FedProx Mu        : {model_config.get('fedproxMu', 0.05)}")
    print(f"  Local Steps       : {model_config.get('localSteps', 3)}")
    print(f"  BFT method        : Weiszfeld Geometric Median (f < 0.5)")
    print(f"  Anti-gaming       : cosine similarity >= {MIN_COSINE_SIMILARITY}")
    print()

    while True:
        await asyncio.sleep(2)  # 2 seconds between rounds

        if job["training_stopped"]:
            config["status"] = "completed"
            await broadcast_dashboard(job_id)
            print(f"[Job {job_id}] Coordinator stopped.")
            break

        active_sockets = job["active_worker_sockets"]
        
        # Check connected idle workers
        idle_worker_ids = []
        for w in config["workers"]:
            w_id = w["id"]
            if w_id in active_sockets and w.get("status", "IDLE") == "IDLE":
                idle_worker_ids.append(w_id)

        if not idle_worker_ids:
            continue

        job["current_round"] += 1
        config["status"] = "running"
        await broadcast_dashboard(job_id)

        print(f"\n--- [Job {job_id}] Starting Round {job['current_round']} ---")
        print(f"Selected {len(idle_worker_ids)} workers.")

        # Dispatch tasks
        for w_id in idle_worker_ids:
            for w in config["workers"]:
                if w["id"] == w_id:
                    w["status"] = "BUSY"

            batch_x, batch_y = get_random_batch()
            payload = pack_payload_for_job(job, batch_x, batch_y)
            try:
                websocket = active_sockets[w_id]
                await websocket.send(payload)
            except Exception as e:
                print(f"[Job {job_id}] Failed to dispatch to worker {w_id}: {e}")

        # Wait a bit for they to process
        await asyncio.sleep(1)

        # Reset statuses
        for w_id in idle_worker_ids:
            for w in config["workers"]:
                if w["id"] == w_id:
                    w["status"] = "IDLE"

        await broadcast_dashboard(job_id)

# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP Request Handler (REST API + model.onnx server) on Port 8000
# ═══════════════════════════════════════════════════════════════════════════════

def create_job_state(job_id, owner_id, model_config):
    # Set default values for modelConfig if not provided
    min_workers = int(model_config.get("minWorkers", 3))
    
    # Initialize isolated weights and bias for this job
    W = np.zeros((NUM_FEATURES, NUM_CLASSES), dtype=np.float32)
    B = np.zeros(NUM_CLASSES, dtype=np.float32)
    
    return {
        "config": {
            "jobId": job_id,
            "ownerId": owner_id,
            "modelConfig": model_config,
            "status": "waiting",
            "workers": [],  # List of dicts: {"id": worker_id, "name": worker_name, "status": "IDLE"|"BUSY"}
            "metrics": {
                "loss": [],
                "accuracy": []
            }
        },
        "W": W,
        "B": B,
        "gradient_buffer": [],  # List of tuples (worker_id, dW, dB)
        "eval_history": [],
        "best_val_loss": float("inf"),
        "patience_counter": 0,
        "current_round": 0,
        "training_stopped": False,
        "active_worker_sockets": {},  # worker_id -> websocket
        "dashboard_sockets": set(),   # websockets of dashboard clients
    }

async def handle_http(reader, writer):
    try:
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            chunk = await reader.read(1024)
            if not chunk:
                break
            header_data += chunk
            if len(header_data) > 8192:
                break

        if not header_data:
            return

        parts = header_data.split(b"\r\n\r\n", 1)
        headers_part = parts[0]
        body_part = parts[1] if len(parts) > 1 else b""

        # Parse request line
        lines = headers_part.split(b"\r\n")
        req_line = lines[0].decode("utf-8")
        req_parts = req_line.split(" ")
        if len(req_parts) < 3:
            return
        method, path, _ = req_parts

        # Parse headers
        headers = {}
        for line in lines[1:]:
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.decode("utf-8").strip().lower()] = v.decode("utf-8").strip()

        # Read remaining body
        content_length = int(headers.get("content-length", 0))
        remaining = content_length - len(body_part)
        if remaining > 0:
            body_part += await reader.readexactly(remaining)

        # Handle CORS
        if method == "OPTIONS":
            writer.write(
                b"HTTP/1.1 204 No Content\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                b"Access-Control-Allow-Headers: Content-Type\r\n"
                b"Access-Control-Max-Age: 86400\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            return

        def send_json(data, status=200):
            body = json.dumps(data).encode("utf-8")
            status_text = "OK" if status == 200 else "Created" if status == 201 else "Not Found" if status == 404 else "Bad Request"
            res = (
                f"HTTP/1.1 {status} {status_text}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("utf-8") + body
            writer.write(res)

        def send_file(filepath, content_type="application/octet-stream"):
            try:
                import os
                if not os.path.exists(filepath):
                    send_json({"error": "File not found"}, 404)
                    return
                with open(filepath, "rb") as f:
                    content = f.read()
                res = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {len(content)}\r\n"
                    f"Access-Control-Allow-Origin: *\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("utf-8") + content
                writer.write(res)
            except Exception as e:
                send_json({"error": str(e)}, 500)

        # Routes
        if method == "GET" and path == "/host/model.onnx":
            send_file("host/model.onnx")
        elif method == "POST" and path == "/job/create":
            try:
                payload = json.loads(body_part.decode("utf-8")) if body_part else {}
            except:
                payload = {}

            job_id = str(uuid.uuid4())[:8]  # Short elegant jobId (e.g. abc123)
            owner_id = payload.get("ownerId", "anonymous")
            model_config = payload.get("modelConfig", {})

            # Defaults
            model_config.setdefault("minWorkers", 3)
            model_config.setdefault("fedproxMu", 0.05)
            model_config.setdefault("dpC", 1.0)
            model_config.setdefault("dpSigma", 0.2)
            model_config.setdefault("topK", 0.1)
            model_config.setdefault("quantize", "int8")
            model_config.setdefault("localSteps", 3)

            job_state = create_job_state(job_id, owner_id, model_config)
            jobs[job_id] = job_state
            print(f"[HTTP] Job Created: {job_id} (minWorkers={model_config['minWorkers']})")
            send_json(job_state["config"], 201)

        elif method == "GET" and path.startswith("/job/"):
            job_id = path.split("/")[-1]
            if job_id in jobs:
                send_json(jobs[job_id]["config"])
            else:
                send_json({"error": f"Job {job_id} not found"}, 404)

        elif method == "POST" and path == "/job/join":
            try:
                payload = json.loads(body_part.decode("utf-8"))
                job_id = payload.get("jobId")
                worker_id = payload.get("workerId", str(uuid.uuid4())[:8])
                worker_name = payload.get("workerName", "User Device")
            except:
                send_json({"error": "Invalid JSON"}, 400)
                return

            if not job_id:
                send_json({"error": "Missing jobId"}, 400)
                return

            if job_id in jobs:
                job = jobs[job_id]
                if not any(w["id"] == worker_id for w in job["config"]["workers"]):
                    job["config"]["workers"].append({
                        "id": worker_id,
                        "name": worker_name,
                        "status": "IDLE"
                    })
                await broadcast_dashboard(job_id)
                send_json(job["config"])
            else:
                send_json({"error": f"Job {job_id} not found"}, 404)
        else:
            send_json({"error": "Route not found"}, 404)

    except Exception as e:
        print(f"[HTTP Request Error] {e}")
    finally:
        writer.close()
        await writer.wait_closed()

# ═══════════════════════════════════════════════════════════════════════════════
#  Main Async Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    # Start REST API server on Port 8000
    http_server = await asyncio.start_server(handle_http, HOST, 8000)
    print(f"HTTP REST API Server listening on http://{HOST}:8000")

    # Start WebSocket Coordinator server on Port 8080
    ws_server = await websockets.serve(router, HOST, PORT)
    print(f"WebSocket Coordinator Server listening on ws://{HOST}:{PORT}")

    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
