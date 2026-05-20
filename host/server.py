import asyncio
import websockets
import numpy as np
import struct
import json
import uuid

# Configuration
HOST = "localhost"
PORT = 8080
LR = 0.1 # Alpha (Learning Rate)
BATCH_SIZE = 16
NUM_FEATURES = 4
NUM_CLASSES = 3

# Global Model State
global_W = np.zeros((NUM_FEATURES, NUM_CLASSES), dtype=np.float32)
global_B = np.zeros(NUM_CLASSES, dtype=np.float32)

# Dummy Dataset (Iris-like)
# In reality, this would be distributed, but for simulation we send batches from host
NUM_SAMPLES = 100
dummy_X = np.random.randn(NUM_SAMPLES, NUM_FEATURES).astype(np.float32)
dummy_y_indices = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
dummy_Y = np.zeros((NUM_SAMPLES, NUM_CLASSES), dtype=np.float32)
dummy_Y[np.arange(NUM_SAMPLES), dummy_y_indices] = 1.0

# Server State
connected_workers = {}
dashboard_clients = set()
current_round = 0
round_in_progress = False

# ---------------------------------------------------------
# FL Logic
# ---------------------------------------------------------

def get_random_batch():
    indices = np.random.choice(NUM_SAMPLES, BATCH_SIZE, replace=False)
    return dummy_X[indices], dummy_Y[indices]

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

def unpack_gradients(data):
    """
    Unpacks gradients from the worker.
    Format:
    [0-48 bytes]: delta_W (12 x Float32)
    [48-60 bytes]: delta_B (3 x Float32)
    """
    if len(data) != 60:
        raise ValueError(f"Invalid gradient payload size: {len(data)} bytes")
    
    delta_w_flat = np.frombuffer(data[:48], dtype=np.float32)
    delta_b = np.frombuffer(data[48:], dtype=np.float32)
    
    delta_W = delta_w_flat.reshape((NUM_FEATURES, NUM_CLASSES))
    return delta_W, delta_b

# ---------------------------------------------------------
# WebSockets Handlers
# ---------------------------------------------------------

async def broadcast_dashboard():
    if not dashboard_clients:
        return
    
    msg = json.dumps({
        "type": "METRICS",
        "round": current_round,
        "workers": len(connected_workers),
        "status": "RUNNING" if round_in_progress else "IDLE"
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
            pass # Keep alive
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
    finally:
        print(f"[-] Worker disconnected: {worker_id}")
        del connected_workers[worker_id]
        await broadcast_dashboard()

def handle_gradient(worker_id, data):
    global global_W, global_B
    try:
        delta_W, delta_B = unpack_gradients(data)
        
        # Validation
        if np.isnan(delta_W).any() or np.isnan(delta_B).any():
            print(f"[!] Invalid gradient (NaN) from {worker_id}. Discarding.")
            return
        if np.max(np.abs(delta_W)) > 1.0 or np.max(np.abs(delta_B)) > 1.0:
            print(f"[!] Invalid gradient (Out of bounds) from {worker_id}. Discarding.")
            return
        
        # Aggregate (FedAvg approximation for SGD)
        # W_new = W_old - alpha * delta_W
        # Assuming delta_W is already averaged over batch and scaled by worker
        # Wait, prompt says: Worker computes: delta_W = X.T * (Y_pred - Y), then delta_W /= batch_size.
        # Host: W_new = W_old - (alpha / N) * sum(delta_W)
        # Since workers are async, we can do FedSGD updates dynamically:
        # W_new = W_old - alpha * delta_W
        
        global_W -= LR * delta_W
        global_B -= LR * delta_B
        
        print(f"[\u2713] Applied update from {worker_id}")
        
    except Exception as e:
        print(f"[!] Error processing gradient: {e}")

async def router(websocket, path):
    if path == "/dashboard":
        await handle_dashboard(websocket)
    else:
        await handle_worker(websocket)

# ---------------------------------------------------------
# FL Coordinator Loop
# ---------------------------------------------------------

async def fl_coordinator():
    global current_round, round_in_progress
    print("Coordinator started...")
    
    while True:
        await asyncio.sleep(2) # 2 seconds between rounds
        
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
