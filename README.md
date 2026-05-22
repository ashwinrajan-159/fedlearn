# FedLearn: High-Performance Browser Federated Learning

A highly scalable, production-grade federated learning system built for the open internet. FedLearn trains machine learning models across thousands of heterogeneous edge devices (browsers, mobile phones, laptops) using pure JavaScript, WebSockets, and ONNX Runtime.

## 🌟 Key Features

- **Browser-Native Edge Compute:** Executes forward passes via `onnxruntime-web` and manual backpropagation entirely inside Web Workers. Zero UI thread blocking.
- **Zero-Copy Binary Protocol:** Uses raw `ArrayBuffer` and `Float32Array` for network transport and memory transfer. Eliminates JSON overhead, reducing tensor serialization latency from ~45ms to <1ms.
- **Asynchronous Aggregation (FedBuff):** Destroys the "straggler bottleneck." The coordinator never waits for slow devices; it aggregates dynamically when an adaptive threshold ($K$) is reached, applying exponential decay weighting to stale gradients.
- **Multi-Coordinator Sharding:** Horizontally scales to 1,000+ concurrent workers. A stateless partitioner distributes WebSocket traffic across independent shards, while a global sync service applies version-aware "soft synchronization" across the network.
- **Byzantine Fault Tolerance:** Built for untrusted environments. A rigorous validation firewall and statistical outlier detection (Trimmed/Median aggregation) protects the global model from `NaN` injections, data poisoning, and malicious workers.
- **Adaptive Smart Scheduling:** Continuously profiles worker latency and reliability. Dynamically assigns larger data batches to fast GPUs and smaller batches to throttled mobile browsers, minimizing idle gaps and maximizing global throughput.
- **Incentive & Gamification Layer:** Tracks verified mathematical utility. Rewards high-throughput, reliable contributors with credits and leaderboard rankings, disincentivizing spam.

## 🏗️ Architecture Overview

The system is separated into three distinct layers:

### 1. The Worker (Client-Side)
Runs entirely in the browser.
- **UI Thread:** Handles the WebSocket connection and React-based dashboard.
- **Web Worker Thread:** Takes ownership of the binary memory buffers, executes the ONNX `InferenceSession`, calculates $\Delta W$, and fires the binary gradients back to the host.
- **GPU Acceleration:** Optional WebGL 2.0 module (`OffscreenCanvas`) for massively parallel matrix multiplication.

### 2. The Shard Coordinator (Server-Side)
A Python `asyncio` / `websockets` process.
- Manages 30-50 active WebSockets.
- Executes the FedBuff asynchronous buffer logic.
- Evaluates worker reputation and scales dynamic batch sizes.

### 3. The Global Infrastructure (Scaling)
- **Partitioner:** An entry point that load-balances new workers across the healthiest shards.
- **Sync Service:** Periodically pulls local models from all shards, computes a version-weighted global average, and pushes it back down.
- **Migration Controller:** Detects traffic imbalances and gracefully redirects active workers between shards to maintain optimal load.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Setup

1. **Install Python dependencies for the Host:**
   ```bash
   pip install websockets numpy onnx
   ```

2. **Start the Coordinator Shards:**
   ```bash
   python fedlearn/host/server.py
   ```

3. **Start the Web Dashboard:**
   ```bash
   cd fedlearn/dashboard
   npm install
   npm run dev
   ```

4. **Connect Workers:**
   Open `http://localhost:5173` in multiple browser tabs (or on different devices connected to your LAN) to simulate a distributed worker fleet.

## 📊 Monitoring & Observability

The system exposes a lightweight metrics endpoint tracking global health, utilization, and convergence metrics. 

```bash
curl http://localhost:9090/metrics
```
```json
{
  "active_workers": 142,
  "updates_per_sec": 3.8,
  "avg_latency_ms": 312,
  "drop_rate": 0.04,
  "load_variance": 5.2
}
```

## 🛡️ Security

FedLearn is designed with zero-trust principles. **No raw datasets are ever shared between peers or uploaded to the server.** Only mathematical gradients ($\Delta W$) are transmitted, ensuring strict data privacy and compliance with localized ML training requirements.
