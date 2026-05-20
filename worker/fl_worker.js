importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let session = null;
let ws = null;
let alpha = 0.1; // Learning rate
let computeThreads = 1;

// Wait for configuration from main thread
self.onmessage = async (e) => {
    const { type, payload } = e.data;
    
    if (type === "INIT") {
        computeThreads = payload.threads || 1;
        ort.env.wasm.numThreads = computeThreads;
        
        // Initialize ONNX session
        try {
            // Fetch from static server
            session = await ort.InferenceSession.create('http://localhost:8000/host/model.onnx');
            self.postMessage({ type: "STATUS", msg: "ONNX model loaded." });
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
        setTimeout(connectWebSocket, 5000); // Reconnect
    };
}

async function processBatch(buffer) {
    self.postMessage({ type: "STATUS", msg: "Processing batch..." });
    const t0 = performance.now();
    
    // Unpack Binary Payload
    // [0-4]: B
    const dataView = new DataView(buffer);
    const B = dataView.getUint32(0, true);
    
    // Float32 views
    const floats = new Float32Array(buffer, 4);
    
    const W = floats.slice(0, 12);
    const bias = floats.slice(12, 15);
    const X = floats.slice(15, 15 + B * 4);
    const Y_true = floats.slice(15 + B * 4, 15 + B * 4 + B * 3);
    
    // Create ONNX Tensors
    const tensorX = new ort.Tensor('float32', X, [B, 4]);
    const tensorW = new ort.Tensor('float32', W, [4, 3]);
    const tensorB = new ort.Tensor('float32', bias, [3]);
    
    // Forward Pass
    const feeds = { X: tensorX, W: tensorW, B: tensorB };
    const results = await session.run(feeds);
    const probs = results.probs.data; // Float32Array of length B*3
    
    // Compute Gradients: dZ = P - Y
    const dZ = new Float32Array(B * 3);
    let loss = 0;
    
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
    
    // Normalize, apply learning rate, and clip
    for (let i = 0; i < 12; i++) {
        let grad = (alpha * dW[i]) / B;
        dW[i] = Math.max(-1.0, Math.min(1.0, grad));
    }
    
    for (let j = 0; j < 3; j++) {
        let grad = (alpha * db[j]) / B;
        db[j] = Math.max(-1.0, Math.min(1.0, grad));
    }
    
    // Pack Response
    const responseBuffer = new ArrayBuffer(60); // 48 bytes for dW + 12 bytes for db
    const responseFloats = new Float32Array(responseBuffer);
    responseFloats.set(dW, 0);
    responseFloats.set(db, 12);
    
    // Send back to Host
    ws.send(responseBuffer);
    
    const t1 = performance.now();
    self.postMessage({ 
        type: "RESULT", 
        msg: `Processed batch (loss: ${loss.toFixed(4)}) in ${(t1-t0).toFixed(1)}ms` 
    });
}
