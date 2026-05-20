"use strict";

const { WebSocketServer, WebSocket } = require("ws");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const RELAY_PORT = parseInt(process.env.RELAY_PORT, 10) || 9000;
const HOST_URL = process.env.HOST_URL || "ws://localhost:8080";

// ---------------------------------------------------------------------------
// Relay Server
//
// For every incoming worker connection the relay opens a mirror connection
// to the Host and transparently pipes messages in both directions.
//
// The relay is fully stateless — it never inspects, stores, or modifies
// payloads.  It adds an `x-relay: true` header on the upstream connection
// so the Host can detect relayed workers.
// ---------------------------------------------------------------------------
const wss = new WebSocketServer({ port: RELAY_PORT });

wss.on("connection", (workerSocket, req) => {
  const workerAddr =
    req.socket.remoteAddress + ":" + req.socket.remotePort;
  console.log(`[relay] Worker connected from ${workerAddr}`);

  // Open upstream connection to the Host
  const hostSocket = new WebSocket(HOST_URL, {
    headers: { "x-relay": "true" },
  });

  let hostReady = false;
  const pendingMessages = []; // buffer messages until upstream is open

  hostSocket.on("open", () => {
    hostReady = true;
    // Flush anything the worker sent before upstream was ready
    for (const msg of pendingMessages) {
      hostSocket.send(msg);
    }
    pendingMessages.length = 0;
  });

  // Worker → Host
  workerSocket.on("message", (data) => {
    if (hostReady) {
      hostSocket.send(data);
    } else {
      pendingMessages.push(data);
    }
  });

  // Host → Worker
  hostSocket.on("message", (data) => {
    if (workerSocket.readyState === WebSocket.OPEN) {
      workerSocket.send(data);
    }
  });

  // --- Cleanup on either side closing ---

  workerSocket.on("close", () => {
    console.log(`[relay] Worker disconnected (${workerAddr})`);
    if (hostSocket.readyState === WebSocket.OPEN) {
      hostSocket.close();
    }
  });

  hostSocket.on("close", () => {
    if (workerSocket.readyState === WebSocket.OPEN) {
      workerSocket.close();
    }
  });

  // --- Error handling ---

  workerSocket.on("error", (err) => {
    console.error(`[relay] Worker socket error: ${err.message}`);
  });

  hostSocket.on("error", (err) => {
    console.error(`[relay] Host socket error: ${err.message}`);
    // If we can't reach the host, tell the worker and close
    if (workerSocket.readyState === WebSocket.OPEN) {
      workerSocket.close(1011, "Relay could not reach host");
    }
  });
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
wss.on("listening", () => {
  console.log(`\n=== FedLearn Relay ===`);
  console.log(`Listening on ws://localhost:${RELAY_PORT}`);
  console.log(`Forwarding to ${HOST_URL}\n`);
});
