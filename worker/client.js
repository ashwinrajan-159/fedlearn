"use strict";

const os = require("os");
const WebSocket = require("ws");
const { MESSAGE_TYPES, WORKER_STATES, safeParse, serialize } = require("../shared/messages");

// ---------------------------------------------------------------------------
// CLI argument parsing
//
// Usage:
//   node client.js <JOIN_CODE> <NAME>                       (direct)
//   node client.js <JOIN_CODE> <NAME> --relay ws://host:9000 (relay)
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);

if (args.length < 2) {
  console.error(
    "Usage: node client.js <JOIN_CODE> <WORKER_NAME> [--relay <RELAY_URL>]"
  );
  process.exit(1);
}

const joinCode = args[0];
const workerName = args[1];

let targetUrl = process.env.HOST_URL || "ws://localhost:8080";
let mode = "direct";

const relayFlagIdx = args.indexOf("--relay");
if (relayFlagIdx !== -1) {
  const relayUrl = args[relayFlagIdx + 1];
  if (!relayUrl) {
    console.error(
      "Error: --relay flag requires a URL argument (e.g. ws://localhost:9000)"
    );
    process.exit(1);
  }
  targetUrl = relayUrl;
  mode = "relay";
}

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------
const capabilities = {
  cpuCores: os.cpus().length,
  ramGB: +(os.totalmem() / (1024 ** 3)).toFixed(1),
  gpu: false,
};

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------
let workerState = WORKER_STATES.IDLE;
let myWorkerId = null;

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------
console.log(`\n=== FedLearn Worker ===`);
console.log(`Name  : ${workerName}`);
console.log(`Code  : ${joinCode}`);
console.log(`Mode  : ${mode}`);
console.log(`Target: ${targetUrl}`);
console.log(
  `Caps  : ${capabilities.cpuCores} CPU cores, ${capabilities.ramGB} GB RAM, GPU: ${capabilities.gpu}\n`
);

const socket = new WebSocket(targetUrl);

socket.on("open", () => {
  console.log("Connected — sending JOIN…");

  socket.send(
    serialize({
      type: MESSAGE_TYPES.JOIN,
      code: joinCode,
      workerName,
      capabilities,
    })
  );
});

socket.on("message", (raw) => {
  const msg = safeParse(raw.toString());
  if (!msg) {
    console.error("Received malformed message from server.");
    return;
  }

  switch (msg.type) {
    case MESSAGE_TYPES.ACK:
      myWorkerId = msg.workerId;
      console.log(`✓ Joined session.  Worker ID: ${msg.workerId}`);
      console.log(`  Server says: ${msg.message}\n`);
      break;

    case MESSAGE_TYPES.REJECT:
      console.error(`✗ Rejected: ${msg.message}`);
      socket.close();
      break;

    case MESSAGE_TYPES.PING:
      handlePing(msg);
      break;

    case MESSAGE_TYPES.ROUND_START:
      handleRoundStart(msg);
      break;

    case MESSAGE_TYPES.ROUND_END:
      handleRoundEnd(msg);
      break;

    case MESSAGE_TYPES.TASK_ASSIGN:
      handleTaskAssign(msg);
      break;

    default:
      break;
  }
});

// ---------------------------------------------------------------------------
// Heartbeat
// ---------------------------------------------------------------------------
function handlePing(msg) {
  const now = Date.now();
  const rtt = typeof msg.timestamp === "number" ? now - msg.timestamp : "?";

  socket.send(
    serialize({ type: MESSAGE_TYPES.PONG, timestamp: msg.timestamp })
  );

  if (workerState === WORKER_STATES.IDLE) {
    process.stdout.write(`\r  ♥ heartbeat  |  rtt ≈ ${rtt} ms  |  state: ${workerState}   `);
  }
}

// ---------------------------------------------------------------------------
// Round handlers
// ---------------------------------------------------------------------------
function handleRoundStart(msg) {
  console.log(`\n┌─ ROUND START: ${msg.roundId.slice(0, 8)}`);
}

function handleRoundEnd(msg) {
  console.log(`└─ ROUND END: ${msg.roundId.slice(0, 8)}\n`);
}

// ---------------------------------------------------------------------------
// Task handler
// ---------------------------------------------------------------------------
function handleTaskAssign(msg) {
  const { taskId, payload, roundId } = msg;

  // Transition to BUSY
  workerState = WORKER_STATES.BUSY;
  console.log(`  │  📥 Task received: ${taskId.slice(0, 8)} (${payload.taskType})`);

  // Send TASK_ACK immediately
  socket.send(
    serialize({ type: MESSAGE_TYPES.TASK_ACK, taskId })
  );

  // Simulate work: random 2–5 second delay
  const workDurationMs = 2000 + Math.floor(Math.random() * 3000);
  console.log(`  │  ⏳ Working… (${(workDurationMs / 1000).toFixed(1)}s)`);

  setTimeout(() => {
    // Send TASK_COMPLETE
    socket.send(
      serialize({
        type: MESSAGE_TYPES.TASK_COMPLETE,
        taskId,
        result: "done",
      })
    );

    // Transition back to IDLE
    workerState = WORKER_STATES.IDLE;
    console.log(`  │  ✅ Task ${taskId.slice(0, 8)} completed`);
  }, workDurationMs);
}

// ---------------------------------------------------------------------------
// Disconnect
// ---------------------------------------------------------------------------
socket.on("close", (code, reason) => {
  console.log(`\nDisconnected (code=${code}, reason=${reason || "none"})`);
});

socket.on("error", (err) => {
  console.error(`Connection error: ${err.message}`);
});
