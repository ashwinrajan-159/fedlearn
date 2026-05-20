"use strict";

const { WebSocketServer, OPEN } = require("ws");
const { v4: uuidv4 } = require("uuid");
const { generateJoinCode } = require("./codeGenerator");
const {
  MESSAGE_TYPES,
  WORKER_STATES,
  TASK_STATUSES,
  safeParse,
  serialize,
} = require("../shared/messages");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const HOST_PORT = parseInt(process.env.HOST_PORT, 10) || 8080;

const HEARTBEAT_INTERVAL_MS = 5_000;   // send PING every 5 s
const WORKER_TIMEOUT_MS     = 15_000;  // consider dead after 15 s
const SESSION_CLEANUP_MS    = 30_000;  // check expired sessions every 30 s
const SESSION_TTL_MS        = 60 * 60 * 1_000; // 1 hour

// --- Coordinator config ---
const WORKERS_PER_ROUND = parseInt(process.env.WORKERS_PER_ROUND, 10) || 3;
const ROUND_DELAY_MS    = 10_000; // wait 10 s after startup for workers to join
const ROUND_AUTO_REPEAT = process.env.ROUND_AUTO_REPEAT !== "false"; // run continuous rounds

// ---------------------------------------------------------------------------
// Session state  —  Map<code, session>
// ---------------------------------------------------------------------------
const sessions = new Map();

// ---------------------------------------------------------------------------
// Task tracker  —  Map<taskId, taskRecord>
//
// taskRecord = {
//   workerId,
//   sessionCode,
//   roundId,
//   status   : TASK_STATUSES.*,
//   assignedAt: number
// }
// ---------------------------------------------------------------------------
const tasks = new Map();

/**
 * Create a new session with a unique join code.
 * @returns {string} The generated join code.
 */
function createSession() {
  let code;
  do {
    code = generateJoinCode();
  } while (sessions.has(code));

  const now = Date.now();
  sessions.set(code, {
    workers: new Map(),
    createdAt: now,
    expiresAt: now + SESSION_TTL_MS,
  });
  return code;
}

/**
 * Detect whether the connection arrived directly or through a relay.
 * @param {import("http").IncomingMessage} req
 * @returns {"relay"|"direct"}
 */
function detectConnectionType(req) {
  if (req.headers["x-relay"] || req.headers["x-forwarded-for"]) {
    return "relay";
  }
  return "direct";
}

// ===========================================================================
//  COORDINATOR — Worker Selection
// ===========================================================================

/**
 * Select the best N idle workers from a session.
 *
 * Ranking:
 *   1. Lowest latencyMs  (ASC)   — prefer responsive workers
 *   2. Highest cpuCores  (DESC)  — tie-break by compute power
 *
 * Workers with null latency are placed last (not yet measured).
 *
 * @param {Map<string, object>} workers
 * @param {number} n
 * @returns {Array<{workerId: string, worker: object}>}
 */
function selectWorkers(workers, n) {
  const idle = [];
  for (const [workerId, worker] of workers) {
    if (worker.state === WORKER_STATES.IDLE && worker.socket.readyState === OPEN) {
      idle.push({ workerId, worker });
    }
  }

  idle.sort((a, b) => {
    const latA = a.worker.latencyMs ?? Infinity;
    const latB = b.worker.latencyMs ?? Infinity;
    if (latA !== latB) return latA - latB;

    const cpuA = a.worker.capabilities.cpuCores || 0;
    const cpuB = b.worker.capabilities.cpuCores || 0;
    return cpuB - cpuA; // DESC
  });

  return idle.slice(0, n);
}

// ===========================================================================
//  COORDINATOR — Round Control
// ===========================================================================

let roundInProgress = false;

/**
 * Execute one complete round:
 *   1. Select workers
 *   2. Broadcast ROUND_START
 *   3. Assign a TASK_ASSIGN to each selected worker
 *   4. Wait for all TASK_COMPLETEs (with fault-tolerance reassignment)
 *   5. Broadcast ROUND_END
 *
 * @param {string} sessionCode
 * @returns {Promise<void>}
 */
async function runRound(sessionCode) {
  const session = sessions.get(sessionCode);
  if (!session) return;

  const roundId = uuidv4();
  roundInProgress = true;

  // --- Select workers ---
  const selected = selectWorkers(session.workers, WORKERS_PER_ROUND);
  if (selected.length === 0) {
    console.log(`[round ${roundId.slice(0, 8)}] No idle workers — skipping round`);
    roundInProgress = false;
    return;
  }

  console.log(
    `\n╔══════════════════════════════════════════════════╗` +
    `\n║  ROUND START: ${roundId.slice(0, 8)}                            ║` +
    `\n║  Workers selected: ${selected.length}/${session.workers.size} available            ║` +
    `\n╚══════════════════════════════════════════════════╝`
  );

  console.log(`  Selected:`);
  for (const { workerId, worker } of selected) {
    console.log(
      `    • ${worker.name} (${workerId.slice(0, 8)}) ` +
      `— latency: ${worker.latencyMs ?? "?"}ms, cpu: ${worker.capabilities.cpuCores}`
    );
  }

  // --- Broadcast ROUND_START to selected workers ---
  for (const { worker } of selected) {
    sendTo(worker.socket, { type: MESSAGE_TYPES.ROUND_START, roundId });
  }

  // --- Assign tasks ---
  const roundTasks = new Map(); // taskId → workerId
  for (const { workerId, worker } of selected) {
    const taskId = uuidv4();
    tasks.set(taskId, {
      workerId,
      sessionCode,
      roundId,
      status: TASK_STATUSES.ASSIGNED,
      assignedAt: Date.now(),
    });
    roundTasks.set(taskId, workerId);

    worker.state = WORKER_STATES.BUSY;

    sendTo(worker.socket, {
      type: MESSAGE_TYPES.TASK_ASSIGN,
      taskId,
      roundId,
      payload: { taskType: "DUMMY_COMPUTE", data: "simulate work" },
    });

    console.log(`  → Task ${taskId.slice(0, 8)} assigned to ${worker.name} (${workerId.slice(0, 8)})`);
  }

  // --- Wait for all tasks to complete (with fault-tolerance) ---
  try {
    await waitForRoundCompletion(roundId, roundTasks, sessionCode);
  } catch (err) {
    console.error(`  ✗ Round ${roundId.slice(0, 8)} failed: ${err.message}`);
  }

  // --- Broadcast ROUND_END ---
  for (const { worker } of selected) {
    if (worker.socket.readyState === OPEN) {
      sendTo(worker.socket, { type: MESSAGE_TYPES.ROUND_END, roundId });
    }
  }

  console.log(
    `\n╔══════════════════════════════════════════════════╗` +
    `\n║  ROUND END: ${roundId.slice(0, 8)}                              ║` +
    `\n╚══════════════════════════════════════════════════╝\n`
  );

  roundInProgress = false;
}

/**
 * Wait until every task in the round reaches COMPLETED status.
 *
 * If a worker disconnects or times out mid-task, the task is reassigned
 * to another idle worker (if available).
 *
 * @param {string} roundId
 * @param {Map<string, string>} roundTasks  taskId → workerId
 * @param {string} sessionCode
 * @returns {Promise<void>}
 */
function waitForRoundCompletion(roundId, roundTasks, sessionCode) {
  return new Promise((resolve, reject) => {
    const MAX_WAIT_MS = 60_000; // hard ceiling per round
    const CHECK_INTERVAL_MS = 1_000;
    const MAX_REASSIGN_ATTEMPTS = 3;
    let elapsed = 0;

    const timer = setInterval(() => {
      elapsed += CHECK_INTERVAL_MS;

      let allDone = true;
      for (const [taskId, origWorkerId] of roundTasks) {
        const task = tasks.get(taskId);
        if (!task) continue;

        if (task.status === TASK_STATUSES.COMPLETED) continue;

        allDone = false;

        // Check if the assigned worker is still alive
        const session = sessions.get(sessionCode);
        if (!session) { clearInterval(timer); reject(new Error("Session gone")); return; }

        const worker = session.workers.get(task.workerId);
        const isAlive = worker && worker.socket.readyState === OPEN;

        if (!isAlive) {
          // Attempt reassignment
          const reassigned = reassignTask(taskId, sessionCode, roundId);
          if (!reassigned) {
            console.log(`  ⚠ Task ${taskId.slice(0, 8)}: no workers available for reassignment`);
          }
        }
      }

      if (allDone) {
        clearInterval(timer);
        resolve();
        return;
      }

      if (elapsed >= MAX_WAIT_MS) {
        clearInterval(timer);
        reject(new Error("Round timed out"));
        return;
      }
    }, CHECK_INTERVAL_MS);

    timer.unref();
  });
}

/**
 * Reassign a failed task to another idle worker.
 *
 * @param {string} taskId
 * @param {string} sessionCode
 * @param {string} roundId
 * @returns {boolean} true if reassigned successfully
 */
function reassignTask(taskId, sessionCode, roundId) {
  const session = sessions.get(sessionCode);
  if (!session) return false;

  const task = tasks.get(taskId);
  if (!task) return false;

  // Pick one idle worker that is NOT the previously assigned one
  const candidates = selectWorkers(session.workers, 1)
    .filter((c) => c.workerId !== task.workerId);

  if (candidates.length === 0) return false;

  const { workerId, worker } = candidates[0];

  // Update tracking
  task.workerId = workerId;
  task.status = TASK_STATUSES.ASSIGNED;
  task.assignedAt = Date.now();

  worker.state = WORKER_STATES.BUSY;

  sendTo(worker.socket, {
    type: MESSAGE_TYPES.TASK_ASSIGN,
    taskId,
    roundId,
    payload: { taskType: "DUMMY_COMPUTE", data: "simulate work" },
  });

  console.log(
    `  ↻ Task ${taskId.slice(0, 8)} reassigned to ${worker.name} (${workerId.slice(0, 8)})`
  );
  return true;
}

// ===========================================================================
//  HELPER
// ===========================================================================

/** Send a serialized message to a socket, swallowing errors. */
function sendTo(socket, msg) {
  try {
    if (socket.readyState === OPEN) {
      socket.send(serialize(msg));
    }
  } catch { /* swallow */ }
}

// ===========================================================================
//  WebSocket Server
// ===========================================================================
const wss = new WebSocketServer({ port: HOST_PORT });

wss.on("connection", (socket, req) => {
  const connectionType = detectConnectionType(req);

  let boundWorkerId = null;
  let boundSessionCode = null;

  socket.on("message", (raw) => {
    const msg = safeParse(raw.toString());
    if (!msg) {
      sendTo(socket, { type: MESSAGE_TYPES.REJECT, message: "Malformed JSON" });
      return;
    }

    switch (msg.type) {
      case MESSAGE_TYPES.JOIN:
        handleJoin(socket, msg, connectionType);
        break;

      case MESSAGE_TYPES.PONG:
        handlePong(msg);
        break;

      case MESSAGE_TYPES.TASK_ACK:
        handleTaskAck(msg);
        break;

      case MESSAGE_TYPES.TASK_COMPLETE:
        handleTaskComplete(msg);
        break;

      default:
        break;
    }
  });

  // -----------------------------------------------------------------------
  //  JOIN handler (unchanged logic, expanded worker model)
  // -----------------------------------------------------------------------
  function handleJoin(ws, msg, connType) {
    const { code, workerName, capabilities } = msg;

    if (!code || !sessions.has(code)) {
      sendTo(ws, { type: MESSAGE_TYPES.REJECT, message: "Invalid code" });
      return;
    }

    const session = sessions.get(code);
    const workerId = uuidv4();
    const now = Date.now();

    session.workers.set(workerId, {
      socket: ws,
      name: workerName || "anonymous",
      connectionType: connType,
      capabilities: capabilities || { cpuCores: 0, ramGB: 0, gpu: false },
      lastSeen: now,
      latencyMs: null,
      state: WORKER_STATES.IDLE,
    });

    boundWorkerId = workerId;
    boundSessionCode = code;

    sendTo(ws, {
      type: MESSAGE_TYPES.ACK,
      workerId,
      message: "Connected",
    });

    const caps = capabilities || {};
    console.log(
      `Worker connected: ${workerName || "anonymous"} (${workerId}) [${connType}]` +
        ` | CPU: ${caps.cpuCores || "?"}  RAM: ${caps.ramGB || "?"}GB  GPU: ${caps.gpu ?? "?"}`
    );
  }

  // -----------------------------------------------------------------------
  //  PONG handler (latency tracking)
  // -----------------------------------------------------------------------
  function handlePong(msg) {
    if (!boundWorkerId || !boundSessionCode) return;
    const session = sessions.get(boundSessionCode);
    if (!session) return;
    const worker = session.workers.get(boundWorkerId);
    if (!worker) return;

    const now = Date.now();
    if (typeof msg.timestamp === "number") {
      worker.latencyMs = now - msg.timestamp;
    }
    worker.lastSeen = now;
  }

  // -----------------------------------------------------------------------
  //  TASK_ACK handler
  // -----------------------------------------------------------------------
  function handleTaskAck(msg) {
    const task = tasks.get(msg.taskId);
    if (!task) return;

    task.status = TASK_STATUSES.ACKED;
    console.log(`  ✓ Task ${msg.taskId.slice(0, 8)} ACKed by worker`);
  }

  // -----------------------------------------------------------------------
  //  TASK_COMPLETE handler
  // -----------------------------------------------------------------------
  function handleTaskComplete(msg) {
    const task = tasks.get(msg.taskId);
    if (!task) return;

    task.status = TASK_STATUSES.COMPLETED;

    // Set worker back to IDLE
    const session = sessions.get(task.sessionCode);
    if (session) {
      const worker = session.workers.get(task.workerId);
      if (worker) {
        worker.state = WORKER_STATES.IDLE;
      }
    }

    console.log(
      `  ✓ Task ${msg.taskId.slice(0, 8)} COMPLETED (result: ${msg.result || "n/a"})`
    );
  }

  // -----------------------------------------------------------------------
  //  Disconnection handling
  // -----------------------------------------------------------------------
  socket.on("close", () => {
    if (boundWorkerId && boundSessionCode && sessions.has(boundSessionCode)) {
      const session = sessions.get(boundSessionCode);
      const worker = session.workers.get(boundWorkerId);
      if (worker) {
        worker.state = WORKER_STATES.DISCONNECTED;
        session.workers.delete(boundWorkerId);
      }
      console.log(`Worker disconnected: ${boundWorkerId}`);
    }
  });

  socket.on("error", (err) => {
    console.error(`Socket error: ${err.message}`);
  });
});

// ---------------------------------------------------------------------------
// Heartbeat — PING every HEARTBEAT_INTERVAL_MS
// ---------------------------------------------------------------------------
const heartbeatTimer = setInterval(() => {
  const now = Date.now();

  for (const [code, session] of sessions) {
    for (const [workerId, worker] of session.workers) {
      if (now - worker.lastSeen > WORKER_TIMEOUT_MS) {
        console.log(`Worker timed out: ${workerId}`);
        worker.state = WORKER_STATES.DISCONNECTED;
        if (worker.socket.readyState === OPEN) {
          worker.socket.close(1000, "Heartbeat timeout");
        }
        session.workers.delete(workerId);
        continue;
      }

      if (worker.socket.readyState === OPEN) {
        worker.socket.send(
          serialize({ type: MESSAGE_TYPES.PING, timestamp: now })
        );
      }
    }
  }
}, HEARTBEAT_INTERVAL_MS);

// ---------------------------------------------------------------------------
// Session expiry — check every SESSION_CLEANUP_MS
// ---------------------------------------------------------------------------
const sessionCleanupTimer = setInterval(() => {
  const now = Date.now();

  for (const [code, session] of sessions) {
    if (now >= session.expiresAt) {
      for (const [, worker] of session.workers) {
        if (worker.socket.readyState === OPEN) {
          worker.socket.close(1000, "Session expired");
        }
      }
      sessions.delete(code);
      console.log(`Session expired: ${code}`);
    }
  }
}, SESSION_CLEANUP_MS);

heartbeatTimer.unref();
sessionCleanupTimer.unref();

// ---------------------------------------------------------------------------
// Startup + Round Scheduler
// ---------------------------------------------------------------------------
const sessionCode = createSession();

wss.on("listening", () => {
  console.log(`\n=== FedLearn Host (Coordinator) ===`);
  console.log(`Listening on ws://localhost:${HOST_PORT}`);
  console.log(`Join code       : ${sessionCode}`);
  console.log(`Heartbeat       : every ${HEARTBEAT_INTERVAL_MS / 1000}s`);
  console.log(`Timeout         : ${WORKER_TIMEOUT_MS / 1000}s`);
  console.log(`Session TTL     : ${SESSION_TTL_MS / 1000 / 60} min`);
  console.log(`Workers/round   : ${WORKERS_PER_ROUND}`);
  console.log(`First round in  : ${ROUND_DELAY_MS / 1000}s`);
  console.log(`Waiting for workers…\n`);

  // Schedule rounds after an initial delay
  setTimeout(() => scheduleRound(sessionCode), ROUND_DELAY_MS);
});

/**
 * Schedule a round.  After it completes, schedule the next one (if auto-repeat).
 */
async function scheduleRound(sessionCode) {
  await runRound(sessionCode);

  if (ROUND_AUTO_REPEAT && sessions.has(sessionCode)) {
    // Wait a short pause between rounds, then run the next
    setTimeout(() => scheduleRound(sessionCode), 5_000);
  }
}
