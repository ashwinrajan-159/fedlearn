"use strict";

/**
 * Shared message type constants used across Host, Worker, and Relay.
 * Every message over the wire MUST use one of these types.
 */
const MESSAGE_TYPES = {
  // --- Connection ---
  JOIN: "JOIN",
  ACK: "ACK",
  REJECT: "REJECT",

  // --- Heartbeat ---
  PING: "PING",
  PONG: "PONG",

  // --- Coordination ---
  TASK_ASSIGN: "TASK_ASSIGN",
  TASK_ACK: "TASK_ACK",
  TASK_COMPLETE: "TASK_COMPLETE",
  ROUND_START: "ROUND_START",
  ROUND_END: "ROUND_END",
};

/**
 * Worker state machine constants.
 */
const WORKER_STATES = {
  IDLE: "IDLE",
  BUSY: "BUSY",
  DISCONNECTED: "DISCONNECTED",
};

/**
 * Task status constants (host-side tracking).
 */
const TASK_STATUSES = {
  ASSIGNED: "ASSIGNED",
  ACKED: "ACKED",
  COMPLETED: "COMPLETED",
};

/**
 * Safely parse a JSON string. Returns null on failure instead of throwing.
 * @param {string} raw
 * @returns {object|null}
 */
function safeParse(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/**
 * Serialize a message object to a JSON string.
 * @param {object} msg
 * @returns {string}
 */
function serialize(msg) {
  return JSON.stringify(msg);
}

module.exports = {
  MESSAGE_TYPES,
  WORKER_STATES,
  TASK_STATUSES,
  safeParse,
  serialize,
};
