"use strict";

const crypto = require("crypto");

/**
 * Generate a join code in the format XX-0000 (e.g. "KX-4729").
 *
 * - Two uppercase ASCII letters (A-Z)
 * - A dash
 * - Four decimal digits (0000–9999)
 *
 * Uses crypto.randomBytes for unpredictability.
 *
 * @returns {string}
 */
function generateJoinCode() {
  const bytes = crypto.randomBytes(4);

  // First two bytes → letters A-Z (mod 26)
  const letter1 = String.fromCharCode(65 + (bytes[0] % 26));
  const letter2 = String.fromCharCode(65 + (bytes[1] % 26));

  // Last two bytes → 4-digit number (0–9999), zero-padded
  const num = ((bytes[2] << 8) | bytes[3]) % 10000;
  const digits = String(num).padStart(4, "0");

  return `${letter1}${letter2}-${digits}`;
}

module.exports = { generateJoinCode };
