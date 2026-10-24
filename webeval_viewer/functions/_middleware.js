// Pages Functions middleware — runs before every /functions/* handler.
//
// Acts as an auth gate on /api/* (except /api/login and /api/logout). It
// verifies the `session` cookie is a valid HMAC-SHA256 signed token issued by
// /api/login and has not expired. Invalid/missing/expired -> 401.
//
// Static assets (HTML/JS/CSS/data) are not subject to this middleware — Pages
// only runs Functions for paths under /functions/, which manifests at runtime
// as /api/* (matching `functions/api/*.js`). Public assets like /index.html
// and /data/* are served directly from the bundled `public/` directory.

const ENC = new TextEncoder();
const DEC = new TextDecoder();

function ub64(str) {
  // URL-safe base64 decode — accepts the two URL-safe characters too.
  const padded = str.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(str.length / 4) * 4, "=");
  return Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
}

function eqBytes(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

async function hmac(payloadB64, secret) {
  const key = await crypto.subtle.importKey(
    "raw", ENC.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return new Uint8Array(await crypto.subtle.sign("HMAC", key, ENC.encode(payloadB64)));
}

function readCookie(request, name) {
  const header = request.headers.get("Cookie") || "";
  for (const part of header.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name) return rest.join("=");
  }
  return null;
}

function unauthorized(detail = "Unauthorized") {
  return new Response(
    JSON.stringify({ error: detail }),
    { status: 401, headers: { "content-type": "application/json" } },
  );
}

async function verifySession(request, env) {
  if (!env.SESSION_SECRET) return { ok: false, reason: "Server missing SESSION_SECRET." };
  const cookie = readCookie(request, "session");
  if (!cookie) return { ok: false, reason: "No session cookie." };
  const dot = cookie.indexOf(".");
  if (dot < 0) return { ok: false, reason: "Malformed session." };
  const payloadB64 = cookie.slice(0, dot);
  const sigB64     = cookie.slice(dot + 1);
  let expected, actual;
  try {
    expected = await hmac(payloadB64, env.SESSION_SECRET);
    actual   = ub64(sigB64);
  } catch {
    return { ok: false, reason: "Bad encoding." };
  }
  if (!eqBytes(actual, expected)) return { ok: false, reason: "Bad signature." };
  let payload;
  try { payload = JSON.parse(DEC.decode(ub64(payloadB64))); }
  catch { return { ok: false, reason: "Bad payload." }; }
  if (typeof payload.exp !== "number" || payload.exp < Date.now()) {
    return { ok: false, reason: "Session expired." };
  }
  return { ok: true };
}

export const onRequest = async (context) => {
  const { request, next, env } = context;
  const url = new URL(request.url);

  // Only gate /api/*; everything else (static assets) is public.
  if (!url.pathname.startsWith("/api/")) return next();

  // Login and logout do not require a valid session.
  if (url.pathname === "/api/login" || url.pathname === "/api/logout") return next();

  const v = await verifySession(request, env);
  if (!v.ok) return unauthorized(v.reason);
  return next();
};
