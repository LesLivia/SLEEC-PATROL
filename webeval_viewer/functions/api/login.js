// POST /api/login
//
// Validates credentials against env.ADMIN_USERNAME / env.ADMIN_PASSWORD. On
// success, issues an HMAC-SHA256-signed session cookie. The cookie is
// HttpOnly + Secure + SameSite=Strict so it can't be read by JS, sent over
// non-HTTPS, or attached to cross-site requests.

const ENC = new TextEncoder();
const SESSION_TTL_SECONDS = 8 * 60 * 60; // 8 hours

function b64(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

async function hmac(payloadB64, secret) {
  const key = await crypto.subtle.importKey(
    "raw", ENC.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return new Uint8Array(await crypto.subtle.sign("HMAC", key, ENC.encode(payloadB64)));
}

function badRequest(msg) {
  return new Response(
    JSON.stringify({ error: msg }),
    { status: 400, headers: { "content-type": "application/json" } },
  );
}

function constantTimeStringEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export const onRequestPost = async ({ env, request }) => {
  if (!env.ADMIN_USERNAME || !env.ADMIN_PASSWORD || !env.SESSION_SECRET) {
    return new Response(
      JSON.stringify({ error: "Server missing ADMIN_USERNAME / ADMIN_PASSWORD / SESSION_SECRET." }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }

  let body;
  try { body = await request.json(); } catch { return badRequest("Body must be JSON."); }
  const username = typeof body?.username === "string" ? body.username : "";
  const password = typeof body?.password === "string" ? body.password : "";

  // Constant-time compare so timing differences don't leak whether the username
  // alone matched.
  const userOk = constantTimeStringEqual(username, env.ADMIN_USERNAME);
  const passOk = constantTimeStringEqual(password, env.ADMIN_PASSWORD);
  if (!userOk || !passOk) {
    return new Response(
      JSON.stringify({ error: "Invalid credentials." }),
      { status: 401, headers: { "content-type": "application/json" } },
    );
  }

  const now = Date.now();
  const payload = { iat: now, exp: now + SESSION_TTL_SECONDS * 1000 };
  const payloadB64 = btoa(JSON.stringify(payload));
  const sig = await hmac(payloadB64, env.SESSION_SECRET);
  const token = `${payloadB64}.${b64(sig)}`;

  const cookie = [
    `session=${token}`,
    "HttpOnly",
    "Secure",
    "SameSite=Strict",
    "Path=/",
    `Max-Age=${SESSION_TTL_SECONDS}`,
  ].join("; ");

  return new Response(
    JSON.stringify({ ok: true }),
    {
      status: 200,
      headers: {
        "content-type": "application/json",
        "set-cookie": cookie,
      },
    },
  );
};
