// POST /api/submit
//
// One submit per environment. The first submit in a session allocates a unique session
// id (with R2 collision check); subsequent submits attach to the same session and either
// overwrite the previous entry for that environment or append a new one.
//
// As UI_static's submit.js, without its coverage counters: every participant sees the
// same rules, so there is nothing to balance and no KV namespace.
//
// Binding (see wrangler.toml):
//   SUBMISSIONS  - R2 bucket; one JSON object per session at submissions/<id>.json
//
// Request body (JSON):
//   {
//     sessionId:   string | null,       // null on the first submit
//     expertise:   string,
//     env:         string,              // which environment is being submitted
//     started_at:  ISO8601 string,
//     evaluations: [ { env, rule, form: { helps: "yes" | "no", clearer } }, ... ],
//   }
//
// Response: { ok: true, sessionId: "<assigned-or-echoed-uuid>" }
//
// Persisted file at r2://<SUBMISSIONS>/submissions/<sessionId>.json:
//   {
//     sessionId, expertise,
//     started_at, first_submitted_at, last_submitted_at,
//     submissions: [ { submitted_at, env, evaluations: [...] }, ... ],
//   }
// No evaluator-identifying field is stored - submissions are anonymous, keyed only by
// sessionId.

const ID_RETRY_LIMIT = 5;

function badRequest(msg) {
  return new Response(
    JSON.stringify({ error: msg }),
    { status: 400, headers: { "content-type": "application/json" } },
  );
}

function serverError(msg) {
  return new Response(
    JSON.stringify({ error: msg }),
    { status: 500, headers: { "content-type": "application/json" } },
  );
}

async function allocateUniqueId(env) {
  for (let attempt = 0; attempt < ID_RETRY_LIMIT; attempt++) {
    const id = crypto.randomUUID();
    const existing = await env.SUBMISSIONS.head(`submissions/${id}.json`);
    if (existing === null) return id;
  }
  throw new Error(`Could not allocate a unique session id after ${ID_RETRY_LIMIT} attempts.`);
}

export const onRequestPost = async ({ env, request }) => {
  let body;
  try {
    body = await request.json();
  } catch {
    return badRequest("Body must be valid JSON.");
  }

  const { sessionId: providedId, expertise, env: studyEnv, evaluations } = body || {};
  if (typeof expertise !== "string" || !expertise.trim()) return badRequest("Missing expertise.");
  if (typeof studyEnv !== "string" || !studyEnv) return badRequest("Missing env.");
  if (!Array.isArray(evaluations) || evaluations.length === 0) {
    return badRequest("evaluations must be a non-empty array.");
  }
  for (const ev of evaluations) {
    if (!ev || typeof ev.env !== "string" || typeof ev.rule !== "string") {
      return badRequest("Each evaluation needs env and rule.");
    }
  }

  const submittedAt = new Date().toISOString();
  const newEntry = { submitted_at: submittedAt, env: studyEnv, evaluations };

  let sessionId;
  let payload;

  if (providedId && typeof providedId === "string") {
    // Subsequent submit: read the existing record, mutate, write back.
    const existing = await env.SUBMISSIONS.get(`submissions/${providedId}.json`);
    if (existing === null) {
      // Unknown sessionId - refuse so a client can't pollute R2 with a chosen id.
      return badRequest("Unknown sessionId.");
    }
    try {
      payload = JSON.parse(await existing.text());
    } catch {
      return serverError("Existing submission record is not valid JSON.");
    }
    payload.submissions = Array.isArray(payload.submissions) ? payload.submissions : [];
    payload.submissions = payload.submissions.filter((s) => s.env !== studyEnv);
    payload.submissions.push(newEntry);
    payload.last_submitted_at = submittedAt;
    sessionId = providedId;
  } else {
    // First submit: allocate a unique id and build the record from scratch.
    try {
      sessionId = await allocateUniqueId(env);
    } catch (err) {
      return serverError(err.message || String(err));
    }
    payload = {
      sessionId,
      expertise,
      started_at: body.started_at || submittedAt,
      first_submitted_at: submittedAt,
      last_submitted_at: submittedAt,
      submissions: [newEntry],
    };
  }

  try {
    await env.SUBMISSIONS.put(
      `submissions/${sessionId}.json`,
      JSON.stringify(payload, null, 2),
      { httpMetadata: { contentType: "application/json" } },
    );
  } catch (err) {
    return serverError(`Failed to save submission: ${err.message || err}`);
  }

  return new Response(
    JSON.stringify({ ok: true, sessionId }),
    { headers: { "content-type": "application/json" } },
  );
};
