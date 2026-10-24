// GET /api/submission?id=<sessionId>
//
// Fetches the full JSON document for one submission so the detail view can
// render every evaluation the evaluator made.
//
// Bindings (see wrangler.toml):
//   SUBMISSIONS  — R2 bucket (same one the writer writes to)

function badRequest(msg) {
  return new Response(
    JSON.stringify({ error: msg }),
    { status: 400, headers: { "content-type": "application/json" } },
  );
}

export const onRequestGet = async ({ env, request }) => {
  const url = new URL(request.url);
  const id = url.searchParams.get("id");
  if (!id) return badRequest("Missing ?id=<sessionId>.");
  // Reject anything with separators so a caller can't traverse to a different
  // key (e.g. "../something"). UUIDs only contain hex and dashes.
  if (!/^[A-Za-z0-9._-]+$/.test(id)) return badRequest("Invalid sessionId.");

  try {
    const file = await env.SUBMISSIONS.get(`submissions/${id}.json`);
    if (file === null) {
      return new Response(
        JSON.stringify({ error: "Submission not found." }),
        { status: 404, headers: { "content-type": "application/json" } },
      );
    }
    return new Response(await file.text(), {
      headers: { "content-type": "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err.message || String(err) }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
};
