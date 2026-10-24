// GET /api/submissions
//
// Lists every R2 object under `submissions/`, fetches each one, and returns
// only the summary fields (no eval form contents). The detail view fetches
// the full object separately via /api/submission?id=<sessionId>.
//
// Bindings (see wrangler.toml):
//   SUBMISSIONS  — R2 bucket (same one the writer writes to)
//
// Response: array of summaries, sorted by last_submitted_at descending.

export const onRequestGet = async ({ env }) => {
  try {
    let cursor = undefined;
    const summaries = [];
    do {
      const page = await env.SUBMISSIONS.list({ prefix: "submissions/", cursor });
      for (const obj of page.objects) {
        const file = await env.SUBMISSIONS.get(obj.key);
        if (!file) continue;
        let body;
        try { body = JSON.parse(await file.text()); } catch { continue; }
        const subs = Array.isArray(body.submissions) ? body.submissions : [];
        summaries.push({
          sessionId:           body.sessionId || obj.key.replace(/^submissions\//, "").replace(/\.json$/, ""),
          expertise:           body.expertise || "",
          started_at:          body.started_at || null,
          first_submitted_at:  body.first_submitted_at || null,
          last_submitted_at:   body.last_submitted_at || null,
          environments_done:   subs.map((s) => s.env),
          submissions_count:   subs.length,
          evaluations_count:   subs.reduce((n, s) => n + (Array.isArray(s.evaluations) ? s.evaluations.length : 0), 0),
        });
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);

    summaries.sort((a, b) => {
      const ta = a.last_submitted_at ? Date.parse(a.last_submitted_at) : 0;
      const tb = b.last_submitted_at ? Date.parse(b.last_submitted_at) : 0;
      return tb - ta;
    });

    return new Response(
      JSON.stringify({ submissions: summaries }),
      { headers: { "content-type": "application/json" } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err.message || String(err) }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
};
