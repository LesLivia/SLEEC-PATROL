// POST /api/logout
//
// Clears the session cookie by issuing a new cookie with Max-Age=0. After this
// the browser will not send any session cookie on subsequent requests, so the
// middleware will return 401 and the client will redirect to the login page.

export const onRequestPost = async () => {
  const cookie = "session=; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=0";
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
