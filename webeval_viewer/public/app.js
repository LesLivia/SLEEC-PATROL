// SLEEC-PATROL answers viewer - the manager's read-only view of the evaluation's answers.
// As UI_static_viewer: three pages - login, list (every submission), detail (one
// submission, with a picker to walk through every answer). Each answer is shown beside the
// rule page it is about, built from the same data as the study site (public/data).

const PAGES = ["page-login", "page-list", "page-detail"];

const S = {
  submissions: [],           // list-page summaries from /api/submissions
  currentSession: null,      // full submission JSON for the detail page
  flatEvals: [],             // its answers in order: [{ env, rule, form, submitted_at }]
  currentEvalIndex: 0,       // which answer is being rendered
  currentEnv: null,          // that answer's environment, for the Background sheet
  // The study's data, as the participants saw it: the manifest, texts keyed by env id,
  // rule fragments by `${env}/${rule}`.
  manifest: null,
  texts: {},
  fragments: {},
};

// =====================================================================
// Page navigation
// =====================================================================
function showPage(id) {
  PAGES.forEach((p) => { document.getElementById(p).hidden = p !== id; });
  // The detail page is wider: the answer sits beside the rule.
  document.body.classList.toggle("wide", id === "page-detail");
  // What the header offers on each page.
  document.getElementById("nav-back").hidden = id !== "page-detail";
  document.getElementById("btn-background").hidden = id !== "page-detail";
  document.getElementById("picker-wrap").hidden = id !== "page-detail";
  document.getElementById("btn-refresh").hidden = id !== "page-list";
  document.getElementById("btn-logout").hidden = id !== "page-list";
  window.scrollTo({ top: 0, behavior: "instant" });
}

function setHeader({ step = "", title, sub = "" }) {
  const stepEl = document.getElementById("top-step");
  stepEl.textContent = step;
  stepEl.hidden = !step;
  document.getElementById("top-title").textContent = title;
  const subEl = document.getElementById("top-sub");
  subEl.textContent = sub;
  subEl.hidden = !sub;
}

function envOf(id) {
  return S.manifest ? S.manifest.environments.find((e) => e.id === id) : null;
}

function envTitle(id) {
  const env = envOf(id);
  return env ? env.title : id;
}

function fmtTime(s) {
  if (!s) return "—";
  try { return new Date(s).toLocaleString(); } catch { return s; }
}

// =====================================================================
// Auth
// =====================================================================
async function doLogin() {
  const username = document.getElementById("login-username").value.trim();
  const password = document.getElementById("login-password").value;
  const errEl = document.getElementById("login-error");
  errEl.hidden = true;
  if (!username || !password) {
    errEl.textContent = "Username and password are required.";
    errEl.hidden = false;
    return;
  }
  const btn = document.getElementById("login-submit");
  btn.disabled = true;
  btn.textContent = "Logging in…";
  try {
    const r = await fetch("/api/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "same-origin",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || "Login failed.");
    document.getElementById("login-password").value = "";
    await goToList();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Log in";
  }
}

async function doLogout() {
  try { await fetch("/api/logout", { method: "POST", credentials: "same-origin" }); }
  catch (_) {}
  showLoginPage();
}

function showLoginPage() {
  setHeader({ title: "SLEEC-PATROL: Answers viewer" });
  showPage("page-login");
}

// Wrap fetch so any 401 routes the manager to the login page.
async function apiFetch(url, opts = {}) {
  const r = await fetch(url, { credentials: "same-origin", ...opts });
  if (r.status === 401) {
    showLoginPage();
    throw new Error("Session expired. Please log in again.");
  }
  return r;
}

// =====================================================================
// List page
// =====================================================================
async function goToList() {
  // Used by: post-login navigation and the Back button on the detail page. Submissions
  // are reloaded so any new ones since the detail was opened show up.
  S.currentSession = null;
  S.flatEvals = [];
  setHeader({ step: "SLEEC-PATROL diagnosis evaluation", title: "Submissions" });
  showPage("page-list");
  await loadSubmissions();
}

async function loadSubmissions() {
  const status = document.getElementById("list-status");
  const errEl = document.getElementById("list-error");
  const tableWrap = document.getElementById("list-table-wrap");
  const emptyEl = document.getElementById("list-empty");
  status.hidden = false;
  errEl.hidden = true;
  tableWrap.hidden = true;
  emptyEl.hidden = true;
  try {
    const r = await apiFetch("/api/submissions");
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Failed to load.");
    S.submissions = data.submissions || [];
    showList();
  } catch (err) {
    status.hidden = true;
    errEl.textContent = err.message;
    errEl.hidden = false;
  }
}

function showList() {
  document.getElementById("list-status").hidden = true;
  if (S.submissions.length === 0) {
    document.getElementById("list-empty").hidden = false;
    return;
  }
  renderList();
  document.getElementById("list-table-wrap").hidden = false;
}

function renderList() {
  const rows = document.getElementById("list-rows");
  rows.replaceChildren();
  for (const sub of S.submissions) {
    const tr = document.createElement("tr");
    const cells = [
      [sub.sessionId || "—", "mono"],
      [sub.expertise || "—", ""],
      [(sub.environments_done || []).map(envTitle).join(", ") || "—", ""],
      [String(sub.evaluations_count ?? 0), ""],
      [fmtTime(sub.last_submitted_at), ""],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    const view = document.createElement("button");
    view.type = "button";
    view.className = "btn is-small";
    view.textContent = "View";
    view.addEventListener("click", () => openSubmission(sub.sessionId));
    td.appendChild(view);
    tr.appendChild(td);
    rows.appendChild(tr);
  }
}

// =====================================================================
// Detail page
// =====================================================================
async function openSubmission(sessionId) {
  try {
    const r = await apiFetch(`/api/submission?id=${encodeURIComponent(sessionId)}`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Failed to load submission.");
    S.currentSession = data;
    S.currentEvalIndex = 0;

    // Flatten the per-environment submissions into a single ordered list of
    // [{ env, rule, form, submitted_at }] so the picker is linear.
    const flat = [];
    for (const s of (data.submissions || [])) {
      for (const ev of (s.evaluations || [])) {
        flat.push({ env: ev.env, rule: ev.rule, form: ev.form || {},
                    submitted_at: s.submitted_at });
      }
    }
    S.flatEvals = flat;

    // Submissions are anonymous: the sessionId stands in the title slot.
    setHeader({
      step: "Submission", title: sessionId,
      sub: `${data.expertise || "—"} · first ${fmtTime(data.first_submitted_at)} · `
           + `last ${fmtTime(data.last_submitted_at)}`,
    });
    const picker = document.getElementById("detail-eval-picker");
    picker.replaceChildren(...flat.map((ev, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `${envTitle(ev.env)} · ${ev.rule}`;
      return opt;
    }));
    picker.value = "0";

    showPage("page-detail");
    await renderSelectedEvaluation();
  } catch (err) {
    alert(err.message);
  }
}

async function renderSelectedEvaluation() {
  const picker = document.getElementById("detail-eval-picker");
  S.currentEvalIndex = Number(picker.value) || 0;
  const ev = S.flatEvals[S.currentEvalIndex];
  if (!ev) return;
  S.currentEnv = ev.env;

  // The rule page, as the participant saw it: in plain words, then its panel.
  const env = envOf(ev.env);
  const known = env && env.rules.includes(ev.rule);
  const words = known ? (env.descriptions || {})[ev.rule] || "" : "";
  document.getElementById("rule-words").textContent = words;
  document.getElementById("rule-words-card").hidden = !words;
  const missing = document.getElementById("rule-missing");
  const panel = document.getElementById("rule-panel");
  if (known) {
    missing.hidden = true;
    panel.innerHTML = await loadFragment(ev.env, ev.rule);   // built from our own data
    wireReport(panel);
  } else {
    missing.textContent = `${envTitle(ev.env)} rule ${ev.rule} is not in the viewer's data, `
      + "so its page cannot be shown. The answer is shown as it was given.";
    missing.hidden = false;
    panel.replaceChildren();
  }

  fillReadonlyEvalForm(ev.form);
  document.getElementById("eval-submitted-at").textContent =
    ev.submitted_at ? `Submitted at ${fmtTime(ev.submitted_at)}` : "";
}

function fillReadonlyEvalForm(form) {
  document.querySelectorAll('input[name="eval-helps"]').forEach((r) => {
    r.checked = !!form && r.value === form.helps;
  });
  const clearer = document.getElementById("eval-clearer");
  const text = form && form.clearer ? form.clearer : "";
  clearer.textContent = text || "— (none provided)";
  clearer.classList.toggle("empty", !text);
}

// =====================================================================
// Data loaders (as the study site's)
// =====================================================================
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url} (${r.status})`);
  return r.text();
}
async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url} (${r.status})`);
  return r.json();
}

async function loadTexts(envId) {
  if (S.texts[envId]) return S.texts[envId];
  const env = envOf(envId);
  const base = `/data/${envId}`;
  const [background, note] = await Promise.all([
    fetchText(`${base}/background.txt`),
    env && env.note ? fetchText(`${base}/note.txt`) : Promise.resolve(""),
  ]);
  S.texts[envId] = { background, note: note.trim() };
  return S.texts[envId];
}

async function loadFragment(envId, rule) {
  const key = `${envId}/${rule}`;
  if (!(key in S.fragments)) {
    S.fragments[key] = await fetchText(`/data/${envId}/rules/${rule}.html`);
  }
  return S.fragments[key];
}

// =====================================================================
// Renderers (as the study site's)
// =====================================================================
// A background text: paragraphs separated by blank lines; a paragraph whose lines are
// all indented is a block kept as it is written (a list of actions or concepts).
function renderText(el, text) {
  el.replaceChildren();
  for (const para of text.trim().split(/\n\s*\n/)) {
    const lines = para.split("\n");
    if (lines.every((l) => l.startsWith("    "))) {
      const pre = document.createElement("pre");
      pre.className = "form";
      pre.textContent = lines.map((l) => l.slice(4)).join("\n");
      el.appendChild(pre);
    } else {
      const p = document.createElement("p");
      p.textContent = para.replace(/\s*\n\s*/g, " ");
      el.appendChild(p);
    }
  }
}

// The report's own behaviour, as in the script of webapp/templates/results.html: the
// constants fold out under their rule, a Diagnosis button opens its sheet, and a sheet's
// Close button closes it. A rule's button sits on the summary that folds its panel, so
// its click must not reach it.
function wireReport(root) {
  root.querySelectorAll("[data-consts]").forEach((button) => {
    const list = button.closest(".spec").querySelector(".consts");
    button.addEventListener("click", () => {
      list.hidden = !list.hidden;
      button.textContent = list.hidden ? "SLEEC rule measure constraints"
        : "Hide SLEEC rule measure constraints";
    });
  });
  root.querySelectorAll("[data-dialog]").forEach((button) => {
    const dialog = document.getElementById(button.getAttribute("data-dialog"));
    if (!dialog) return;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      dialog.showModal();
    });
  });
  root.querySelectorAll("dialog [data-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
}

// =====================================================================
// Background sheet - the environment of the answer shown
// =====================================================================
async function openBackground() {
  if (!S.currentEnv) return;
  const texts = await loadTexts(S.currentEnv);
  document.getElementById("background-modal-title").textContent =
    `Background — ${envTitle(S.currentEnv)}`;
  renderText(document.getElementById("background-modal-body"), texts.background);
  const note = document.getElementById("background-modal-note");
  note.textContent = texts.note;
  note.hidden = !texts.note;
  document.getElementById("background-modal").showModal();
}

// =====================================================================
// Boot
// =====================================================================
(async function boot() {
  // The page's own sheet closes from its Close button; Esc closes it natively.
  document.querySelectorAll("dialog [data-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  try {
    S.manifest = await fetchJson("/data/manifest.json");
  } catch (_) {
    S.manifest = null;     // answers still list and show; rule pages cannot
  }
  // Probe the auth state with a cheap call. If the session cookie is valid the list page
  // loads immediately; otherwise the login page is shown.
  try {
    const r = await fetch("/api/submissions", { credentials: "same-origin" });
    if (r.ok) {
      const data = await r.json();
      S.submissions = data.submissions || [];
      setHeader({ step: "SLEEC-PATROL diagnosis evaluation", title: "Submissions" });
      showPage("page-list");
      showList();
      return;
    }
  } catch (_) { /* fall through to login */ }
  showLoginPage();
})();
