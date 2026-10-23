// SLEEC-PATROL diagnosis evaluation - static site for Cloudflare Pages.
// Flow: entry -> instructions -> per environment (intro + one page per violated rule)
// -> done page. As in UI_static: state persists in sessionStorage so a refresh resumes
// where the participant left off, and each environment is submitted on its own.

const STORAGE_KEY = "sleec_patrol_eval_state";
const PAGES = ["page-entry", "page-instructions", "page-intro", "page-rule", "page-done"];

const S = {
  // Assigned by /api/submit after the first successful submission - until then null.
  sessionId: null,
  expertise: "",
  startedAt: null,
  // The build the answers belong to (manifest.built); a rebuilt site starts afresh.
  built: null,
  // [{env, rule, form: {...}} | null] - index aligned with the rule steps' evalIndex.
  evaluations: [],
  // Linear list of steps the participant walks through, built from the manifest.
  steps: [],
  // Current index into `steps`; -1 is the instructions page.
  stepIndex: -1,
  // Every environment submitted: the done page is shown.
  finished: false,
  // Loaded data: the manifest, texts keyed by env id, rule fragments by `${env}/${rule}`.
  manifest: null,
  texts: {},
  fragments: {},
};

// =====================================================================
// Page navigation
// =====================================================================
function showPage(id) {
  PAGES.forEach((p) => { document.getElementById(p).hidden = p !== id; });
  // A rule page is wider: the form sits beside the rule.
  document.body.classList.toggle("wide", id === "page-rule");
  window.scrollTo({ top: 0, behavior: "instant" });
}

// What the header shows: the step line, the title, Back, and the two references.
function setHeader({ step = "", title, back = false, actions = false, background = false }) {
  const stepEl = document.getElementById("top-step");
  stepEl.textContent = step;
  stepEl.hidden = !step;
  document.getElementById("top-title").textContent = title;
  document.getElementById("nav-back").hidden = !back;
  document.getElementById("top-actions").hidden = !actions;
  document.getElementById("btn-background").hidden = !background;
}

function envOf(id) {
  return S.manifest.environments.find((e) => e.id === id);
}

function buildSteps(manifest) {
  // For each environment: its intro page first, then one page per violated rule.
  const steps = [];
  let evalIndex = 0;
  manifest.environments.forEach((env, envIdx) => {
    steps.push({ type: "intro", env: env.id, envIdx });
    env.rules.forEach((rule, slot) => {
      steps.push({ type: "rule", env: env.id, envIdx, rule, slot, evalIndex });
      evalIndex += 1;
    });
  });
  return steps;
}

function ruleCount() {
  return S.steps.filter((s) => s.type === "rule").length;
}

// =====================================================================
// Persistence
// =====================================================================
function saveState() {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      sessionId: S.sessionId,
      expertise: S.expertise,
      startedAt: S.startedAt,
      built: S.built,
      evaluations: S.evaluations,
      stepIndex: S.stepIndex,
      finished: S.finished,
    }));
  } catch (_) { /* storage full / disabled - skip silently */ }
}

function loadState() {
  let raw;
  try { raw = sessionStorage.getItem(STORAGE_KEY); } catch (_) { return false; }
  if (!raw) return false;
  let d;
  try { d = JSON.parse(raw); } catch (_) {
    clearState();
    return false;
  }
  // Answers saved against another build may point at rules that are no longer there.
  if (!d.expertise || d.built !== S.manifest.built) {
    clearState();
    return false;
  }
  S.sessionId = d.sessionId || null;
  S.expertise = d.expertise;
  S.startedAt = d.startedAt || null;
  S.built = d.built;
  S.evaluations = Array.isArray(d.evaluations) && d.evaluations.length === ruleCount()
    ? d.evaluations : S.steps.filter((s) => s.type === "rule").map(() => null);
  S.stepIndex = typeof d.stepIndex === "number" ? d.stepIndex : -1;
  S.finished = d.finished === true;
  return true;
}

function clearState() {
  try { sessionStorage.removeItem(STORAGE_KEY); } catch (_) {}
  S.sessionId = null;
  S.expertise = "";
  S.startedAt = null;
  S.built = null;
  S.evaluations = [];
  S.stepIndex = -1;
  S.finished = false;
}

// =====================================================================
// Entry page
// =====================================================================
function startSession() {
  if (!S.manifest) return;           // the study did not load: the page says so
  const expertise = document.getElementById("entry-expertise").value.trim();
  const errEl = document.getElementById("entry-error");
  errEl.hidden = true;
  if (!expertise) {
    errEl.textContent = "Please describe your expertise.";
    errEl.hidden = false;
    return;
  }
  S.sessionId = null;
  S.expertise = expertise;
  S.startedAt = new Date().toISOString();
  S.built = S.manifest.built;
  S.evaluations = S.steps.filter((s) => s.type === "rule").map(() => null);
  S.stepIndex = -1;
  S.finished = false;
  saveState();
  renderCurrentStep();
}

// =====================================================================
// Data loaders
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
    env.note ? fetchText(`${base}/note.txt`) : Promise.resolve(""),
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
// Renderers
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

function renderPictures(env) {
  const intro = document.getElementById("intro-pictures-intro");
  intro.textContent = env.pictures.intro || "";
  intro.hidden = !env.pictures.intro;
  const box = document.getElementById("intro-pictures");
  box.replaceChildren();
  for (const item of env.pictures.items) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = item.src;
    img.alt = item.caption;
    img.title = "Click to enlarge";
    img.addEventListener("click", () => openImageLightbox(item.src, item.caption));
    const cap = document.createElement("figcaption");
    cap.textContent = item.caption;
    fig.append(img, cap);
    box.appendChild(fig);
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
// Intro page
// =====================================================================
async function renderIntroStep() {
  const step = S.steps[S.stepIndex];
  const env = envOf(step.env);
  const texts = await loadTexts(step.env);

  setHeader({
    step: `Environment ${step.envIdx + 1} of ${S.manifest.environments.length}`,
    title: env.title, back: true, actions: true, background: false,
  });
  renderText(document.getElementById("intro-background"), texts.background);
  const note = document.getElementById("intro-note");
  note.textContent = texts.note;
  note.hidden = !texts.note;
  renderPictures(env);
  showPage("page-intro");
}

function goNextFromIntro(event) {
  return navigate(async () => {
    S.stepIndex += 1;
    saveState();
    await renderCurrentStep();
  }, event);
}

// "Begin" on the instructions page jumps into the first step (the first intro).
function beginEvaluation(event) {
  return navigate(async () => {
    S.stepIndex = 0;
    saveState();
    await renderCurrentStep();
  }, event);
}

function showInstructionsPage() {
  setHeader({ step: "Before you begin", title: "Instructions" });
  showPage("page-instructions");
}

// =====================================================================
// Rule page
// =====================================================================
function isLastRuleOfEnv(step) {
  return step.slot === envOf(step.env).rules.length - 1;
}

function isFinalRule(step) {
  return step.envIdx === S.manifest.environments.length - 1 && isLastRuleOfEnv(step);
}

async function renderRuleStep() {
  const step = S.steps[S.stepIndex];
  const env = envOf(step.env);
  const fragment = await loadFragment(step.env, step.rule);

  setHeader({
    step: `${env.title} · rule ${step.slot + 1} of ${env.rules.length}`,
    title: `Rule ${step.rule}`, back: true, actions: true, background: true,
  });
  // The rule in plain words, above the rule as SLEEC-PATROL reports it.
  const words = (env.descriptions || {})[step.rule] || "";
  document.getElementById("rule-words").textContent = words;
  document.getElementById("rule-words-card").hidden = !words;

  const panel = document.getElementById("rule-panel");
  panel.innerHTML = fragment;   // built by scripts/build_data.py from our own data
  wireReport(panel);

  // Reset the form and restore earlier answers if the participant came back.
  resetEvalForm();
  const prior = S.evaluations[step.evalIndex];
  if (prior && prior.form) populateEvalForm(prior.form);

  // Each environment is submitted with its last rule; the final one closes the session.
  document.getElementById("btn-next").textContent =
    isFinalRule(step) ? "Submit" : isLastRuleOfEnv(step) ? "Submit & Continue" : "Next";

  showPage("page-rule");
}

// =====================================================================
// Evaluation form
// =====================================================================
function onHelpsChange() {
  const v = (document.querySelector('input[name="eval-helps"]:checked') || {}).value || "";
  const clearer = document.getElementById("eval-clearer");
  const required = document.getElementById("eval-clearer-required");
  if (v === "yes") {
    clearer.disabled = true;
    clearer.value = "";
    clearer.placeholder = "Not required.";
    required.hidden = true;
  } else {
    clearer.disabled = false;
    clearer.placeholder = "Required if you answer no.";
    required.hidden = false;
  }
}

function resetEvalForm() {
  document.querySelectorAll('input[name="eval-helps"]').forEach((r) => { r.checked = false; });
  const clearer = document.getElementById("eval-clearer");
  clearer.value = "";
  clearer.disabled = false;
  clearer.placeholder = "Required if you answer no.";
  document.getElementById("eval-clearer-required").hidden = false;
  document.getElementById("eval-error").hidden = true;
}

function populateEvalForm(form) {
  const r = document.querySelector(`input[name="eval-helps"][value="${form.helps}"]`);
  if (r) {
    r.checked = true;
    onHelpsChange();
  }
  if (form.clearer) document.getElementById("eval-clearer").value = form.clearer;
}

function readEvalForm() {
  const helps = (document.querySelector('input[name="eval-helps"]:checked') || {}).value || "";
  const clearer = helps === "no" ? document.getElementById("eval-clearer").value.trim() : "";
  return { helps, clearer };
}

function validateEvalForm(form) {
  if (!form.helps) return "Please answer whether the diagnosis helps you.";
  if (form.helps === "no" && !form.clearer) return "Please say what could be clearer.";
  return null;
}

// =====================================================================
// Navigation
// =====================================================================
// One page change at a time. A step moves on at once, but its page shows only once its
// data has loaded, and an environment's last rule waits for its submission; a press of
// Begin, Continue, Next or Back in between is ignored. So is the second click of a
// double-click, which lands on the next page when that page was already loaded. Otherwise
// a double-click moves on twice: it skips a page, or stores one rule's answer for the next
// rule as well.
let busy = false;

async function navigate(change, event) {
  if (busy || (event && event.detail > 1)) return;
  busy = true;
  try {
    await change();
  } finally {
    busy = false;
  }
}

// Back is pure navigation - it never writes to S.evaluations. Only Next/Submit commit
// form data, so there is exactly one writer for evaluations. From the first step, Back
// returns to the instructions page.
function goBack(event) {
  return navigate(async () => {
    if (S.stepIndex <= 0) {
      S.stepIndex = -1;
      saveState();
      showInstructionsPage();
      return;
    }
    S.stepIndex -= 1;
    saveState();
    await renderCurrentStep();
  }, event);
}

function goNextFromRule(event) {
  return navigate(async () => {
    const step = S.steps[S.stepIndex];
    const form = readEvalForm();
    const err = validateEvalForm(form);
    const errEl = document.getElementById("eval-error");
    if (err) {
      errEl.textContent = err;
      errEl.hidden = false;
      return;
    }
    S.evaluations[step.evalIndex] = { env: step.env, rule: step.rule, form };
    saveState();

    if (isLastRuleOfEnv(step)) {
      const ok = await submitEnv(step.env);
      if (!ok) return;               // failed -> stay on this rule, error shown
      if (isFinalRule(step)) {
        S.finished = true;
        saveState();
        showDonePage();
        return;
      }
    }
    S.stepIndex += 1;
    saveState();
    await renderCurrentStep();
  }, event);
}

async function renderCurrentStep() {
  if (!S.expertise) return showEntryPage();
  if (S.finished) return showDonePage();
  if (S.stepIndex < 0) return showInstructionsPage();
  const step = S.steps[S.stepIndex];
  if (!step) return showDonePage();
  if (step.type === "intro") return renderIntroStep();
  if (step.type === "rule") return renderRuleStep();
}

// =====================================================================
// Submit (one POST per environment)
// =====================================================================
async function submitEnv(envId) {
  const envEvals = S.evaluations.filter((e) => e && e.env === envId);
  const btn = document.getElementById("btn-next");
  const errEl = document.getElementById("eval-error");
  const label = btn.textContent;

  btn.disabled = true;
  btn.textContent = "Submitting…";
  errEl.hidden = true;
  try {
    const r = await fetch("/api/submit", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sessionId: S.sessionId || null,
        expertise: S.expertise,
        env: envId,
        started_at: S.startedAt,
        evaluations: envEvals,
      }),
    });
    let data = {};
    try { data = await r.json(); } catch (_) { /* not JSON: said below */ }
    if (!r.ok) throw new Error(data.error || `Submission failed (${r.status}).`);
    if (data.sessionId) S.sessionId = data.sessionId;
    saveState();
    return true;
  } catch (err) {
    errEl.textContent = err.message || String(err);
    errEl.hidden = false;
    return false;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function showEntryPage() {
  setHeader({ title: "SLEEC-PATROL: Diagnosis evaluation" });
  showPage("page-entry");
}

function showDonePage() {
  setHeader({ title: "SLEEC-PATROL: Diagnosis evaluation" });
  document.getElementById("done-session-id").textContent =
    S.sessionId ? `Reference: ${S.sessionId}` : "";
  showPage("page-done");
}

function startOver() {
  clearState();
  document.getElementById("entry-expertise").value = "";
  document.getElementById("entry-error").hidden = true;
  showEntryPage();
}

// =====================================================================
// Sheets: instructions, background, picture
// =====================================================================
// The instructions sheet is cloned from the instructions page, so the text lives in one
// place and any edit shows up in both.
function openInstructions() {
  const src = document.getElementById("instructions-content-source");
  document.getElementById("modal-instructions-body").innerHTML = src.innerHTML;
  document.getElementById("instructions-modal").showModal();
}

// The background sheet shows the current environment's background and note.
async function openBackground() {
  const step = S.steps[S.stepIndex];
  if (!step) return;
  const texts = await loadTexts(step.env);
  document.getElementById("background-modal-title").textContent =
    `Background — ${envOf(step.env).title}`;
  renderText(document.getElementById("background-modal-body"), texts.background);
  const note = document.getElementById("background-modal-note");
  note.textContent = texts.note;
  note.hidden = !texts.note;
  document.getElementById("background-modal").showModal();
}

function openImageLightbox(src, alt) {
  const img = document.getElementById("image-lightbox-img");
  img.src = src;
  img.alt = alt || "";
  document.getElementById("image-lightbox").showModal();
}

// =====================================================================
// Boot
// =====================================================================
(async function boot() {
  // The page's own sheets close from their Close button; Esc closes any sheet natively.
  document.querySelectorAll("dialog [data-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  document.getElementById("entry-expertise").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startSession();
  });

  try {
    S.manifest = await fetchJson("/data/manifest.json");
  } catch (err) {
    document.getElementById("entry-error").textContent =
      "The study could not be loaded. Please reload the page.";
    document.getElementById("entry-error").hidden = false;
    document.getElementById("entry-submit").disabled = true;
    showEntryPage();
    return;
  }
  S.steps = buildSteps(S.manifest);
  document.querySelectorAll("[data-rule-count]").forEach((el) => {
    const env = envOf(el.getAttribute("data-rule-count"));
    const n = env ? env.rules.length : 0;
    el.textContent = `${n} rule${n === 1 ? "" : "s"}`;
  });

  if (loadState()) {
    renderCurrentStep().catch((err) => {
      // If the saved place cannot be shown (e.g. the data was rebuilt), restart fresh.
      console.warn("Failed to resume session:", err);
      clearState();
      showEntryPage();
    });
  } else {
    showEntryPage();
  }
})();
