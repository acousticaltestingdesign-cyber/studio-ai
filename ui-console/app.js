"use strict";

const THEME_IDS = new Set(["dark_studio", "cyberpunk_synth", "retro_analog"]);
const THEME_LABELS = {
  dark_studio: "Dark Studio",
  cyberpunk_synth: "Cyberpunk Synth",
  retro_analog: "Retro Analog",
};
const THEME_STORAGE_KEY = "studio-ai-theme";
const LIMITS = { infer: 8000, orchestrate: 4000 };
const MODEL = "grok-4.7";
const API_BASE = resolveApiBase();
const SESSION_ID = createSessionId();

const els = {
  build: document.getElementById("build-label"),
  themeNote: document.getElementById("theme-note"),
  routeKicker: document.getElementById("route-kicker"),
  transcript: document.getElementById("transcript"),
  placeholder: document.getElementById("transcript-placeholder"),
  form: document.getElementById("command-form"),
  input: document.getElementById("command-input"),
  manualsField: document.getElementById("manuals-field"),
  useManuals: document.getElementById("use-manuals"),
  count: document.getElementById("char-count"),
  send: document.querySelector(".send"),
  dropZone: document.getElementById("drop-zone"),
  manualInput: document.getElementById("manual-input"),
  chunkCount: document.getElementById("chunk-count"),
  dropLog: document.getElementById("drop-log"),
  desktopLamp: document.getElementById("desktop-lamp"),
  desktopAddress: document.getElementById("desktop-address"),
  desktopState: document.getElementById("desktop-state"),
  brainLamp: document.getElementById("brain-lamp"),
  brainAddress: document.getElementById("brain-address"),
  brainState: document.getElementById("brain-state"),
  clock: document.getElementById("clock"),
  themeGroup: document.querySelector(".themes"),
  routeGroup: document.querySelector(".route"),
};

const state = {
  theme: document.documentElement.dataset.theme || "dark_studio",
  route: "infer",
  themeLock: false,
  sending: false,
  uploads: 0,
  desktopProbe: false,
  desktop: { ip: "192.168.0.103", port: 8080 },
};

boot();

function boot() {
  els.brainAddress.textContent = hostOf(API_BASE);
  applyTheme(state.theme);
  bind();
  updateCount();
  fitInput();
  tickClock();
  setInterval(tickClock, 1000);
  loadSettings();
  setInterval(checkBrain, 12000);
  setInterval(checkDesktop, 8000);
}

function bind() {
  els.themeGroup.querySelectorAll("button[data-theme-choice]").forEach((button) => {
    button.addEventListener("click", () => chooseTheme(button.getAttribute("data-theme-choice")));
  });
  bindRadio(els.themeGroup, "data-theme-choice", chooseTheme);

  els.routeGroup.querySelectorAll("button[data-route]").forEach((button) => {
    button.addEventListener("click", () => setRoute(button.getAttribute("data-route")));
  });
  bindRadio(els.routeGroup, "data-route", setRoute);

  els.form.addEventListener("submit", submitCommand);
  els.input.addEventListener("input", () => {
    updateCount();
    fitInput();
  });
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  window.addEventListener("resize", fitInput);

  bindDrop();
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    checkBrain();
    checkDesktop();
  });
}

function bindRadio(group, attribute, choose) {
  group.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const buttons = [...group.querySelectorAll("button")];
    const index = buttons.findIndex((button) => button.getAttribute("aria-pressed") === "true");
    const step = event.key === "ArrowRight" ? 1 : -1;
    const next = buttons[(index + step + buttons.length) % buttons.length];
    if (!next) return;
    event.preventDefault();
    next.focus();
    choose(next.getAttribute(attribute));
  });
}

function bindDrop() {
  const zone = els.dropZone;
  let depth = 0;
  zone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    zone.dataset.hot = "true";
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  zone.addEventListener("dragleave", () => {
    depth -= 1;
    if (depth <= 0) {
      depth = 0;
      zone.dataset.hot = "false";
    }
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    depth = 0;
    zone.dataset.hot = "false";
    ingestFiles(event.dataTransfer ? event.dataTransfer.files : []);
  });
  els.manualInput.addEventListener("change", () => {
    const files = [...els.manualInput.files];
    els.manualInput.value = "";
    ingestFiles(files);
  });
}

async function loadSettings() {
  try {
    applySettings(await request("/v1/settings", { timeout: 8000 }));
  } catch {
    /* The brain lamp records the miss. */
  }
  checkDesktop();
  checkBrain();
}

function applySettings(view) {
  if (!view || typeof view !== "object") return;
  if (typeof view.ui_theme === "string") applyTheme(view.ui_theme);
  const keys = view.provider_keys || {};
  setKey("xai_grok", keys.xai_grok && keys.xai_grok.configured === true, "xAI");
  setKey("openai", keys.openai && keys.openai.configured === true, "OpenAI");
  setKey("anthropic", keys.anthropic && keys.anthropic.configured === true, "Anthropic");
  const network = view.network || {};
  const next = normalizeDesktop(network.desktop_executor_ip, network.port);
  const changed = next.ip !== state.desktop.ip || next.port !== state.desktop.port;
  state.desktop = next;
  setText(els.desktopAddress, `${next.ip}:${next.port}`);
  if (changed) {
    setLamp(els.desktopLamp, els.desktopState, "checking", "Checking");
    checkDesktop();
  }
}

function applyTheme(theme) {
  const next = THEME_IDS.has(theme) ? theme : "dark_studio";
  state.theme = next;
  document.documentElement.dataset.theme = next;
  document.querySelectorAll("button[data-theme-choice]").forEach((button) => {
    button.setAttribute("aria-pressed", button.getAttribute("data-theme-choice") === next ? "true" : "false");
  });
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    /* storage unavailable */
  }
}

async function chooseTheme(theme) {
  if (!THEME_IDS.has(theme) || theme === state.theme || state.themeLock) return;
  const previous = state.theme;
  applyTheme(theme);
  state.themeLock = true;
  try {
    applySettings(await request("/v1/settings/update", {
      method: "POST",
      body: { ui_theme: theme },
      timeout: 8000,
    }));
    setThemeNote("");
  } catch (error) {
    applyTheme(previous);
    setThemeNote(messageOf(error));
  } finally {
    state.themeLock = false;
  }
}

function setThemeNote(text) {
  els.themeNote.textContent = text || "";
}

function setKey(provider, on, label) {
  const node = document.querySelector(`[data-provider="${provider}"]`);
  if (!node) return;
  node.dataset.on = on ? "true" : "false";
  node.setAttribute("aria-label", on ? `${label} key is configured` : `${label} key is empty`);
}

function setRoute(route) {
  state.route = route === "orchestrate" ? "orchestrate" : "infer";
  els.form.dataset.route = state.route;
  setText(els.routeKicker, state.route === "infer" ? "Infer" : "Orchestrate");
  document.querySelectorAll("button[data-route]").forEach((button) => {
    button.setAttribute("aria-pressed", button.getAttribute("data-route") === state.route ? "true" : "false");
  });
  els.manualsField.hidden = state.route !== "infer";
  els.input.maxLength = LIMITS[state.route];
  updateCount();
  fitInput();
}

function updateCount() {
  const limit = LIMITS[state.route];
  const length = els.input.value.length;
  setText(els.count, `${length} / ${limit}`);
  els.count.dataset.over = length > limit ? "true" : "false";
}

function fitInput() {
  const input = els.input;
  input.style.height = "auto";
  const max = Math.max(96, Math.round(window.innerHeight * 0.32));
  input.style.height = `${Math.min(input.scrollHeight, max)}px`;
}

async function submitCommand(event) {
  event.preventDefault();
  if (state.sending) return;
  const prompt = els.input.value.trim();
  const route = state.route;
  const limit = LIMITS[route];
  if (!prompt) return;
  if (prompt.length > limit) {
    pushEntry("error", "Console", "", `This ${route} command accepts ${limit} characters.`);
    return;
  }

  state.sending = true;
  els.send.disabled = true;
  els.form.setAttribute("aria-busy", "true");
  els.send.textContent = "Sending";
  els.input.value = "";
  updateCount();
  fitInput();
  pushEntry("user", "You", route === "infer" ? "Infer" : "Orchestrate", prompt);

  try {
    const body = route === "infer"
      ? { prompt, model: MODEL, use_manuals: els.useManuals.checked }
      : { prompt, target_device_ids: [], session_id: SESSION_ID };
    const payload = await request(`/v1/${route}`, {
      method: "POST",
      body,
      timeout: route === "infer" ? 90000 : 20000,
    });
    if (route === "infer") renderInfer(payload);
    else renderPlan(payload);
  } catch (error) {
    pushEntry("error", "Brain", "", messageOf(error));
  } finally {
    state.sending = false;
    els.send.disabled = false;
    els.form.setAttribute("aria-busy", "false");
    els.send.textContent = "Send";
    els.input.focus();
  }
}

function renderInfer(data) {
  const model = data && typeof data.model === "string" ? data.model : MODEL;
  const meta = [model, shortId(data && data.request_id)].filter(Boolean).join(" · ");
  pushEntry("infer", "Brain", meta, data && typeof data.content === "string" ? data.content : "", (entry) => {
    appendSources(entry, data && data.sources);
  });
}

function renderPlan(data) {
  const steps = data && Array.isArray(data.steps) ? data.steps : [];
  const status = data && data.status === "planned" ? "Planned" : "No match";
  const meta = [`${status} · ${steps.length} ${steps.length === 1 ? "step" : "steps"}`, shortId(data && data.request_id)]
    .filter(Boolean)
    .join(" · ");
  pushEntry("plan", "Plan", meta, "", (entry) => {
    if (steps.length) {
      const list = document.createElement("ol");
      list.className = "steps";
      steps.forEach((step, index) => list.append(textEl("li", "", stepLine(step, index))));
      entry.append(list);
    }
    const warnings = data && Array.isArray(data.warnings)
      ? data.warnings.filter((item) => typeof item === "string" && item)
      : [];
    if (warnings.length) {
      entry.append(textEl("p", "subkicker", "Warnings"));
      const list = document.createElement("ul");
      list.className = "warnings";
      warnings.forEach((warning) => list.append(textEl("li", "", warning)));
      entry.append(list);
    }
  });
}

function appendSources(entry, sources) {
  if (!Array.isArray(sources) || !sources.length) return;
  entry.append(textEl("p", "subkicker", "Manuals"));
  const list = document.createElement("ul");
  list.className = "sources";
  sources.forEach((source) => {
    const name = source && typeof source.source === "string" ? source.source : "manual";
    const page = source && Number.isInteger(source.page) ? ` · p.${source.page}` : "";
    const score = source && typeof source.score === "number" && Number.isFinite(source.score)
      ? ` · ${source.score.toFixed(2)}`
      : "";
    list.append(textEl("li", "", `${name}${page}${score}`));
  });
  entry.append(list);
}

function stepLine(step, index) {
  const safe = step && typeof step === "object" ? step : {};
  const number = String((Number.isInteger(safe.step_index) ? safe.step_index : index) + 1).padStart(2, "0");
  const device = textOr(safe.device_id, "device");
  const label = textOr(safe.label, textOr(safe.control_id, "control"));
  const port = typeof safe.port_hint === "string" && safe.port_hint ? ` · ${safe.port_hint}` : "";
  return `${number}  ${device} · ${label} · ${midiLine(safe.midi)}${port}`;
}

function midiLine(midi) {
  if (!midi || typeof midi !== "object") return "MIDI";
  if (midi.type === "control_change") return `CC ${midi.controller} = ${midi.value} · ch ${midi.channel}`;
  if (midi.type === "mmc") return `MMC ${String(midi.command || "command").replaceAll("_", " ")}`;
  return textOr(midi.type, "MIDI");
}

let uploadChain = Promise.resolve();

function ingestFiles(fileList) {
  [...(fileList || [])].forEach((file) => {
    uploadChain = uploadChain.then(() => indexManual(file)).catch(() => {});
  });
}

async function indexManual(file) {
  const name = file && file.name ? file.name : "That file";
  const id = `drop-${createSessionId()}`;
  if (!isPdf(file)) {
    logDrop(id, `${name} is not a PDF.`, "error");
    return;
  }

  // The brain indexes a PDF that already lives in manuals/. The drop sends that file name.
  state.uploads += 1;
  els.dropZone.dataset.busy = "true";
  els.dropZone.setAttribute("aria-busy", "true");
  logDrop(id, `Indexing ${name}`, "pending");
  try {
    const report = await request("/v1/rag/upload", {
      method: "POST",
      body: { filename: file.name },
      timeout: 60000,
    });
    const chunks = report && Number.isFinite(report.chunk_count) ? report.chunk_count : null;
    if (chunks !== null) setText(els.chunkCount, `${chunks} chunks`);
    const errors = report && Array.isArray(report.errors) ? report.errors.filter(Boolean) : [];
    logDrop(id, describeReport(file.name, report), errors.length ? "error" : "ok");
  } catch (error) {
    logDrop(id, messageOf(error), "error");
  } finally {
    state.uploads = Math.max(0, state.uploads - 1);
    if (state.uploads === 0) {
      delete els.dropZone.dataset.busy;
      els.dropZone.setAttribute("aria-busy", "false");
    }
  }
}

function isPdf(file) {
  if (!file || typeof file.name !== "string" || !file.name.trim()) return false;
  return /\.pdf$/i.test(file.name) || file.type === "application/pdf" || file.type === "application/x-pdf";
}

function describeReport(name, report) {
  if (!report || typeof report !== "object") return `Index updated for ${name}.`;
  const chunks = Number.isFinite(report.chunk_count) ? `${report.chunk_count} chunks in the index.` : "";
  const warning = Array.isArray(report.warnings) && report.warnings.length ? ` ${report.warnings[0]}` : "";
  if (listsName(report.indexed, name)) return `Indexed ${name}. ${chunks}${warning}`.trim();
  if (listsName(report.unchanged, name)) return `${name} is already current. ${chunks}${warning}`.trim();
  if (Array.isArray(report.errors) && report.errors.length) return report.errors.join(" ");
  return `Manual index updated. ${chunks}`.trim();
}

function listsName(list, name) {
  const folded = String(name).toLowerCase();
  return Array.isArray(list) && list.some((item) => {
    const value = String(item).toLowerCase().replaceAll("\\", "/");
    return value === folded || value.endsWith(`/${folded}`);
  });
}

function logDrop(id, message, tone) {
  let item = document.getElementById(id);
  if (!item) {
    item = document.createElement("li");
    item.id = id;
    els.dropLog.prepend(item);
  }
  item.dataset.tone = tone;
  item.textContent = message;
  while (els.dropLog.children.length > 6) els.dropLog.lastElementChild.remove();
}

async function checkBrain() {
  if (state.sending || state.uploads > 0 || document.hidden) return;
  try {
    const health = await request("/health", { timeout: 4000 });
    if (health && typeof health.service_version === "string" && health.service_version) {
      setText(els.build, `v${health.service_version}`);
    }
  } catch {
    /* The lamp is updated by request(). */
  }
}

async function checkDesktop() {
  if (document.hidden || state.desktopProbe) return;
  state.desktopProbe = true;
  const { ip, port } = state.desktop;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    await fetch(`http://${ip}:${port}/`, {
      mode: "no-cors",
      cache: "no-store",
      credentials: "omit",
      signal: controller.signal,
    });
    setLamp(els.desktopLamp, els.desktopState, "online", "Linked");
  } catch {
    setLamp(els.desktopLamp, els.desktopState, "offline", "Unreachable");
  } finally {
    clearTimeout(timer);
    state.desktopProbe = false;
  }
}

function setLamp(lamp, label, lampState, text) {
  if (lamp.dataset.state !== lampState) lamp.dataset.state = lampState;
  if (label.dataset.state !== lampState) label.dataset.state = lampState;
  setText(label, text);
}

function markBrain(online) {
  setLamp(els.brainLamp, els.brainState, online ? "online" : "offline", online ? "Online" : "Offline");
}

async function request(path, options) {
  const settings = options || {};
  const headers = { Accept: "application/json" };
  let body;
  if (settings.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(settings.body);
  }
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: settings.method || "GET",
      headers,
      body,
      cache: "no-store",
      credentials: "omit",
      signal: timeoutSignal(settings.timeout || 15000),
    });
  } catch (error) {
    markBrain(false);
    const name = error && error.name;
    if (name === "TimeoutError" || name === "AbortError") throw new Error("Timed out waiting for the brain.");
    throw new Error("The brain is unreachable.");
  }
  markBrain(true);
  if (!response.ok) throw new Error(await readError(response));
  try {
    return await response.json();
  } catch {
    throw new Error("The brain returned an unreadable response.");
  }
}

async function readError(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  const detail = payload && payload.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (detail && typeof detail === "object" && !Array.isArray(detail) && typeof detail.message === "string") {
    return detail.message;
  }
  if (Array.isArray(detail)) {
    const message = detail.map((item) => (item && typeof item.msg === "string" ? item.msg : "")).find(Boolean);
    if (message) return message;
  }
  return `The brain rejected the request (${response.status}).`;
}

function pushEntry(tone, kicker, meta, body, build) {
  if (els.placeholder) els.placeholder.remove();
  const entry = document.createElement("article");
  entry.className = "entry";
  entry.dataset.tone = tone;
  const head = document.createElement("div");
  head.className = "entry-head";
  head.append(textEl("span", "kicker", kicker));
  if (meta) head.append(textEl("span", "meta", meta));
  entry.append(head);
  if (body) entry.append(textEl("p", "body", body));
  if (build) build(entry);
  els.transcript.append(entry);
  while (els.transcript.children.length > 40) els.transcript.firstElementChild.remove();
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function textEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function setText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function textOr(value, fallback) {
  return typeof value === "string" && value ? value : fallback;
}

function shortId(value) {
  return typeof value === "string" && value.length >= 8 ? value.slice(0, 8) : "";
}

function messageOf(error) {
  return error instanceof Error && error.message ? error.message : "The command could not be sent.";
}

function tickClock() {
  const now = new Date();
  els.clock.dateTime = now.toISOString();
  setText(els.clock, new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(now));
}

function normalizeDesktop(ip, port) {
  if (typeof ip !== "string" || !isIpv4(ip)) return state.desktop;
  const parsed = Number(port);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) return { ip, port: state.desktop.port };
  return { ip, port: parsed };
}

function isIpv4(value) {
  const parts = value.split(".");
  if (parts.length !== 4) return false;
  return parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255);
}

function hostOf(base) {
  try {
    return new URL(base).host;
  } catch {
    return "127.0.0.1:8000";
  }
}

function resolveApiBase() {
  const fallback = "http://127.0.0.1:8000";
  try {
    const raw = new URLSearchParams(window.location.search).get("api");
    if (!raw) return fallback;
    const url = new URL(raw);
    if (url.protocol !== "http:" && url.protocol !== "https:") return fallback;
    return url.origin;
  } catch {
    return fallback;
  }
}

function createSessionId() {
  try {
    if (window.crypto && typeof crypto.randomUUID === "function") {
      return `console-${crypto.randomUUID()}`.slice(0, 128);
    }
  } catch {
    /* randomUUID needs a secure context */
  }
  return `console-${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`.slice(0, 128);
}

function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(ms);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}
