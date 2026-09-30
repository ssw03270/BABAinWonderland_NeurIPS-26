import { fetchJson, formatError, formatLines, shouldIgnoreHotkeys } from "/static/common.js";

const state = {
  classes: [],
  selectedClassIdx: null,
  selectedPosition: 0,
  selectedPayload: null,
  busy: false,
};

const elements = {
  status: document.getElementById("viewer-status"),
  classCount: document.getElementById("viewer-class-count"),
  rootInfo: document.getElementById("viewer-root-info"),
  classList: document.getElementById("viewer-class-list"),
  classTitle: document.getElementById("viewer-class-title"),
  transitionPosition: document.getElementById("viewer-transition-position"),
  prevButton: document.getElementById("viewer-prev-transition"),
  nextButton: document.getElementById("viewer-next-transition"),
  prevBoard: document.getElementById("viewer-prev-state"),
  nextBoard: document.getElementById("viewer-next-state"),
  transitionMetadata: document.getElementById("viewer-transition-metadata"),
  transitionStateDiff: document.getElementById("viewer-class-state-diff"),
  transitionRaw: document.getElementById("viewer-transition-raw"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(message) {
  elements.status.textContent = message || "idle";
}

function setBusy(value) {
  const busy = Boolean(value);
  state.busy = busy;
  elements.prevButton.disabled = busy;
  elements.nextButton.disabled = busy;
  for (const item of elements.classList.children) {
    item.disabled = busy;
  }
}

function classButtonLabel(payload) {
  const action = payload.action || "UNKNOWN";
  return `#${payload.classIdx} | ${action} | count ${payload.transitionCount}`;
}

function selectFirstAvailableClass() {
  if (!state.classes.length) {
    return null;
  }
  return state.classes[0].classIdx;
}

function updateClassListSelection() {
  const selectedIdx = state.selectedClassIdx;
  for (const child of elements.classList.children) {
    const classIdx = Number(child.dataset.classIdx);
    if (Number.isNaN(classIdx)) {
      continue;
    }
    if (classIdx === selectedIdx) {
      child.classList.add("is-active");
    } else {
      child.classList.remove("is-active");
    }
  }
}

function updateNavigationState(total, position) {
  elements.transitionPosition.textContent = `${Math.max(0, position + 1)} / ${Math.max(0, total)}`;
  elements.prevButton.disabled = state.busy || position <= 0;
  elements.nextButton.disabled = state.busy || position < 0 || position + 1 >= total;
}

function formatTransitionMetadata(payload) {
  const transition = payload.transition || {};
  const bundle = payload.bundle || {};
  const classData = payload.class || {};
  return formatLines([
    `class: #${classData.classIdx ?? "-"}`,
    `action: ${transition.action || "N/A"}`,
    `transition index: ${transition.transitionIndex ?? "-"}`,
    `state index: ${transition.stateIndex ?? "-"} -> ${transition.nextStateIndex ?? "-"}`,
    `reward: ${typeof transition.reward === "number" ? transition.reward.toFixed(5) : transition.reward}`,
    `done: ${Boolean(transition.done)}`,
    `terminated: ${Boolean(transition.terminated)}`,
    `truncated: ${Boolean(transition.truncated)}`,
    `dataset: ${bundle.datasetLabel || "?"}`,
    `dataset root: ${bundle.datasetRoot || "-"}`,
    `artifact: ${bundle.artifactStem || "-"}`,
    `scenario: ${bundle.scenarioType || "-"}`,
    `transitions file: ${bundle.transitionsPath || "-"}`,
  ]);
}

function formatDiff(payload) {
  const classData = payload.class || {};
  const stateDiff = classData.stateDiff || {};
  return JSON.stringify(stateDiff, null, 2);
}

function formatRawTransition(payload) {
  const transition = payload.transition || {};
  const debug = payload.debug || {};
  return formatLines([
    `navigation: prev=${payload.navigation?.previous ?? "-"}, next=${payload.navigation?.next ?? "-"}`,
    `previous state lookup: ${debug.previousStateLookup || "-"}`,
    `next state lookup: ${debug.nextStateLookup || "-"}`,
  ]);
}

function renderStateImageFrame(host, artifact, emptyText) {
  if (!host) {
    return;
  }
  host.innerHTML = "";
  if (!artifact?.url) {
    host.innerHTML = `<div class="transition-empty">${escapeHtml(emptyText || "image unavailable")}</div>`;
    return;
  }
  const link = document.createElement("a");
  link.className = "transition-image-link";
  link.href = artifact.url;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.title = String(artifact.path || artifact.url || "");

  const image = document.createElement("img");
  image.className = "transition-image";
  image.src = artifact.url;
  image.alt = String(artifact.path || "state image");
  image.loading = "lazy";

  link.append(image);
  host.append(link);
}

function renderTransition(payload) {
  state.selectedPayload = payload;
  const classData = payload.class || {};
  const total = Number(payload.total || 0);
  const position = Number(payload.position || 0);

  state.selectedClassIdx = Number(classData.classIdx);
  state.selectedPosition = position;

  elements.classTitle.textContent = `Class #${classData.classIdx ?? "-"} (${classData.action || "UNKNOWN"})`;
  updateNavigationState(total, position);
  updateClassListSelection();

  renderStateImageFrame(elements.prevBoard, payload.previousImage || null, "Previous state unavailable");
  renderStateImageFrame(elements.nextBoard, payload.nextImage || null, "Next state unavailable");

  elements.transitionMetadata.textContent = formatTransitionMetadata(payload);
  elements.transitionStateDiff.textContent = formatDiff(payload);
  elements.transitionRaw.textContent = formatRawTransition(payload);
  setStatus(`class ${classData.classIdx} loaded`);
  setBusy(false);
}

function renderClassList() {
  elements.classList.textContent = "";
  for (const cls of state.classes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button viewer-class-item";
    button.dataset.classIdx = String(cls.classIdx);
    button.textContent = classButtonLabel(cls);
    button.addEventListener("click", () => {
      if (state.busy) {
        return;
      }
      loadClass(cls.classIdx);
    });
    elements.classList.append(button);
  }
  elements.classCount.textContent = `${state.classes.length} classes`;
}

async function loadClassList() {
  const payload = await fetchJson("/api/classes");
  const classes = Array.isArray(payload?.classes) ? payload.classes : [];
  state.classes = classes.slice().sort((a, b) => Number(a.classIdx) - Number(b.classIdx));
  renderClassList();
  if (!state.classes.length) {
    setStatus("No classes found");
    elements.rootInfo.textContent = "No class mappings could be loaded from discovery JSON.";
    return;
  }
  const firstClass = selectFirstAvailableClass();
  elements.rootInfo.textContent = `total classes: ${state.classes.length} / total transitions shown in this UI: ${
    state.classes.reduce((acc, item) => acc + Number(item.transitionCount || 0), 0)
  }`;
  await loadClass(firstClass);
}

async function loadClass(classIdx) {
  const normalized = Number(classIdx);
  if (Number.isNaN(normalized)) {
    return;
  }
  const classEntry = state.classes.find((item) => Number(item.classIdx) === normalized);
  if (!classEntry) {
    return;
  }
  await loadTransition(normalized, 0);
}

async function loadTransition(classIdx, position) {
  if (state.busy) {
    return;
  }
  setBusy(true);
  setStatus(`loading class ${classIdx} transition`);
  try {
    const payload = await fetchJson(`/api/classes/${classIdx}/transition/${position}`);
    renderTransition(payload);
  } catch (error) {
    elements.transitionMetadata.textContent = formatError(error);
    setBusy(false);
    setStatus("failed to load transition");
  }
}

function nextTransition() {
  if (state.busy || state.selectedClassIdx === null || !state.selectedPayload) {
    return;
  }
  const next = state.selectedPayload.navigation?.next;
  if (next === null || next === undefined) {
    return;
  }
  loadTransition(state.selectedClassIdx, Number(next)).catch(() => {});
}

function prevTransition() {
  if (state.busy || state.selectedClassIdx === null || !state.selectedPayload) {
    return;
  }
  const previous = state.selectedPayload.navigation?.previous;
  if (previous === null || previous === undefined) {
    return;
  }
  loadTransition(state.selectedClassIdx, Number(previous)).catch(() => {});
}

function registerClicks() {
  elements.prevButton.addEventListener("click", prevTransition);
  elements.nextButton.addEventListener("click", nextTransition);
}

function registerHotkeys() {
  window.addEventListener("keydown", (event) => {
    if (event.repeat || shouldIgnoreHotkeys(event.target) || state.busy) {
      return;
    }
    if (event.key === "ArrowLeft") {
      prevTransition();
      event.preventDefault();
    } else if (event.key === "ArrowRight") {
      nextTransition();
      event.preventDefault();
    }
  });
}

function registerResize() {
  let resizeFrame = null;
  window.addEventListener("resize", () => {
    if (!state.selectedPayload) {
      return;
    }
    if (resizeFrame !== null) {
      cancelAnimationFrame(resizeFrame);
    }
    resizeFrame = requestAnimationFrame(() => {
      renderTransition(state.selectedPayload);
    });
  });
}

async function boot() {
  registerClicks();
  registerHotkeys();
  registerResize();
  try {
    await loadClassList();
  } catch (error) {
    setStatus("failed to load");
    elements.rootInfo.textContent = formatError(error);
    elements.transitionMetadata.textContent = formatError(error);
    setBusy(false);
  }
}

boot().catch((error) => {
  setStatus("failed");
  elements.rootInfo.textContent = formatError(error);
});
