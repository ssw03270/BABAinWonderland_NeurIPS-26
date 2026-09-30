import { buildStateScene, fetchJson, formatError, renderBoard, shouldIgnoreHotkeys } from "/static/common.js";

const API_ROOT = "/api/compare";

const state = {
  payload: null,
  busy: false,
  saveNotice: "",
  errorNotice: "",
};

const elements = {
  status: document.getElementById("compare-status"),
  meta: document.getElementById("compare-meta"),
  boardMeta: document.getElementById("compare-board-meta"),
  gtBefore: document.getElementById("compare-gt-before"),
  gtAfter: document.getElementById("compare-gt-after"),
  predBefore: document.getElementById("compare-pred-before"),
  predAfter: document.getElementById("compare-pred-after"),
  mapList: document.getElementById("compare-map-list"),
  actionHistory: document.getElementById("compare-action-history"),
  resetButton: document.getElementById("reset-compare-button"),
  undoButton: document.getElementById("undo-compare-button"),
  nextMapButton: document.getElementById("next-map-button"),
  saveButton: document.getElementById("save-compare-button"),
};

function setStatus(value) {
  elements.status.textContent = value;
}

function compareStatusLabel(payload) {
  if (!payload) return "idle";
  const frame = payload.frame;
  if (frame.terminated) return frame.reward > 0 ? "win" : "ended";
  if (frame.truncated) return "time-limit";
  return "running";
}

function compareStatusTone(payload) {
  const status = compareStatusLabel(payload);
  if (status === "win") return "success";
  if (status === "ended") return "danger";
  if (status === "time-limit") return "warning";
  return "neutral";
}

function booleanTone(value, positiveTone = "success") {
  return value ? positiveTone : "muted";
}

function setBusy(value) {
  state.busy = value;
  const disabled = Boolean(value);
  elements.resetButton.disabled = disabled;
  elements.undoButton.disabled = disabled || !Boolean(state.payload?.canUndo);
  elements.nextMapButton.disabled = disabled || (state.payload?.maps || []).length <= 1;
  elements.saveButton.disabled = disabled;
  elements.mapList.querySelectorAll("button").forEach((button) => {
    button.disabled = disabled;
  });
}

function createSessionHudChip(item) {
  const chip = document.createElement("div");
  chip.className = "hud-inline-chip";
  const tone = String(item?.tone || "").trim();
  if (tone) {
    chip.classList.add(`is-${tone}`);
  }

  const labelText = String(item?.label || "").trim();
  if (labelText) {
    const label = document.createElement("span");
    label.className = "hud-inline-chip-label";
    label.textContent = labelText;
    chip.append(label);
  }

  const value = document.createElement("span");
  value.className = "hud-inline-chip-value";
  value.textContent = String(item?.value ?? "");
  chip.append(value);
  return chip;
}

function createSessionHudSection({ title = "", chipRows = [], rawRows = [], variant = "" }) {
  const section = document.createElement("section");
  section.className = "hud-metric-section";
  if (variant) {
    section.classList.add(`is-${variant}`);
  }

  const titleText = String(title || "").trim();
  if (titleText) {
    const header = document.createElement("div");
    header.className = "hud-metric-section-title";
    header.textContent = titleText;
    section.append(header);
  }

  if (chipRows.length) {
    const stack = document.createElement("div");
    stack.className = "hud-summary-stack";
    chipRows.forEach((rowItems) => {
      const row = document.createElement("div");
      row.className = "hud-summary-strip";
      rowItems.forEach((item) => {
        row.append(createSessionHudChip(item));
      });
      stack.append(row);
    });
    section.append(stack);
  }

  if (rawRows.length) {
    const detailStack = document.createElement("div");
    detailStack.className = "hud-detail-stack";
    rawRows.forEach((rowItem) => {
      const row = document.createElement("div");
      row.className = "hud-detail-row";

      const labelText = String(rowItem?.label || "").trim();
      if (labelText) {
        const label = document.createElement("div");
        label.className = "hud-detail-label";
        label.textContent = labelText;
        row.append(label);
      }

      const content = document.createElement("div");
      content.className = "hud-detail-raw";
      content.textContent = String(rowItem?.value || "");
      row.append(content);
      detailStack.append(row);
    });
    section.append(detailStack);
  }

  return section;
}

function renderSessionHudError(message) {
  elements.meta.innerHTML = "";
  elements.meta.append(
    createSessionHudSection({
      title: "Alert",
      variant: "alert",
      rawRows: [{ label: "Error", value: String(message || "unknown error") }],
    }),
  );
}

function renderSessionHud(payload) {
  elements.meta.innerHTML = "";
  if (!payload) {
    renderSessionHudError("compare session unavailable");
    return;
  }

  const frame = payload.frame || {};
  const rewardValue = Number(frame.reward || 0);
  const rewardTone = rewardValue > 0 ? "success" : rewardValue < 0 ? "danger" : "neutral";
  const saveNotice = String(state.saveNotice || "").trim();
  const errorNotice = String(state.errorNotice || "").trim();
  const predictionError = String(frame.predictionError || "").trim();

  if (errorNotice) {
    elements.meta.append(
      createSessionHudSection({
        title: "Alert",
        variant: "alert",
        rawRows: [{ label: "Error", value: errorNotice }],
      }),
    );
  }

  elements.meta.append(
    createSessionHudSection({
      title: "System",
      chipRows: [
        [
          { label: "Version", value: String(payload.versionTag || "-"), tone: "neutral" },
          { label: "Seed", value: String(payload.seed ?? frame.seed ?? "-"), tone: "neutral" },
          { label: "Map", value: String(payload.currentMapLabel || payload.scenarioDisplayName || payload.scenarioType || "-"), tone: "muted" },
        ],
        [
          { label: "Predictor", value: String(payload.predictorMode || "-"), tone: "neutral" },
          { label: "Undo", value: String(Boolean(payload.canUndo)).toUpperCase(), tone: booleanTone(payload.canUndo) },
        ],
      ],
    }),
  );

  elements.meta.append(
    createSessionHudSection({
      title: "Episode",
      chipRows: [
        [
          { label: "Status", value: compareStatusLabel(payload), tone: compareStatusTone(payload) },
          { label: "Reward", value: rewardValue.toFixed(3), tone: rewardTone },
          { label: "Step", value: String(frame.stepIndex ?? "-"), tone: "neutral" },
        ],
        [
          { label: "Terminated", value: String(Boolean(frame.terminated)).toUpperCase(), tone: booleanTone(frame.terminated, rewardValue > 0 ? "success" : "danger") },
          { label: "Prediction", value: predictionError || "OK", tone: predictionError ? "warning" : "success" },
        ],
      ],
    }),
  );

  elements.meta.append(
    createSessionHudSection({
      title: "Paths",
      rawRows: [
        { label: "Program", value: String(payload.programPath || "-") },
        { label: "Output Dir", value: String(payload.outputDir || "-") },
        ...(saveNotice ? [{ label: "Saved", value: saveNotice }] : []),
      ],
    }),
  );
}

function renderBoardMeta(payload) {
  elements.boardMeta.innerHTML = "";
  if (!payload) {
    return;
  }

  function createBoardMetaCard(variant, labelText, valueText) {
    const card = document.createElement("div");
    card.className = `player-board-meta-card is-${variant}`;

    const label = document.createElement("div");
    label.className = "player-board-meta-label";
    label.textContent = labelText;

    const value = document.createElement("div");
    value.className = "player-board-meta-value";
    value.textContent = valueText;

    card.title = valueText;

    card.append(label, value);
    return card;
  }

  const diffSummary = String(payload.predictionDiffSummary || payload.frame?.differenceSummary || "").trim();
  const diffValue = payload.hasPredictionDiff
    ? diffSummary.replace(/^Diff:\s*/i, "").trim() || "Mismatch detected"
    : "Matched";

  const fragment = document.createDocumentFragment();
  fragment.append(
    createBoardMetaCard("map", "Map", String(payload.currentMapLabel || payload.scenarioDisplayName || payload.scenarioType || "default/randomized")),
    createBoardMetaCard("step", "Step", String(payload.frame?.stepIndex ?? "-")),
    createBoardMetaCard(payload.hasPredictionDiff ? "danger" : "success", "Diff", diffValue),
  );
  elements.boardMeta.append(fragment);
}

function renderMapList(payload) {
  const maps = Array.isArray(payload?.maps) ? payload.maps : [];
  elements.mapList.innerHTML = "";
  if (!maps.length) {
    elements.mapList.textContent = "available maps unavailable";
    return;
  }

  const fragment = document.createDocumentFragment();
  maps.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `player-map-button${entry?.isActive ? " is-active" : ""}`;
    button.disabled = state.busy;

    const label = document.createElement("span");
    label.className = "player-map-button-label";
    label.textContent = String(entry?.label || entry?.scenarioType || "unnamed map");
    button.append(label);

    const scenarioType = String(entry?.scenarioType || "").trim();
    if (scenarioType) {
      const meta = document.createElement("span");
      meta.className = "player-map-button-meta";
      meta.textContent = scenarioType;
      button.append(meta);
    }

    button.addEventListener("click", () => withBusy(() => selectMap(scenarioType || null)));
    fragment.append(button);
  });
  elements.mapList.append(fragment);
}

function renderActionHistory(payload) {
  const entries = Array.isArray(payload?.actionHistory) ? payload.actionHistory : [];
  elements.actionHistory.innerHTML = "";
  if (!entries.length) {
    elements.actionHistory.textContent = "no actions yet";
    return;
  }

  const fragment = document.createDocumentFragment();
  entries
    .slice()
    .reverse()
    .forEach((entry, index) => {
      const row = document.createElement("div");
      row.className = `player-history-entry is-${String(entry?.statusKey || "running")}`;
      if (index === 0) {
        row.classList.add("is-latest");
      }

      const badge = document.createElement("div");
      badge.className = "player-history-badge";
      badge.textContent = String(entry?.label || entry?.actionName || "ACTION");
      row.append(badge);

      const body = document.createElement("div");
      body.className = "player-history-body";

      const primary = document.createElement("div");
      primary.className = "player-history-primary";
      primary.textContent = `STEP ${String(Number(entry?.stepIndex || 0)).padStart(4, "0")}`;
      body.append(primary);

      const secondaryParts = [String(entry?.rewardText || "")];
      const note = String(entry?.note || "").trim();
      if (note) {
        secondaryParts.push(note);
      }
      const secondary = document.createElement("div");
      secondary.className = "player-history-secondary";
      secondary.textContent = secondaryParts.filter(Boolean).join(" · ");
      body.append(secondary);

      row.append(body);
      fragment.append(row);
    });
  elements.actionHistory.append(fragment);
}

function syncBoardHostHeight(hostElement) {
  if (!(hostElement instanceof HTMLElement)) {
    return;
  }
  const panelElement = hostElement.closest(".compare-panel");
  hostElement.style.display = "flex";
  hostElement.style.justifyContent = "center";
  hostElement.style.alignItems = "flex-start";
  hostElement.style.padding = "0";
  hostElement.style.minHeight = "0";
  hostElement.style.overflow = "hidden";
  const boardElement = hostElement.querySelector(".board");
  if (!(boardElement instanceof HTMLElement)) {
    hostElement.style.height = "";
    hostElement.style.maxHeight = "";
    if (panelElement instanceof HTMLElement) {
      panelElement.style.height = "";
      panelElement.style.minHeight = "";
      panelElement.style.maxHeight = "";
    }
    return;
  }
  const boardHeight = Math.ceil(boardElement.getBoundingClientRect().height);
  if (boardHeight > 0) {
    const heightValue = `${boardHeight}px`;
    hostElement.style.height = heightValue;
    hostElement.style.maxHeight = heightValue;
    if (panelElement instanceof HTMLElement) {
      const panelStyle = window.getComputedStyle(panelElement);
      const headerElement = panelElement.querySelector(".hud-panel-head");
      const headerHeight = headerElement instanceof HTMLElement
        ? Math.ceil(headerElement.getBoundingClientRect().height)
          + Math.ceil(Number.parseFloat(window.getComputedStyle(headerElement).marginBottom || "0") || 0)
        : 0;
      const paddingTop = Math.ceil(Number.parseFloat(panelStyle.paddingTop || "0") || 0);
      const paddingBottom = Math.ceil(Number.parseFloat(panelStyle.paddingBottom || "0") || 0);
      const panelHeightValue = `${headerHeight + boardHeight + paddingTop + paddingBottom}px`;
      panelElement.style.height = panelHeightValue;
      panelElement.style.minHeight = panelHeightValue;
      panelElement.style.maxHeight = panelHeightValue;
    }
    return;
  }
  hostElement.style.height = "";
  hostElement.style.maxHeight = "";
  if (panelElement instanceof HTMLElement) {
    panelElement.style.height = "";
    panelElement.style.minHeight = "";
    panelElement.style.maxHeight = "";
  }
}

function render(payload) {
  state.payload = payload;
  setStatus(compareStatusLabel(payload));

  if (!payload) {
    renderSessionHud(null);
    renderBoardMeta(null);
    renderBoard(null, elements.gtBefore, { emptyMessage: "" });
    renderBoard(null, elements.gtAfter, { emptyMessage: "" });
    renderBoard(null, elements.predBefore, { emptyMessage: "" });
    renderBoard(null, elements.predAfter, { emptyMessage: "" });
    syncBoardHostHeight(elements.gtBefore);
    syncBoardHostHeight(elements.gtAfter);
    syncBoardHostHeight(elements.predBefore);
    syncBoardHostHeight(elements.predAfter);
    setBusy(false);
    return;
  }

  renderBoardMeta(payload);
  renderSessionHud(payload);
  renderMapList(payload);
  renderActionHistory(payload);

  const boardOptions = {
    tileSize: 40,
    fitToHost: true,
    fitAxis: "width",
    hostPadding: 0,
    minTileSize: 8,
    maxTileSize: 56,
    fallbackHostHeight: Math.min(window.innerHeight * 0.72, 820),
  };
  const visualConfig = payload.visualConfig || null;
  const frame = payload.frame || {};
  renderBoard(buildStateScene(frame.gtBeforeState, visualConfig), elements.gtBefore, boardOptions);
  renderBoard(buildStateScene(frame.gtAfterState, visualConfig), elements.gtAfter, boardOptions);
  renderBoard(buildStateScene(frame.predBeforeState, visualConfig), elements.predBefore, boardOptions);
  renderBoard(buildStateScene(frame.predAfterState, visualConfig), elements.predAfter, {
    ...boardOptions,
    emptyMessage: "prediction unavailable",
  });
  syncBoardHostHeight(elements.gtBefore);
  syncBoardHostHeight(elements.gtAfter);
  syncBoardHostHeight(elements.predBefore);
  syncBoardHostHeight(elements.predAfter);
  setBusy(false);
}

async function withBusy(work) {
  if (state.busy) return;
  setBusy(true);
  try {
    await work();
  } catch (error) {
    state.errorNotice = formatError(error);
    render(state.payload);
  }
}

async function loadSession() {
  const payload = await fetchJson(`${API_ROOT}/session`);
  state.errorNotice = "";
  render(payload);
}

async function step(actionName) {
  const payload = await fetchJson(`${API_ROOT}/session/actions`, {
    method: "POST",
    body: JSON.stringify({ action: actionName }),
  });
  state.errorNotice = "";
  render(payload);
}

async function reset() {
  const payload = await fetchJson(`${API_ROOT}/session/reset`, {
    method: "POST",
    body: JSON.stringify({ seed: Number(state.payload?.seed || state.payload?.frame?.seed || 42) }),
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function undo() {
  const payload = await fetchJson(`${API_ROOT}/session/undo`, {
    method: "POST",
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function nextMap() {
  const payload = await fetchJson(`${API_ROOT}/session/next-scenario`, {
    method: "POST",
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function selectMap(scenarioType) {
  const payload = await fetchJson(`${API_ROOT}/session/select-map`, {
    method: "POST",
    body: JSON.stringify({ scenarioType }),
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function saveFrame() {
  const payload = await fetchJson(`${API_ROOT}/session/save`, {
    method: "POST",
  });
  state.saveNotice = `${payload.imagePath} | ${payload.jsonPath}`;
  state.errorNotice = "";
  render(state.payload);
}

function registerHotkeys() {
  window.addEventListener("keydown", (event) => {
    if (event.repeat || shouldIgnoreHotkeys(event.target) || state.busy) {
      return;
    }

    let command = null;
    if (event.key === "ArrowUp") command = () => step("up");
    else if (event.key === "ArrowDown") command = () => step("down");
    else if (event.key === "ArrowLeft") command = () => step("left");
    else if (event.key === "ArrowRight") command = () => step("right");
    else if (event.key === " ") command = () => step("idle");
    else if (event.key.toLowerCase() === "z") command = () => undo();
    else if (event.key.toLowerCase() === "r") command = () => reset();
    else if (event.key.toLowerCase() === "t") command = () => nextMap();
    else if (event.key.toLowerCase() === "s") command = () => saveFrame();

    if (command) {
      event.preventDefault();
      withBusy(command);
    }
  });
}

function registerClicks() {
  elements.resetButton.addEventListener("click", () => withBusy(() => reset()));
  elements.undoButton.addEventListener("click", () => withBusy(() => undo()));
  elements.nextMapButton.addEventListener("click", () => withBusy(() => nextMap()));
  elements.saveButton.addEventListener("click", () => withBusy(() => saveFrame()));
}

function registerResize() {
  window.addEventListener("resize", () => {
    if (state.payload) {
      render(state.payload);
    }
  });
}

async function boot() {
  registerClicks();
  registerHotkeys();
  registerResize();
  await loadSession();
}

boot().catch((error) => {
  state.errorNotice = formatError(error);
  render(state.payload);
});
