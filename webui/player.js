import {
  buildStateScene,
  fetchJson,
  formatError,
  formatMapDisplayName,
  loadMapDisplayNames,
  renderBoard,
  shouldIgnoreHotkeys,
} from "/static/common.js";

const state = {
  payload: null,
  busy: false,
  saveNotice: "",
  errorNotice: "",
};

const elements = {
  status: document.getElementById("player-status"),
  meta: document.getElementById("player-meta"),
  boardMeta: document.getElementById("player-board-meta"),
  board: document.getElementById("player-board"),
  recordingIndicator: document.getElementById("player-recording-indicator"),
  mapList: document.getElementById("player-map-list"),
  actionHistory: document.getElementById("player-action-history"),
  resetButton: document.getElementById("reset-session-button"),
  undoButton: document.getElementById("undo-player-button"),
  nextScenarioButton: document.getElementById("next-scenario-button"),
  recordingButton: document.getElementById("toggle-recording-button"),
  saveButton: document.getElementById("save-player-button"),
};

function setStatus(value) {
  elements.status.textContent = value;
}

function playerStatusLabel(payload) {
  if (!payload) return "idle";
  if (payload.terminated) return payload.reward > 0 ? "win" : "ended";
  if (payload.truncated) return "time-limit";
  return "running";
}

function setBusy(value) {
  state.busy = value;
  const disabled = Boolean(value);
  elements.resetButton.disabled = disabled;
  elements.undoButton.disabled = disabled || !Boolean(state.payload?.canUndo);
  elements.nextScenarioButton.disabled = disabled || !(state.payload?.scenarios || []).length;
  elements.recordingButton.disabled = disabled || !state.payload;
  elements.saveButton.disabled = disabled;
  elements.mapList.querySelectorAll("button").forEach((button) => {
    button.disabled = disabled;
  });
}

function updateRecordingButton(payload) {
  const enabled = Boolean(payload?.autosaveEnabled);
  elements.recordingButton.textContent = enabled ? "Recording ON" : "Recording OFF";
  elements.recordingButton.classList.toggle("is-active", enabled);
  elements.recordingButton.setAttribute("aria-pressed", enabled ? "true" : "false");
  elements.recordingIndicator.hidden = !enabled;
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

function playerStatusTone(payload) {
  const status = playerStatusLabel(payload);
  if (status === "win") return "success";
  if (status === "ended") return "danger";
  if (status === "time-limit") return "warning";
  return "neutral";
}

function booleanTone(value, positiveTone = "success") {
  return value ? positiveTone : "muted";
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
    renderSessionHudError("player session unavailable");
    return;
  }

  const currentScenario = String(payload?.scenarioType || "").trim();
  const rewardValue = Number(payload?.reward || 0);
  const rewardTone = rewardValue > 0 ? "success" : rewardValue < 0 ? "danger" : "neutral";
  const saveNotice = String(state.saveNotice || "").trim();
  const errorNotice = String(state.errorNotice || "").trim();

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
          { label: "Env", value: String(payload.envLabel || "-"), tone: "neutral" },
          { label: "Seed", value: String(payload.seed ?? "-"), tone: "neutral" },
          { label: "Scenario", value: currentScenario || "-", tone: "muted" },
        ],
      ],
    }),
  );

  elements.meta.append(
    createSessionHudSection({
      title: "Episode",
      chipRows: [
        [
          { label: "Status", value: playerStatusLabel(payload), tone: playerStatusTone(payload) },
          { label: "Reward", value: rewardValue.toFixed(3), tone: rewardTone },
          { label: "Terminated", value: String(Boolean(payload.terminated)).toUpperCase(), tone: booleanTone(payload.terminated, rewardValue > 0 ? "success" : "danger") },
        ],
      ],
    }),
  );

  elements.meta.append(
    createSessionHudSection({
      title: "Paths",
      rawRows: [
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

  const currentMapLabel = (
    payload.currentMapLabel
    || payload.scenarioDisplayName
    || formatMapDisplayName(payload.scenarioType, payload.scenarioLabel)
    || "default/randomized"
  );

  function createBoardMetaCard(variant, labelText, valueText) {
    const card = document.createElement("div");
    card.className = `player-board-meta-card is-${variant}`;

    const label = document.createElement("div");
    label.className = "player-board-meta-label";
    label.textContent = labelText;

    const value = document.createElement("div");
    value.className = "player-board-meta-value";
    value.textContent = valueText;

    card.append(label, value);
    return card;
  }

  const fragment = document.createDocumentFragment();
  fragment.append(
    createBoardMetaCard("map", "Map", currentMapLabel),
    createBoardMetaCard("step", "Step", String(payload.stepIndex ?? "-")),
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
    label.textContent = String(entry?.label || formatMapDisplayName(entry?.scenarioType) || "unnamed map");
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

function render(payload) {
  state.payload = payload;
  setStatus(playerStatusLabel(payload));

  if (!payload) {
    renderSessionHud(null);
    renderBoardMeta(null);
    updateRecordingButton(null);
    renderBoard(null, elements.board, { emptyMessage: "" });
    setBusy(false);
    return;
  }

  renderBoardMeta(payload);
  renderSessionHud(payload);
  updateRecordingButton(payload);
  renderMapList(payload);
  renderActionHistory(payload);
  renderBoard(buildStateScene(payload.boardState, payload.visualConfig), elements.board, {
    tileSize: 44,
    fitToHost: true,
    minTileSize: 14,
    maxTileSize: 56,
    fallbackHostHeight: Math.min(window.innerHeight * 0.72, 820),
    emptyMessage: "current scene unavailable",
  });
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
  const payload = await fetchJson("/api/session");
  state.errorNotice = "";
  render(payload);
}

async function step(actionName) {
  const payload = await fetchJson("/api/session/actions", {
    method: "POST",
    body: JSON.stringify({ action: actionName }),
  });
  state.errorNotice = "";
  render(payload);
}

async function reset() {
  const payload = await fetchJson("/api/session/reset", {
    method: "POST",
    body: JSON.stringify({ seed: Number(state.payload?.seed || 42) }),
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function nextScenario() {
  const payload = await fetchJson("/api/session/next-scenario", {
    method: "POST",
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function selectMap(scenarioType) {
  const payload = await fetchJson("/api/session/select-map", {
    method: "POST",
    body: JSON.stringify({ scenarioType }),
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function undo() {
  const payload = await fetchJson("/api/session/undo", {
    method: "POST",
  });
  state.saveNotice = "";
  state.errorNotice = "";
  render(payload);
}

async function toggleRecording() {
  const nextEnabled = !Boolean(state.payload?.autosaveEnabled);
  const payload = await fetchJson("/api/session/recording", {
    method: "POST",
    body: JSON.stringify({ enabled: nextEnabled }),
  });
  state.errorNotice = "";
  render(payload);
}

async function saveFrame() {
  const payload = await fetchJson("/api/session/save", {
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
    else if (event.key.toLowerCase() === "t") command = () => nextScenario();
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
  elements.nextScenarioButton.addEventListener("click", () => withBusy(() => nextScenario()));
  elements.recordingButton.addEventListener("click", () => withBusy(() => toggleRecording()));
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
  await loadMapDisplayNames();
  await loadSession();
}

boot().catch((error) => {
  state.errorNotice = formatError(error);
  render(state.payload);
});
