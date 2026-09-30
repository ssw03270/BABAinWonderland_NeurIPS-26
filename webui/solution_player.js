import { buildStateScene, fetchJson, formatError, formatLines, renderBoard, shouldIgnoreHotkeys } from "/static/common.js";

const state = {
  payload: null,
  busy: false,
};

const elements = {
  status: document.getElementById("player-status"),
  meta: document.getElementById("player-meta"),
  board: document.getElementById("player-board"),
  resetButton: document.getElementById("reset-session-button"),
  saveSolutionButton: document.getElementById("save-solution-button"),
  nextScenarioButton: document.getElementById("next-scenario-button"),
  actionButtons: [...document.querySelectorAll(".action-button")],
};

function setStatus(value) {
  elements.status.textContent = value;
}

function collectorStatusLabel(payload) {
  if (!payload) return "idle";
  if (payload.collectionComplete) return "complete";
  if (payload.currentSolutionSaved) return "saved";
  if (payload.currentSolutionReady) return "solved";
  if (payload.terminated) return payload.reward > 0 ? "win" : "ended";
  if (payload.truncated) return "time-limit";
  return "running";
}

function setBusy(value) {
  state.busy = value;
  const disabled = Boolean(value);
  elements.resetButton.disabled = disabled;
  elements.nextScenarioButton.disabled = disabled || !state.payload || Boolean(state.payload.collectionComplete);
  elements.saveSolutionButton.disabled = (
    disabled
    || !state.payload
    || !state.payload.currentSolutionReady
    || Boolean(state.payload.currentSolutionSaved)
  );
  elements.actionButtons.forEach((button) => {
    button.disabled = (
      disabled
      || !state.payload
      || Boolean(state.payload.terminated)
      || Boolean(state.payload.truncated)
      || Boolean(state.payload.collectionComplete)
    );
  });
}

function render(payload) {
  state.payload = payload;
  setStatus(collectorStatusLabel(payload));

  if (!payload) {
    elements.meta.textContent = "solution collector unavailable";
    renderBoard(null, elements.board, { emptyMessage: "" });
    setBusy(false);
    return;
  }

  const lastInfo = payload.lastSolutionInfo || {};
  elements.meta.textContent = formatLines([
    `env: ${payload.envLabel}`,
    `split: ${payload.difficulty}/${payload.scenarioSplit}`,
    `scenario: ${payload.scenarioLabel || "default/randomized"}`,
    `progress: ${payload.scenarioIndex}/${payload.scenarioCount}`,
    `saved: ${payload.solvedCount}/${payload.scenarioCount}`,
    `remaining: ${payload.remainingCount}`,
    `step: ${payload.stepIndex}`,
    `raw actions: ${payload.actionCountRaw}`,
    `reward: ${payload.reward.toFixed(3)}`,
    `terminated: ${String(payload.terminated)}`,
    `truncated: ${String(payload.truncated)}`,
    `solution ready: ${payload.currentSolutionReady ? "YES" : "NO"}`,
    `solution saved: ${payload.currentSolutionSaved ? "YES" : "NO"}`,
    `max steps: ${payload.envMaxStepsLabel || payload.envMaxSteps || "unlimited"}`,
    `output root: ${payload.outputRoot || "-"}`,
    lastInfo.message ? `note: ${lastInfo.message}` : "",
    lastInfo.transitionsPath ? `transitions: ${lastInfo.transitionsPath}` : "",
    lastInfo.statesPath ? `states: ${lastInfo.statesPath}` : "",
  ]);
  renderBoard(buildStateScene(payload.boardState, payload.visualConfig), elements.board, {
    tileSize: 64,
    fitToHost: true,
    minTileSize: 28,
    maxTileSize: 92,
    fallbackHostHeight: Math.min(window.innerHeight * 0.76, 920),
  });
  setBusy(false);
}

async function withBusy(work) {
  if (state.busy) return;
  setBusy(true);
  try {
    await work();
  } catch (error) {
    elements.meta.textContent = formatError(error);
    setBusy(false);
  }
}

async function loadSession() {
  const payload = await fetchJson("/api/session");
  render(payload);
}

async function step(actionName) {
  const payload = await fetchJson("/api/session/actions", {
    method: "POST",
    body: JSON.stringify({ action: actionName }),
  });
  render(payload);
}

async function reset() {
  const payload = await fetchJson("/api/session/reset", {
    method: "POST",
    body: JSON.stringify({ seed: Number(state.payload?.seed || 42) }),
  });
  render(payload);
}

async function saveSolution() {
  const payload = await fetchJson("/api/session/save-solution", {
    method: "POST",
  });
  render(payload);
}

async function nextScenario() {
  const payload = await fetchJson("/api/session/next-scenario", {
    method: "POST",
  });
  render(payload);
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
    else if (event.key.toLowerCase() === "r") command = () => reset();
    else if (event.key.toLowerCase() === "s") command = () => saveSolution();
    else if (event.key.toLowerCase() === "t") command = () => nextScenario();

    if (command) {
      event.preventDefault();
      withBusy(command);
    }
  });
}

function registerClicks() {
  elements.resetButton.addEventListener("click", () => withBusy(() => reset()));
  elements.saveSolutionButton.addEventListener("click", () => withBusy(() => saveSolution()));
  elements.nextScenarioButton.addEventListener("click", () => withBusy(() => nextScenario()));
  elements.actionButtons.forEach((button) => {
    button.addEventListener("click", () => withBusy(() => step(button.dataset.action)));
  });
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
  elements.meta.textContent = formatError(error);
});
