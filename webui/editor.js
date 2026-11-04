import {
  fetchJson,
  formatError,
  formatLines,
  normalizeNullableInteger,
  renderBoard,
  shouldIgnoreHotkeys,
  spritePathFor,
} from "/static/common.js";

const state = {
  payload: null,
  catalog: null,
  busy: false,
  selectedToken: ".",
  selectedDirection: null,
  selectedCell: [1, 1],
};

const elements = {
  status: document.getElementById("editor-status"),
  meta: document.getElementById("editor-meta"),
  catalogMeta: document.getElementById("editor-catalog-meta"),
  board: document.getElementById("editor-board"),
  palette: document.getElementById("editor-palette"),
  saveButton: document.getElementById("editor-save-button"),
  clearInteriorButton: document.getElementById("editor-clear-interior-button"),
  loadDifficulty: document.getElementById("editor-load-difficulty"),
  loadScenario: document.getElementById("editor-load-scenario"),
  loadMapButton: document.getElementById("editor-load-map-button"),
  newDifficulty: document.getElementById("editor-new-difficulty"),
  newScenarioName: document.getElementById("editor-new-scenario-name"),
  newDisplayName: document.getElementById("editor-new-display-name"),
  newWidth: document.getElementById("editor-new-width"),
  newHeight: document.getElementById("editor-new-height"),
  createMapButton: document.getElementById("editor-create-map-button"),
  directionButtons: [...document.querySelectorAll(".editor-direction-button")],
};

function setStatus(value) {
  elements.status.textContent = value;
}

function clampSelectedCell(payload) {
  const width = Number(payload?.width || 1);
  const height = Number(payload?.height || 1);
  const minX = width > 2 ? 1 : 0;
  const minY = height > 2 ? 1 : 0;
  const maxX = width > 2 ? width - 2 : width - 1;
  const maxY = height > 2 ? height - 2 : height - 1;
  state.selectedCell = [
    Math.min(Math.max(state.selectedCell[0], minX), maxX),
    Math.min(Math.max(state.selectedCell[1], minY), maxY),
  ];
}

function selectedPaletteItem() {
  return (state.payload?.palette || []).find((item) => item.token === state.selectedToken) || null;
}

function catalogDifficulties() {
  return Array.isArray(state.catalog?.difficulties) ? state.catalog.difficulties : [];
}

function difficultyEntry(difficultyId) {
  return catalogDifficulties().find((entry) => entry.id === difficultyId) || null;
}

function difficultyLabel(entry) {
  if (!entry) {
    return "-";
  }
  const sizeText = entry.fixedSize
    ? `${entry.defaultWidth}x${entry.defaultHeight}`
    : `default ${entry.defaultWidth}x${entry.defaultHeight}`;
  return `${entry.label} · ${sizeText}`;
}

function setSelectOptions(select, options, preferredValue) {
  const previousValue = String(select.value || "");
  select.innerHTML = "";

  if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No entries";
    select.append(option);
    select.disabled = true;
    select.value = "";
    return "";
  }

  for (const item of options) {
    const option = document.createElement("option");
    option.value = String(item.value);
    option.textContent = String(item.label);
    select.append(option);
  }

  select.disabled = false;
  const preferredText = String(preferredValue || "");
  const resolvedValue = options.some((item) => String(item.value) === preferredText)
    ? preferredText
    : options.some((item) => String(item.value) === previousValue)
      ? previousValue
      : String(options[0].value);
  select.value = resolvedValue;
  return resolvedValue;
}

function syncNewMapSizeInputs(forceDefaults = false) {
  const entry = difficultyEntry(elements.newDifficulty.value);
  if (!entry) {
    elements.newWidth.disabled = true;
    elements.newHeight.disabled = true;
    return;
  }

  const widthValue = normalizeNullableInteger(elements.newWidth.value);
  const heightValue = normalizeNullableInteger(elements.newHeight.value);
  if (entry.fixedSize) {
    elements.newWidth.value = String(entry.defaultWidth);
    elements.newHeight.value = String(entry.defaultHeight);
    elements.newWidth.disabled = true;
    elements.newHeight.disabled = true;
    return;
  }

  elements.newWidth.disabled = false;
  elements.newHeight.disabled = false;
  if (forceDefaults || widthValue === null) {
    elements.newWidth.value = String(entry.defaultWidth);
  }
  if (forceDefaults || heightValue === null) {
    elements.newHeight.value = String(entry.defaultHeight);
  }
}

function updateCatalogControls() {
  const difficulties = catalogDifficulties();
  const difficultyOptions = difficulties.map((entry) => ({
    value: entry.id,
    label: difficultyLabel(entry),
  }));

  const preferredLoadDifficulty = String(elements.loadDifficulty.value || state.payload?.difficulty || "");
  const loadDifficulty = setSelectOptions(elements.loadDifficulty, difficultyOptions, preferredLoadDifficulty);
  const loadDifficultyEntry = difficultyEntry(loadDifficulty);
  const maps = Array.isArray(loadDifficultyEntry?.maps) ? loadDifficultyEntry.maps : [];
  const mapOptions = maps.map((entry) => ({
    value: entry.mapPath,
    label: `${entry.label} · ${entry.mapFileName}`,
  }));
  const preferredMapPath = maps.some((entry) => entry.mapPath === elements.loadScenario.value)
    ? elements.loadScenario.value
    : maps.some((entry) => entry.mapPath === state.payload?.mapPath)
      ? state.payload.mapPath
      : maps.find((entry) => entry.isCurrent)?.mapPath || "";
  const selectedMapPath = setSelectOptions(elements.loadScenario, mapOptions, preferredMapPath);
  const selectedMap = maps.find((entry) => entry.mapPath === selectedMapPath) || null;

  const preferredNewDifficulty = String(elements.newDifficulty.value || state.payload?.difficulty || "");
  setSelectOptions(elements.newDifficulty, difficultyOptions, preferredNewDifficulty);
  syncNewMapSizeInputs(false);

  if (!String(elements.newScenarioName.placeholder || "").trim()) {
    elements.newScenarioName.placeholder = "custom_map";
  }
  if (!String(elements.newDisplayName.placeholder || "").trim()) {
    elements.newDisplayName.placeholder = "Optional title";
  }

  const totalMaps = Number(state.catalog?.totalMaps || 0);
  elements.catalogMeta.textContent = formatLines([
    `catalog maps: ${totalMaps}`,
    `difficulty: ${loadDifficultyEntry ? difficultyLabel(loadDifficultyEntry) : "-"}`,
    `available scenarios: ${maps.length}`,
    `selected: ${selectedMap ? selectedMap.label : "-"}`,
    `selected size: ${selectedMap ? `${selectedMap.width}x${selectedMap.height}` : "-"}`,
  ]);
}

function updateDirectionButtons() {
  elements.directionButtons.forEach((button) => {
    const value = button.dataset.direction;
    const normalized = value === "" ? null : Number(value);
    button.classList.toggle("active", normalized === state.selectedDirection);
  });
}

function buildPaletteVisual(item) {
  const visual = document.createElement("span");
  visual.className = "palette-button-visual";

  if (item.sprite) {
    const spritePath = spritePathFor({
      kind: item.sprite.kind,
      word: item.sprite.word,
      spriteKey: item.sprite.word,
    });
    if (spritePath) {
      const image = document.createElement("img");
      image.className = "palette-button-sprite";
      image.src = spritePath;
      image.alt = item.label;
      image.addEventListener("error", () => {
        visual.classList.add("is-missing");
        image.remove();
      });
      visual.append(image);
    }

    const fallback = document.createElement("span");
    fallback.className = "palette-fallback";
    fallback.textContent = String(item.token || "?");
    visual.append(fallback);
    return visual;
  }

  const swatch = document.createElement("span");
  swatch.className = item.token === "#"
    ? "palette-swatch palette-swatch-border"
    : "palette-swatch palette-swatch-empty";
  visual.append(swatch);
  return visual;
}

function updatePalette() {
  elements.palette.innerHTML = "";
  for (const item of state.payload?.palette || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button palette-button";
    button.title = item.label;
    button.setAttribute("aria-label", item.label);
    button.classList.toggle("active", item.token === state.selectedToken);
    button.append(buildPaletteVisual(item));
    button.addEventListener("click", () => {
      state.selectedToken = item.token;
      render(state.payload);
    });
    elements.palette.append(button);
  }
}

function setBusy(value) {
  state.busy = value;
  const disabled = value || !state.payload;
  const newDifficultyConfig = difficultyEntry(elements.newDifficulty.value);
  const hasLoadMaps = Boolean(difficultyEntry(elements.loadDifficulty.value)?.maps?.length);
  elements.saveButton.disabled = disabled;
  elements.clearInteriorButton.disabled = disabled;
  elements.loadDifficulty.disabled = value || !catalogDifficulties().length;
  elements.loadScenario.disabled = value || !hasLoadMaps;
  elements.loadMapButton.disabled = value || !elements.loadScenario.value;
  elements.newDifficulty.disabled = value || !catalogDifficulties().length;
  elements.newScenarioName.disabled = value;
  elements.newDisplayName.disabled = value;
  elements.newWidth.disabled = value || !newDifficultyConfig || Boolean(newDifficultyConfig.fixedSize);
  elements.newHeight.disabled = value || !newDifficultyConfig || Boolean(newDifficultyConfig.fixedSize);
  elements.createMapButton.disabled = value || !elements.newDifficulty.value;
}

function render(payload) {
  state.payload = payload;
  setStatus(payload?.dirty ? "dirty" : payload ? "ready" : "idle");

  if (!payload) {
    elements.meta.textContent = "editor session unavailable";
    elements.catalogMeta.textContent = "";
    renderBoard(null, elements.board, { emptyMessage: "" });
    setBusy(false);
    return;
  }

  clampSelectedCell(payload);
  if (!(payload.palette || []).some((item) => item.token === state.selectedToken) && payload.palette?.length) {
    state.selectedToken = payload.palette[0].token;
  }

  const selectedToken = selectedPaletteItem();
  elements.meta.textContent = formatLines([
    `file: ${payload.mapFileName}`,
    `path: ${payload.mapPath}`,
    `difficulty: ${payload.difficulty}`,
    `scenario: ${payload.scenarioName}`,
    `display: ${payload.displayName || "-"}`,
    `size: ${payload.width}x${payload.height}`,
    `dirty: ${String(payload.dirty)}`,
    `selected cell: (${state.selectedCell[0]}, ${state.selectedCell[1]})`,
    `selected token: ${selectedToken ? `${selectedToken.label} [${selectedToken.token}]` : state.selectedToken}`,
    `selected direction: ${state.selectedDirection === null ? "none" : state.selectedDirection}`,
    `status: ${payload.status}`,
  ]);

  renderBoard(payload.scene, elements.board, {
    tileSize: 44,
    fitToHost: true,
    minTileSize: 24,
    maxTileSize: 68,
    fallbackHostHeight: Math.min(window.innerHeight * 0.78, 920),
    selectedCell: state.selectedCell,
    onCellClick: async (x, y, event) => {
      state.selectedCell = [x, y];
      if (event.shiftKey) {
        await withBusy(() => clearSelectedCell());
      } else {
        await withBusy(() => appendSelectedToken());
      }
    },
    onCellContextMenu: async (x, y) => {
      state.selectedCell = [x, y];
      await withBusy(() => popSelectedCell());
    },
  });

  updatePalette();
  updateDirectionButtons();
  updateCatalogControls();
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

async function loadCatalog() {
  state.catalog = await fetchJson("/api/catalog");
  updateCatalogControls();
}

async function loadInitialState() {
  const [payload, catalog] = await Promise.all([
    fetchJson("/api/session"),
    fetchJson("/api/catalog"),
  ]);
  state.catalog = catalog;
  render(payload);
}

async function editCell(changePayload) {
  const payload = await fetchJson("/api/session/cells", {
    method: "POST",
    body: JSON.stringify(changePayload),
  });
  render(payload);
}

async function appendSelectedToken() {
  const [x, y] = state.selectedCell;
  await editCell({
    x,
    y,
    mode: "append",
    token: state.selectedToken,
    direction: state.selectedDirection,
  });
}

async function popSelectedCell() {
  const [x, y] = state.selectedCell;
  await editCell({ x, y, mode: "pop" });
}

async function clearSelectedCell() {
  const [x, y] = state.selectedCell;
  await editCell({ x, y, mode: "clear" });
}

async function saveSession() {
  const payload = await fetchJson("/api/session/save", {
    method: "POST",
  });
  render(payload);
  await loadCatalog();
}

async function clearInterior() {
  const payload = await fetchJson("/api/session/clear-interior", {
    method: "POST",
  });
  render(payload);
}

async function openSelectedMap() {
  const mapPath = String(elements.loadScenario.value || "").trim();
  if (!mapPath) {
    return;
  }
  const payload = await fetchJson("/api/session/open", {
    method: "POST",
    body: JSON.stringify({ mapPath }),
  });
  render(payload);
  await loadCatalog();
}

async function createNewMapSession() {
  const payload = await fetchJson("/api/session/new", {
    method: "POST",
    body: JSON.stringify({
      difficulty: String(elements.newDifficulty.value || "").trim(),
      scenarioName: String(elements.newScenarioName.value || "").trim() || null,
      displayName: String(elements.newDisplayName.value || "").trim() || null,
      width: normalizeNullableInteger(elements.newWidth.value),
      height: normalizeNullableInteger(elements.newHeight.value),
    }),
  });
  render(payload);
  await loadCatalog();
}

function moveSelection(dx, dy) {
  if (!state.payload) return;
  clampSelectedCell(state.payload);
  const width = Number(state.payload.width || 1);
  const height = Number(state.payload.height || 1);
  const minX = width > 2 ? 1 : 0;
  const minY = height > 2 ? 1 : 0;
  const maxX = width > 2 ? width - 2 : width - 1;
  const maxY = height > 2 ? height - 2 : height - 1;
  state.selectedCell = [
    Math.min(Math.max(state.selectedCell[0] + dx, minX), maxX),
    Math.min(Math.max(state.selectedCell[1] + dy, minY), maxY),
  ];
  render(state.payload);
}

function handleTokenHotkey(key) {
  const palette = state.payload?.palette || [];
  const item = palette.find((entry) => entry.token === key);
  if (!item) return false;
  state.selectedToken = item.token;
  render(state.payload);
  return true;
}

function registerHotkeys() {
  window.addEventListener("keydown", (event) => {
    if (event.repeat || shouldIgnoreHotkeys(event.target) || state.busy || !state.payload) {
      return;
    }

    const lower = event.key.toLowerCase();
    let command = null;
    if (event.key === "ArrowUp") command = () => moveSelection(0, -1);
    else if (event.key === "ArrowDown") command = () => moveSelection(0, 1);
    else if (event.key === "ArrowLeft") command = () => moveSelection(-1, 0);
    else if (event.key === "ArrowRight") command = () => moveSelection(1, 0);
    else if (event.key === " " || event.key === "Enter") command = () => withBusy(() => appendSelectedToken());
    else if (event.key === "Backspace" || event.key === "Delete") {
      command = () => withBusy(() => (event.shiftKey ? clearSelectedCell() : popSelectedCell()));
    } else if (lower === "s") command = () => withBusy(() => saveSession());
    else if (lower === "c") command = () => withBusy(() => clearInterior());
    else if (lower === "x") {
      command = () => {
        state.selectedDirection = null;
        updateDirectionButtons();
        render(state.payload);
      };
    } else if (["0", "1", "2", "3"].includes(event.key)) {
      command = () => {
        state.selectedDirection = Number(event.key);
        updateDirectionButtons();
        render(state.payload);
      };
    } else if (handleTokenHotkey(event.key)) {
      event.preventDefault();
      return;
    }

    if (command) {
      event.preventDefault();
      command();
    }
  });
}

function registerClicks() {
  elements.saveButton.addEventListener("click", () => withBusy(() => saveSession()));
  elements.clearInteriorButton.addEventListener("click", () => withBusy(() => clearInterior()));
  elements.loadMapButton.addEventListener("click", () => withBusy(() => openSelectedMap()));
  elements.createMapButton.addEventListener("click", () => withBusy(() => createNewMapSession()));
  elements.loadDifficulty.addEventListener("change", () => updateCatalogControls());
  elements.loadScenario.addEventListener("change", () => updateCatalogControls());
  elements.newDifficulty.addEventListener("change", () => {
    syncNewMapSizeInputs(true);
    updateCatalogControls();
  });
  elements.directionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const value = button.dataset.direction;
      state.selectedDirection = value === "" ? null : Number(value);
      updateDirectionButtons();
      render(state.payload);
    });
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
  await loadInitialState();
}

boot().catch((error) => {
  elements.meta.textContent = formatError(error);
});
