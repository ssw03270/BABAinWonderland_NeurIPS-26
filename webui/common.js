export function formatError(error) {
  return String(error?.message || error || "unknown error");
}

export function formatLines(lines) {
  return lines.filter((line) => line !== null && line !== undefined && line !== "").join("\n");
}

export async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === "string") {
        message = payload.detail;
      }
    } catch (_error) {
      // Keep the default message.
    }
    throw new Error(message);
  }
  return response.json();
}

export function normalizeNullableInteger(rawValue) {
  if (rawValue === null || rawValue === undefined || rawValue === "") {
    return null;
  }
  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null;
}

export function spritePathFor(objectPayload) {
  const spriteKey = String(objectPayload?.spriteKey || objectPayload?.word || "").trim();
  if (!spriteKey) {
    return null;
  }
  const group = objectPayload?.kind === "rule" ? "text" : "icon";
  return `/assets/babagui_sprites/${group}/${spriteKey.toUpperCase()}.gif`;
}

function spriteFallbackLabel(objectPayload) {
  return String(objectPayload?.word || "?").toUpperCase().slice(0, 5);
}

function sceneSignature(scenePayload, selectedCell, tileSize) {
  return JSON.stringify({
    gridSize: scenePayload?.gridSize || null,
    cells: scenePayload?.cells || null,
    selectedCell: selectedCell || null,
    tileSize,
  });
}

function directionGlyph(direction) {
  const normalized = String(direction || "").toLowerCase();
  if (normalized.includes("up")) return "\u2191";
  if (normalized.includes("down")) return "\u2193";
  if (normalized.includes("left")) return "\u2190";
  if (normalized.includes("right")) return "\u2192";
  return "";
}

function spriteClassSuffix(rawValue) {
  return String(rawValue || "unknown")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
}

function createSpriteLayer(objectPayload, offsetIndex, visibleCount) {
  const layer = document.createElement("div");
  layer.className = `sprite-layer ${objectPayload.kind === "rule" ? "rule" : "world"}`;
  layer.classList.add(`obj-${spriteClassSuffix(objectPayload.objType)}`);
  layer.classList.add(`word-${spriteClassSuffix(objectPayload.word)}`);
  const offsetsByCount = {
    1: [{ x: 0, y: 0 }],
    2: [{ x: -4, y: 4 }, { x: 4, y: -4 }],
    3: [{ x: -6, y: 6 }, { x: 0, y: 0 }, { x: 6, y: -6 }],
  };
  const offsets = offsetsByCount[visibleCount] || offsetsByCount[1];
  const activeOffset = offsets[offsetIndex] || offsets[offsets.length - 1];
  layer.style.transform = `translate(${activeOffset.x}px, ${activeOffset.y}px)`;
  if (objectPayload.highlightKind) {
    layer.classList.add(`highlight-${objectPayload.highlightKind}`);
  }

  const fallback = document.createElement("span");
  fallback.className = "sprite-fallback";
  fallback.textContent = spriteFallbackLabel(objectPayload);

  const spritePath = typeof objectPayload?.spritePath === "string"
    ? objectPayload.spritePath
    : spritePathFor(objectPayload);
  if (spritePath) {
    const image = document.createElement("img");
    image.src = spritePath;
    image.alt = objectPayload.word;
    image.addEventListener("error", () => {
      layer.classList.add("missing");
      image.remove();
    });
    layer.append(image, fallback);
  } else {
    layer.classList.add("missing");
    layer.append(fallback);
  }
  return layer;
}

function resolveTileSize(scenePayload, hostElement, options) {
  const {
    tileSize = 64,
    fitToHost = false,
    fitAxis = "both",
    minTileSize = 20,
    maxTileSize = 96,
    hostPadding = 24,
    fallbackHostHeight = null,
  } = options;

  if (!fitToHost || !scenePayload || !Array.isArray(scenePayload.gridSize)) {
    return Number(tileSize);
  }

  const [width, height] = scenePayload.gridSize;
  const hostRect = hostElement.getBoundingClientRect();
  const availableWidth = Math.max(160, Math.floor(hostRect.width - Number(hostPadding)));
  const heightSource = hostRect.height > 0 ? hostRect.height : Number(fallbackHostHeight || 0);
  const availableHeight = Math.max(140, Math.floor(heightSource - Number(hostPadding)));
  const byWidth = Math.floor(availableWidth / Math.max(1, Number(width)));
  const byHeight = Math.floor(availableHeight / Math.max(1, Number(height)));
  const normalizedFitAxis = String(fitAxis || "both").toLowerCase();
  let fitted = Math.min(byWidth, byHeight);
  if (normalizedFitAxis === "width") {
    fitted = byWidth;
  } else if (normalizedFitAxis === "height") {
    fitted = byHeight;
  }
  return Math.max(Number(minTileSize), Math.min(Number(maxTileSize), fitted || Number(tileSize)));
}

export function renderBoard(scenePayload, hostElement, options = {}) {
  const {
    onCellClick = null,
    onCellContextMenu = null,
    emptyMessage = "scene unavailable",
    selectedCell = null,
  } = options;

  if (!scenePayload || !Array.isArray(scenePayload.gridSize)) {
    hostElement.innerHTML = "";
    delete hostElement.dataset.sceneSignature;
    hostElement.textContent = emptyMessage;
    return;
  }

  const [width, height] = scenePayload.gridSize;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    hostElement.innerHTML = "";
    delete hostElement.dataset.sceneSignature;
    hostElement.textContent = "invalid scene dimensions";
    return;
  }

  const resolvedTileSize = resolveTileSize(scenePayload, hostElement, options);
  const signature = sceneSignature(scenePayload, selectedCell, resolvedTileSize);
  if (hostElement.dataset.sceneSignature === signature) {
    return;
  }

  const board = document.createElement("div");
  board.className = "board";
  board.style.setProperty("--tile-size", `${resolvedTileSize}px`);
  board.style.gridTemplateColumns = `repeat(${width}, ${resolvedTileSize}px)`;

  const cellMap = new Map();
  for (const cell of scenePayload.cells || []) {
    cellMap.set(`${cell.x},${cell.y}`, cell);
  }

  const selectedKey = Array.isArray(selectedCell) ? `${selectedCell[0]},${selectedCell[1]}` : null;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const cellElement = document.createElement("div");
      cellElement.className = "board-cell";
      if (`${x},${y}` === selectedKey) {
        cellElement.classList.add("selected");
      }
      if (typeof onCellClick === "function") {
        cellElement.addEventListener("click", (event) => onCellClick(x, y, event));
      }
      if (typeof onCellContextMenu === "function") {
        cellElement.addEventListener("contextmenu", (event) => {
          event.preventDefault();
          onCellContextMenu(x, y, event);
        });
      }

      const payload = cellMap.get(`${x},${y}`);
      if (payload && Array.isArray(payload.objects) && payload.objects.length > 0) {
        const orderedObjects = payload.objects.slice();
        const visibleObjects = orderedObjects.slice(-3);
        const stack = document.createElement("div");
        stack.className = "cell-stack";
        visibleObjects.forEach((objectPayload, index) => {
          stack.append(createSpriteLayer(objectPayload, index, visibleObjects.length));
        });
        cellElement.append(stack);

        const topObject = visibleObjects[visibleObjects.length - 1];
        if (topObject && topObject.direction) {
          const badge = document.createElement("div");
          badge.className = "direction-badge";
          badge.textContent = directionGlyph(topObject.direction);
          cellElement.append(badge);
        }
        if (Number(payload.stackCount) > visibleObjects.length) {
          const stackBadge = document.createElement("div");
          stackBadge.className = "stack-badge";
          stackBadge.textContent = String(payload.stackCount);
          cellElement.append(stackBadge);
        }
      }

      board.append(cellElement);
    }
  }

  hostElement.innerHTML = "";
  hostElement.append(board);
  hostElement.dataset.sceneSignature = signature;
}

export function shouldIgnoreHotkeys(target) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  if (tagName === "input" || tagName === "textarea" || tagName === "select") {
    return true;
  }
  return Boolean(target.closest("[contenteditable='true']"));
}
