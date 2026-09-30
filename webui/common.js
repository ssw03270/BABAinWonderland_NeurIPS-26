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

let mapDisplayNameIndex = new Map();
let mapDisplayNameLoadPromise = null;

function mapDisplayLookupKeys(rawValue) {
  const text = String(rawValue || "").trim();
  if (!text) {
    return [];
  }
  const keys = [];
  const seen = new Set();
  const slashNormalized = text.replaceAll("\\", "/");
  const basename = slashNormalized.split("/").pop() || text;
  for (const candidate of [text, basename]) {
    const normalized = String(candidate || "").trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    keys.push(normalized);
    if (normalized.toLowerCase().endsWith(".json")) {
      const stem = normalized.slice(0, -5).trim();
      if (stem && !seen.has(stem)) {
        seen.add(stem);
        keys.push(stem);
      }
    }
  }
  return keys;
}

export async function loadMapDisplayNames() {
  if (!mapDisplayNameLoadPromise) {
    mapDisplayNameLoadPromise = fetchJson("/api/map-display-names")
      .then((payload) => {
        const nextIndex = new Map();
        const rawDisplayNames = payload?.displayNames;
        if (rawDisplayNames && typeof rawDisplayNames === "object") {
          for (const [rawKey, rawValue] of Object.entries(rawDisplayNames)) {
            const key = String(rawKey || "").trim();
            const value = String(rawValue || "").trim();
            if (!key || !value) {
              continue;
            }
            nextIndex.set(key, value);
          }
        }
        mapDisplayNameIndex = nextIndex;
        return nextIndex;
      })
      .catch(() => mapDisplayNameIndex);
  }
  return mapDisplayNameLoadPromise;
}

export function formatMapDisplayName(...candidates) {
  for (const candidate of candidates) {
    for (const key of mapDisplayLookupKeys(candidate)) {
      const displayName = String(mapDisplayNameIndex.get(key) || "").trim();
      if (displayName) {
        return displayName;
      }
    }
  }
  for (const candidate of candidates) {
    const text = String(candidate || "").trim();
    if (text) {
      return text;
    }
  }
  return null;
}

export function normalizeNullableInteger(rawValue) {
  if (rawValue === null || rawValue === undefined || rawValue === "") {
    return null;
  }
  const parsed = Number(rawValue);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null;
}

const RULE_OBJECT_TYPES = new Set(["rule_noun", "rule_operator", "rule_property"]);
const RULE_END_TYPES = new Set(["rule_noun", "rule_property"]);
const HAZARD_PROPERTIES = new Set(["defeat", "hot", "sink", "melt"]);
const TYPE_ALIASES = new Map([
  ["a", "rule_noun"],
  ["b", "rule_operator"],
  ["c", "rule_property"],
  ["d", "world_object"],
  ["rule_noun", "rule_noun"],
  ["rule_operator", "rule_operator"],
  ["rule_property", "rule_property"],
  ["world_object", "world_object"],
  ["noun", "rule_noun"],
  ["operator", "rule_operator"],
  ["property", "rule_property"],
  ["object", "world_object"],
]);

function normalizeWord(rawValue) {
  return String(rawValue || "unknown").trim().toLowerCase() || "unknown";
}

function normalizeStateObjectType(rawValue) {
  const normalized = normalizeWord(rawValue);
  return TYPE_ALIASES.get(normalized) || normalized;
}

function resolveReverseWordAliases(visualConfig) {
  const rawAliases = visualConfig?.reverseWordAliases || visualConfig?.reverse_word_aliases;
  if (!rawAliases || typeof rawAliases !== "object") {
    return {};
  }
  const normalized = {};
  for (const [rawType, rawMapping] of Object.entries(rawAliases)) {
    const objType = normalizeStateObjectType(rawType);
    if (!rawMapping || typeof rawMapping !== "object") {
      continue;
    }
    const aliasMap = {};
    for (const [rawAlias, rawWord] of Object.entries(rawMapping)) {
      aliasMap[normalizeWord(rawAlias)] = normalizeWord(rawWord);
    }
    if (Object.keys(aliasMap).length > 0) {
      normalized[objType] = aliasMap;
    }
  }
  return normalized;
}

function canonicalizeVisualWord(objType, rawWord, visualConfig) {
  const normalizedType = normalizeStateObjectType(objType);
  const normalizedWord = normalizeWord(rawWord);
  const reverseWordAliases = resolveReverseWordAliases(visualConfig);
  return reverseWordAliases[normalizedType]?.[normalizedWord] || normalizedWord;
}

function canonicalizeVisualState(statePayload, visualConfig) {
  if (!statePayload || typeof statePayload !== "object") {
    return null;
  }
  const objects = Array.isArray(statePayload.objects) ? statePayload.objects : [];
  return {
    ...statePayload,
    objects: objects
      .filter((obj) => obj && typeof obj === "object")
      .map((obj) => {
        const objType = normalizeStateObjectType(obj.type);
        const word = canonicalizeVisualWord(objType, obj.word ?? obj.text, visualConfig);
        const normalized = {
          ...obj,
          type: objType,
          word,
        };
        delete normalized.text;
        delete normalized.sprite_key;
        delete normalized.spriteKey;
        return normalized;
      }),
  };
}

function extractGridSizeFromState(statePayload) {
  if (Array.isArray(statePayload?.grid_size) && statePayload.grid_size.length === 2) {
    const width = Number(statePayload.grid_size[0]);
    const height = Number(statePayload.grid_size[1]);
    if (Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0) {
      return [width, height];
    }
  }
  let maxX = -1;
  let maxY = -1;
  for (const obj of statePayload?.objects || []) {
    if (!obj || typeof obj !== "object" || !Array.isArray(obj.position) || obj.position.length !== 2) {
      continue;
    }
    const x = Number(obj.position[0]);
    const y = Number(obj.position[1]);
    if (!Number.isInteger(x) || !Number.isInteger(y)) {
      continue;
    }
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return maxX >= 0 && maxY >= 0 ? [maxX + 1, maxY + 1] : [0, 0];
}

function extractRuleTriplesFromState(statePayload) {
  const byPosition = new Map();
  for (const obj of statePayload?.objects || []) {
    if (!obj || typeof obj !== "object") {
      continue;
    }
    const objType = normalizeStateObjectType(obj.type);
    if (!RULE_OBJECT_TYPES.has(objType)) {
      continue;
    }
    if (!Array.isArray(obj.position) || obj.position.length !== 2) {
      continue;
    }
    const x = Number(obj.position[0]);
    const y = Number(obj.position[1]);
    if (!Number.isInteger(x) || !Number.isInteger(y)) {
      continue;
    }
    const key = `${x},${y}`;
    const bucket = byPosition.get(key) || [];
    bucket.push([objType, normalizeWord(obj.word ?? obj.text)]);
    byPosition.set(key, bucket);
  }

  function optionsAt(x, y) {
    return byPosition.get(`${x},${y}`) || [];
  }

  function collectSubjectSegment(startX, startY, dx, dy) {
    const tokens = [];
    let x = startX;
    let y = startY;
    let seenNoun = false;
    let expectNoun = true;
    while (true) {
      const token = optionsAt(x, y).find(([objType]) =>
        objType === "rule_noun" || objType === "rule_property" || objType === "rule_operator",
      );
      if (!token) {
        break;
      }
      const [objType, word] = token;
      if (expectNoun) {
        if (objType === "rule_operator" && word === "and" && !seenNoun) {
          tokens.push(token);
        } else if (objType === "rule_noun") {
          tokens.push(token);
          seenNoun = true;
          expectNoun = false;
        } else {
          break;
        }
      } else if (objType === "rule_operator" && word === "and") {
        tokens.push(token);
        expectNoun = true;
      } else {
        break;
      }
      x += dx;
      y += dy;
    }
    return tokens.reverse();
  }

  function collectPredicateSegment(startX, startY, dx, dy) {
    const tokens = [];
    let x = startX;
    let y = startY;
    let seenItem = false;
    let expectItem = true;
    while (true) {
      const token = optionsAt(x, y).find(([objType]) =>
        objType === "rule_noun" || objType === "rule_property" || objType === "rule_operator",
      );
      if (!token) {
        break;
      }
      const [objType, word] = token;
      if (expectItem) {
        if (objType === "rule_operator" && word === "and" && !seenItem) {
          tokens.push(token);
        } else if (objType === "rule_noun" || objType === "rule_property") {
          tokens.push(token);
          seenItem = true;
          expectItem = false;
        } else {
          break;
        }
      } else if (objType === "rule_operator" && word === "and") {
        tokens.push(token);
        expectItem = true;
      } else {
        break;
      }
      x += dx;
      y += dy;
    }
    return tokens;
  }

  function parseConjoinedWords(tokens, allowedItemTypes) {
    if (!Array.isArray(tokens) || tokens.length === 0) {
      return null;
    }
    const trimmed = tokens.slice();
    while (trimmed.length > 0 && trimmed[0][0] === "rule_operator" && trimmed[0][1] === "and") {
      trimmed.shift();
    }
    while (
      trimmed.length > 0
      && trimmed[trimmed.length - 1][0] === "rule_operator"
      && trimmed[trimmed.length - 1][1] === "and"
    ) {
      trimmed.pop();
    }
    if (trimmed.length === 0) {
      return null;
    }
    const allowed = new Set(Array.from(allowedItemTypes || []).map((value) => normalizeWord(value)));
    const words = [];
    let expectItem = true;
    for (const [objType, word] of trimmed) {
      if (expectItem) {
        if (!allowed.has(objType)) {
          return null;
        }
        words.push(word);
      } else if (objType !== "rule_operator" || word !== "and") {
        return null;
      }
      expectItem = !expectItem;
    }
    return expectItem ? null : words;
  }

  const triples = [];
  const seen = new Set();
  for (const [key, operators] of byPosition.entries()) {
    if (!operators.some(([objType, word]) => objType === "rule_operator" && word === "is")) {
      continue;
    }
    const [x, y] = key.split(",").map((value) => Number(value));
    for (const [dx, dy] of [[1, 0], [0, 1]]) {
      const subjects = parseConjoinedWords(
        collectSubjectSegment(x - dx, y - dy, -dx, -dy),
        ["rule_noun"],
      );
      if (!subjects) {
        continue;
      }
      const predicates = parseConjoinedWords(
        collectPredicateSegment(x + dx, y + dy, dx, dy),
        Array.from(RULE_END_TYPES),
      );
      if (!predicates) {
        continue;
      }
      for (const subject of subjects) {
        for (const predicate of predicates) {
          const tripleKey = `${subject}|is|${predicate}`;
          if (!seen.has(tripleKey)) {
            seen.add(tripleKey);
            triples.push([subject, "is", predicate]);
          }
        }
      }
    }
  }
  return triples;
}

function extractActivePropertyMapFromState(statePayload) {
  const active = new Map();
  for (const [lhs, , rhs] of extractRuleTriplesFromState(statePayload)) {
    const bucket = active.get(lhs) || new Set();
    bucket.add(rhs);
    active.set(lhs, bucket);
  }
  return active;
}

function determineHighlightKind(properties) {
  if (properties?.has("you")) return "you";
  if (properties?.has("win")) return "win";
  for (const property of HAZARD_PROPERTIES) {
    if (properties?.has(property)) {
      return "danger";
    }
  }
  return null;
}

function normalizeDirection(direction) {
  const normalized = String(direction || "").trim().toLowerCase();
  return normalized || null;
}

export function spritePathFor(objectPayload) {
  const spriteKey = String(objectPayload?.spriteKey || objectPayload?.word || "").trim();
  if (!spriteKey) {
    return null;
  }
  const group = objectPayload?.kind === "rule" ? "text" : "icon";
  return `/assets/babagui_sprites/${group}/${spriteKey.toUpperCase()}.gif`;
}

function sceneObjectCountKey(sceneObject) {
  return [
    String(sceneObject?.kind || ""),
    String(sceneObject?.objType || ""),
    String(sceneObject?.word || ""),
  ].join("|");
}

export function buildStateScene(statePayload, visualConfig = null) {
  const visualState = canonicalizeVisualState(statePayload, visualConfig);
  if (!visualState) {
    return null;
  }
  const [width, height] = extractGridSizeFromState(visualState);
  if (width <= 0 || height <= 0) {
    return { gridSize: [0, 0], cells: [] };
  }
  const propertyMap = extractActivePropertyMapFromState(visualState);
  const entries = [];
  const countByObject = new Map();
  for (const obj of (visualState.objects || [])) {
    if (!obj || typeof obj !== "object" || !Array.isArray(obj.position) || obj.position.length !== 2) {
      continue;
    }
    const x = Number(obj.position[0]);
    const y = Number(obj.position[1]);
    if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= width || y >= height) {
      continue;
    }
    const objType = normalizeStateObjectType(obj.type);
    const word = normalizeWord(obj.word ?? obj.text);
    const kind = RULE_OBJECT_TYPES.has(objType) ? "rule" : "world";
    const sceneObject = {
      kind,
      word,
      objType: kind === "world" ? (objType || "world_object") : objType,
      spriteKey: word,
    };
    const direction = normalizeDirection(obj.direction);
    if (direction) {
      sceneObject.direction = direction;
    }
    if (kind === "world") {
      const highlightKind = determineHighlightKind(propertyMap.get(word) || new Set());
      if (highlightKind) {
        sceneObject.highlightKind = highlightKind;
      }
    }
    entries.push({ x, y, sceneObject });
    const countKey = sceneObjectCountKey(sceneObject);
    countByObject.set(countKey, (countByObject.get(countKey) || 0) + 1);
  }

  const byCell = new Map();
  for (const entry of entries) {
    const key = `${entry.x},${entry.y}`;
    const bucket = byCell.get(key) || [];
    bucket.push(entry.sceneObject);
    byCell.set(key, bucket);
  }

  const cells = Array.from(byCell.entries())
    .sort((left, right) => {
      const [leftX, leftY] = left[0].split(",").map((value) => Number(value));
      const [rightX, rightY] = right[0].split(",").map((value) => Number(value));
      return leftY - rightY || leftX - rightX;
    })
    .map(([key, objects]) => {
      const [x, y] = key.split(",").map((value) => Number(value));
      const resolvedObjects = objects.slice().sort(
        (left, right) =>
          (countByObject.get(sceneObjectCountKey(right)) || 0)
          - (countByObject.get(sceneObjectCountKey(left)) || 0),
      );
      return {
        x,
        y,
        stackCount: resolvedObjects.length,
        objects: resolvedObjects,
      };
    });

  return {
    gridSize: [width, height],
    cells,
  };
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
