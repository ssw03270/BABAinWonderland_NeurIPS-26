import {
  buildStateScene,
  fetchJson,
  formatError,
  formatLines,
  formatMapDisplayName,
  loadMapDisplayNames,
  renderBoard,
} from "/static/common.js";

const queryRunId = new URLSearchParams(window.location.search).get("run_id");
const viewerId = (() => {
  try {
    const storageKey = "baba-discovery-viewer-id";
    const existing = window.sessionStorage.getItem(storageKey);
    if (existing) return existing;
    const created = window.crypto?.randomUUID?.() || `viewer-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.sessionStorage.setItem(storageKey, created);
    return created;
  } catch (_error) {
    return `viewer-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
})();

const SERVER_SHUTDOWN_HEALTHCHECK_INITIAL_DELAY_MS = 750;
const SERVER_SHUTDOWN_HEALTHCHECK_INTERVAL_MS = 500;
const SERVER_SHUTDOWN_HEALTHCHECK_TIMEOUT_MS = 8000;
const SERVER_SHUTDOWN_RELOAD_DELAY_MS = 2000;
const RUN_DELETE_CONFIRMATION_TEXT = "Yes. Delete this run.";

const FIELD_LABELS = {
  episode: "EP",
  ep: "EP",
  step: "STEP",
  global_step: "GLOBAL",
  action: "ACTION",
  epsilon: "EPS",
  eps: "EPS",
  rh: "R_h",
  rz: "R_z",
  rtotal: "R_total",
  contrastive_loss: "CONTRASTIVE",
  prototype_top1_accuracy: "TOP1_ACC",
  rh_mean: "R_h_mean",
  rz_mean: "R_z_mean",
  rtotal_mean: "R_total_mean",
  rtotal_std: "RTOTAL_STD",
  current_dynamics_class: "CUR_CLASS",
  predicted_dynamics_class: "PRED_CLASS",
  predicted_dynamics_confidence: "PRED_CONF",
  softmax_temperature: "SOFTMAX_T",
  sample_store_size: "SAMPLES",
  one_step_exact_match_rate: "EXACT",
  terminated_match_rate: "TERM_MATCH",
  object_count_alignment_rate: "COUNT_ALIGN",
  object_identity_overlap_rate: "IDENTITY",
  object_exact_overlap_rate: "OBJ_OVERLAP",
  queued: "QUEUED",
  unknown_hit_ratio: "UNK_HIT",
  frontier_size: "TOTAL_FRONTIER",
  live_map_frontier_size: "MAP_FRONTIER",
};

const SERIES_LABELS = {
  contrastive_loss: "contrastive",
  prototype_top1_accuracy: "top-1",
  rh: "R_h",
  rz: "R_z",
  rtotal: "R_total",
  rh_mean: "R_h",
  rz_mean: "R_z",
  rtotal_mean: "R_total",
  rtotal_std: "R std",
  train_iter: "train step",
  dataset_size: "dataset",
  added_count: "added",
  frontier_size: "total frontier",
  live_map_frontier_size: "map frontier",
  one_step_exact_match_rate: "exact",
  terminated_match_rate: "terminated",
  object_count_alignment_rate: "count align",
  object_identity_overlap_rate: "identity overlap",
  object_exact_overlap_rate: "object overlap",
};

const PREFIX_STATE_STYLE = {
  NEW: "state-new",
  ERR: "state-err",
  FAIL: "state-fail",
  OK: "state-ok",
  MISS: "state-miss",
  WAIT: "state-wait",
};

const OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS = ["expected", "predicted", "previous"];
const OFFLINE_EVAL_IMAGE_KINDS = [...OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS, "comparison"];
const OFFLINE_EVAL_SUMMARY_POLL_INTERVAL_MS = 50;
const OFFLINE_EVAL_SUMMARY_POLL_MIN_INTERVAL_MS = 50;
const OFFLINE_EVAL_CLASS_PAGE_SIZE = 200;

const CLASS_ROW_HEIGHT = 42;
const CLASS_ROW_GAP = 10;
const CLASS_ROW_STRIDE = CLASS_ROW_HEIGHT + CLASS_ROW_GAP;
const CLASS_ROW_OVERSCAN = 8;
const DEFAULT_POLL_INTERVAL_MS = 400;
const POLL_INTERVAL_STORAGE_KEY = "baba-discovery-poll-live-ms";
const LIVE_VIEW_AUTO_PAUSE_MS = 5 * 60 * 1000;
const LIVE_VIEW_COUNTDOWN_REFRESH_MS = 1000;
const OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY = "baba-discovery-offline-eval-export-run-id";
const MIN_POLL_INTERVAL_MS = 100;
const MAX_POLL_INTERVAL_MS = 10000;
const LOG_AUTO_FOLLOW_THRESHOLD_PX = 24;
const PAGE_TITLE_ROTATION_INTERVAL_MS = 10000;
const PAGE_TITLE_TRANSITION_TOTAL_MS = 5000;
const PAGE_TITLE_TRANSITION_OUT_MS = 2000;
const PAGE_TITLE_TRANSITION_IN_MS = 3000;
const DISCOVERY_PAGE_TITLES = [
  "BABA, keep going!",
  "BABA, this is the real run.",
  "BABA, stay focused.",
  "BABA, push the block.",
  "BABA, find a way.",
  "BABA, move out.",
  "BABA, do not stop.",
  "BABA, try the direct move.",
  "BABA, solve it one step at a time.",
  "BABA, keep the questions for later.",
  "BABA, add a little more effort.",
  "BABA, now is the moment.",
  "BABA, recover and continue.",
  "BABA, press forward.",
  "BABA, another hard-won success.",
  "BABA, staying alive is progress.",
  "BABA, advance anyway.",
  "BABA, keep producing outputs.",
  "BABA, you are here to solve it.",
].map((title) => title.trim()).filter(Boolean);

const PAGE_TITLE_TRANSITIONS = [
  {
    name: "explosion",
    out: [
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(-10px, 6px, 0) rotate(-5deg) scale(0.86)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(18px, -12px, 0) rotate(10deg) scale(1.32)", filter: "blur(1px) brightness(1.35)" },
      { opacity: 0, transform: "translate3d(0, -42px, 0) rotate(-18deg) scale(2.25)", filter: "blur(22px) brightness(1.9)" },
    ],
    outTiming: { duration: PAGE_TITLE_TRANSITION_OUT_MS, easing: "cubic-bezier(0.2, 0.9, 0.3, 1)", fill: "forwards" },
    in: [
      { opacity: 0, transform: "translate3d(0, 56px, 0) rotate(14deg) scale(0.16)", filter: "blur(26px) brightness(2)" },
      { opacity: 1, transform: "translate3d(-24px, -10px, 0) rotate(-8deg) scale(1.26)", filter: "blur(3px) brightness(1.3)" },
      { opacity: 1, transform: "translate3d(12px, 4px, 0) rotate(4deg) scale(0.92)", filter: "blur(1px)" },
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
    ],
    inTiming: { duration: PAGE_TITLE_TRANSITION_IN_MS, easing: "cubic-bezier(0.16, 1, 0.3, 1)", fill: "forwards" },
  },
  {
    name: "fly-offscreen",
    out: [
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(52px, -8px, 0) rotate(8deg) scale(1.08)", filter: "blur(1px)" },
      { opacity: 0, transform: "translate3d(180vw, -56px, 0) rotate(34deg) scale(0.58)", filter: "blur(18px)" },
    ],
    outTiming: { duration: PAGE_TITLE_TRANSITION_OUT_MS, easing: "cubic-bezier(0.18, 0.9, 0.32, 1)", fill: "forwards" },
    in: [
      { opacity: 0, transform: "translate3d(-180vw, 44px, 0) rotate(-28deg) scale(0.54)", filter: "blur(22px)" },
      { opacity: 1, transform: "translate3d(42px, -14px, 0) rotate(10deg) scale(1.18)", filter: "blur(4px)" },
      { opacity: 1, transform: "translate3d(-18px, 4px, 0) rotate(-4deg) scale(0.94)", filter: "blur(1px)" },
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
    ],
    inTiming: { duration: PAGE_TITLE_TRANSITION_IN_MS, easing: "cubic-bezier(0.15, 1, 0.25, 1)", fill: "forwards" },
  },
  {
    name: "over-rotation",
    out: [
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(14px, -8px, 0) rotate(360deg) scale(0.96)", filter: "blur(1px)" },
      { opacity: 0, transform: "translate3d(-22px, -18px, 0) rotate(1080deg) scale(0.42)", filter: "blur(20px)" },
    ],
    outTiming: { duration: PAGE_TITLE_TRANSITION_OUT_MS, easing: "cubic-bezier(0.32, 0.02, 0.7, 0.16)", fill: "forwards" },
    in: [
      { opacity: 0, transform: "translate3d(24px, 24px, 0) rotate(-1260deg) scale(0.2)", filter: "blur(24px)" },
      { opacity: 1, transform: "translate3d(-18px, -8px, 0) rotate(-180deg) scale(1.24)", filter: "blur(4px)" },
      { opacity: 1, transform: "translate3d(8px, 4px, 0) rotate(36deg) scale(0.92)", filter: "blur(1px)" },
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
    ],
    inTiming: { duration: PAGE_TITLE_TRANSITION_IN_MS, easing: "cubic-bezier(0.1, 1, 0.2, 1)", fill: "forwards" },
  },
  {
    name: "chaos-drop",
    out: [
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(-16px, 14px, 0) rotate(-8deg) scale(1.08)", filter: "blur(1px)" },
      { opacity: 0, transform: "translate3d(34px, 96px, 0) rotate(22deg) scale(0.62)", filter: "blur(14px)" },
    ],
    outTiming: { duration: PAGE_TITLE_TRANSITION_OUT_MS, easing: "cubic-bezier(0.55, 0.03, 0.68, 0.19)", fill: "forwards" },
    in: [
      { opacity: 0, transform: "translate3d(-18px, -150px, 0) rotate(-18deg) scale(1.26)", filter: "blur(18px)" },
      { opacity: 1, transform: "translate3d(22px, 42px, 0) rotate(16deg) scale(0.86)", filter: "blur(3px)" },
      { opacity: 1, transform: "translate3d(-12px, -10px, 0) rotate(-6deg) scale(1.08)", filter: "blur(1px)" },
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
    ],
    inTiming: { duration: PAGE_TITLE_TRANSITION_IN_MS, easing: "cubic-bezier(0.17, 0.89, 0.32, 1.28)", fill: "forwards" },
  },
  {
    name: "dogged-revival",
    out: [
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
      { opacity: 1, transform: "translate3d(0, 18px, 0) rotate(-4deg) scale(1.14, 0.76)", filter: "blur(2px)" },
      { opacity: 0, transform: "translate3d(0, 54px, 0) rotate(9deg) scale(0.14, 0.04)", filter: "blur(16px) brightness(0.7)" },
    ],
    outTiming: { duration: PAGE_TITLE_TRANSITION_OUT_MS, easing: "cubic-bezier(0.42, 0, 1, 1)", fill: "forwards" },
    in: [
      { opacity: 0, transform: "translate3d(0, 90px, 0) rotate(-12deg) scale(0.06, 0.02)", filter: "blur(22px) brightness(1.6)" },
      { opacity: 1, transform: "translate3d(0, -26px, 0) rotate(10deg) scale(1.42, 1.24)", filter: "blur(5px) brightness(1.4)" },
      { opacity: 1, transform: "translate3d(10px, 10px, 0) rotate(-5deg) scale(0.86)", filter: "blur(1px)" },
      { opacity: 1, transform: "translate3d(0, 0, 0) rotate(0deg) scale(1)", filter: "blur(0px)" },
    ],
    inTiming: { duration: PAGE_TITLE_TRANSITION_IN_MS, easing: "cubic-bezier(0.12, 0.9, 0.22, 1.18)", fill: "forwards" },
  },
];

function sanitizePollIntervalMs(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(MAX_POLL_INTERVAL_MS, Math.max(MIN_POLL_INTERVAL_MS, parsed));
}

function loadStoredPollIntervalMs() {
  try {
    return sanitizePollIntervalMs(window.localStorage.getItem(POLL_INTERVAL_STORAGE_KEY), DEFAULT_POLL_INTERVAL_MS);
  } catch (_error) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
}

function loadStoredOfflineEvalExportRunId() {
  try {
    return String(window.localStorage.getItem(OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY) || "").trim() || null;
  } catch (_error) {
    return null;
  }
}

function persistOfflineEvalExportRunId(runId) {
  const safeRunId = String(runId || "").trim();
  try {
    if (!safeRunId) {
      window.localStorage.removeItem(OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY, safeRunId);
  } catch (_error) {
    // Ignore storage failures and keep the in-memory export state for this session.
  }
}

function clearStoredOfflineEvalExportRunId(expectedRunId = null) {
  const safeExpectedRunId = String(expectedRunId || "").trim();
  try {
    const storedRunId = String(window.localStorage.getItem(OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY) || "").trim();
    if (safeExpectedRunId && storedRunId && storedRunId !== safeExpectedRunId) {
      return;
    }
    window.localStorage.removeItem(OFFLINE_EVAL_EXPORT_RUN_STORAGE_KEY);
  } catch (_error) {
    // Ignore storage failures and continue with the in-memory export state.
  }
}

const state = {
  runs: [],
  activeRunId: queryRunId,
  activeRunPayload: null,
  viewMode: "live",
  projectionMode: "live",
  projectionExcludeSparseClasses: false,
  pollHandle: null,
  liveViewStartedAtMs: Date.now(),
  liveViewCountdownHandle: null,
  pollInFlight: false,
  pollIntervalMs: loadStoredPollIntervalMs(),
  pollIntervalEditing: false,
  canShutdownServer: false,
  shutdownInFlight: false,
  tabsSignature: null,
  viewerId,
  lastViewerSyncAt: 0,
  lastViewerSyncKey: null,
  renderCache: {
    runKey: null,
    panels: {},
  },
  configSnapshotVisibilityByRun: {},
  classRowsScrollTopByRun: {},
  logScrollStateByRun: {},
  projectionPinnedClassByRun: {},
  projectionPinnedWorldByRun: {},
  projectionTargetWorldPanelExpandedByRun: {},
  targetWorldSelectionByRun: {},
  targetWorldPanelExpandedByRun: {},
  transitionSelectionByRun: {},
  transitionWitnessFailGroupsExpandedByRun: {},
  transitionWitnessSplitsExpandedByRun: {},
  programInspectorByRun: {},
  transitionBundleMetaByUrl: {},
  transitionBundleMetaRevisionByUrl: {},
  transitionBundleMetaRequestByUrl: {},
  pageTitleRotationHandle: null,
  pageTitleAnimating: false,
  pageTitleIndex: null,
  pageTitleActiveLayerIndex: 0,
  programInspectorText: null,
  programCopyResetHandle: null,
  programArtifactTextByUrl: {},
  programArtifactRevisionByUrl: {},
  programArtifactRequestByUrl: {},
  offlineEval: {
    defaults: null,
    defaultsPromise: null,
    runs: [],
    runsPromise: null,
    modalSource: null,
    modalRunId: null,
    modalRunMode: "accuracy",
    modalBusy: false,
    drawerOpen: false,
    runId: null,
    runSummary: null,
    classResults: [],
    classPageOffset: 0,
    classPageLimit: OFFLINE_EVAL_CLASS_PAGE_SIZE,
    classPageHasMore: false,
    classPageTotalCount: 0,
    classPageLoading: false,
    classPageHydrated: false,
    classPageUnavailable: false,
    classPageRequestToken: 0,
    selectedFailureId: null,
    selectedClassRowKey: null,
    selectedClassRepresentativeId: null,
    failureDetail: null,
    loadingFailureId: null,
    classRepresentativeDetail: null,
    loadingClassRepresentativeId: null,
    statusFilter: "failed",
    notice: "",
    imageFailureKey: null,
    activeImageKind: "expected",
    exportBusy: false,
    exportStatus: null,
    exportToken: 0,
    pollHandle: null,
    pollInFlight: false,
    restoreAttempted: false,
  },
  runDelete: {
    open: false,
    busy: false,
    runId: null,
    inputText: "",
    notice: "",
  },
};

const elements = {
  pageTitle: document.getElementById("discovery-page-title"),
  pageTitleLayers: Array.from(document.querySelectorAll("#discovery-page-title .discovery-page-title-layer")),
  status: document.getElementById("discovery-status"),
  shutdownButton: document.getElementById("discovery-shutdown-server"),
  viewModeButton: document.getElementById("discovery-view-mode"),
  projectionModeButton: document.getElementById("discovery-projection-mode"),
  projectionSparseFilterInput: document.getElementById("discovery-projection-sparse-filter"),
  livePollInput: document.getElementById("discovery-poll-live"),
  tabs: document.getElementById("discovery-run-tabs"),
  meta: document.getElementById("discovery-meta"),
  configSnapshots: document.getElementById("discovery-config-snapshots"),
  hud: document.getElementById("discovery-hud"),
  hudTitle: document.getElementById("discovery-hud-title"),
  hudColumn: document.getElementById("discovery-hud-column"),
  classRowsColumn: document.getElementById("discovery-class-rows-column"),
  classRowsTitle: document.getElementById("discovery-class-rows-title"),
  targetWorldsHead: document.getElementById("discovery-target-worlds-head"),
  targetWorldsTitle: document.getElementById("discovery-target-worlds-title"),
  board: document.getElementById("discovery-board"),
  boardWrap: document.getElementById("discovery-live-board-wrap"),
  liveFooter: document.querySelector(".discovery-live-footer"),
  targetWorlds: document.getElementById("discovery-target-worlds"),
  targetWorldBody: document.getElementById("discovery-target-worlds-body"),
  targetWorldList: document.getElementById("discovery-target-worlds-list"),
  targetWorldBoard: document.getElementById("discovery-target-world-board"),
  targetWorldMeta: document.getElementById("discovery-target-world-meta"),
  log: document.getElementById("discovery-log"),
  versionSummary: document.getElementById("discovery-version-summary"),
  versionList: document.getElementById("discovery-version-list"),
  programPanel: document.getElementById("discovery-program-panel"),
  programTitle: document.getElementById("discovery-program-title"),
  programMeta: document.getElementById("discovery-program-meta"),
  programTabs: document.getElementById("discovery-program-tabs"),
  programCode: document.getElementById("discovery-program-code"),
  programCopy: document.getElementById("discovery-program-copy"),
  programClose: document.getElementById("discovery-program-close"),
  transitionStage: document.getElementById("discovery-transition-stage"),
  transitionToolbar: document.getElementById("discovery-transition-toolbar"),
  transitionPreviousMeta: document.getElementById("discovery-transition-previous-meta"),
  transitionPreviousFrame: document.getElementById("discovery-transition-previous"),
  transitionNextMeta: document.getElementById("discovery-transition-next-meta"),
  transitionNextFrame: document.getElementById("discovery-transition-next"),
  transitionNextLabel: document.getElementById("discovery-transition-next-label"),
  transitionPreviousPanel: document.getElementById("discovery-transition-previous-panel"),
  transitionNextPanel: document.getElementById("discovery-transition-next-panel"),
  transitionWitnessStrip: document.getElementById("discovery-transition-witness-strip"),
  topLeftTitle: document.getElementById("discovery-top-left-title"),
  middleLeftTitle: document.getElementById("discovery-middle-left-title"),
  bottomLeftTitle: document.getElementById("discovery-bottom-left-title"),
  projectionTitle: document.getElementById("discovery-projection-title"),
  projectionSummary: document.getElementById("discovery-projection-summary"),
  prototypeLegend: document.getElementById("discovery-prototype-legend"),
  projectionTargetWorlds: document.getElementById("discovery-projection-target-worlds"),
  projectionTargetWorldsHead: document.getElementById("discovery-projection-target-worlds-head"),
  projectionTargetWorldsTitle: document.getElementById("discovery-projection-target-worlds-title"),
  projectionTargetWorldBody: document.getElementById("discovery-projection-target-worlds-body"),
  projectionTargetWorldList: document.getElementById("discovery-projection-target-worlds-list"),
  probabilitiesTitle: document.getElementById("discovery-class-probabilities-title"),
  probabilitiesSummary: document.getElementById("discovery-class-probabilities-summary"),
  dashboardComposite: document.querySelector(".dashboard-composite"),
  dashboardLeftStack: document.querySelector(".dashboard-left-stack"),
  dashboardRightStack: document.querySelector(".dashboard-right-stack"),
  topLeftChart: document.getElementById("discovery-top-left-chart"),
  middleLeftChart: document.getElementById("discovery-middle-left-chart"),
  bottomLeftChart: document.getElementById("discovery-bottom-left-chart"),
  projection: document.getElementById("discovery-projection"),
  probabilities: document.getElementById("discovery-class-probabilities"),
  classRows: document.getElementById("discovery-class-rows"),
  evalModal: document.getElementById("discovery-eval-modal"),
  evalModalClose: document.getElementById("discovery-eval-modal-close"),
  evalModalTitle: document.getElementById("discovery-eval-modal-title"),
  evalModalSubtitle: document.getElementById("discovery-eval-modal-subtitle"),
  evalModalSource: document.getElementById("discovery-eval-modal-source"),
  evalDatasetRoot: document.getElementById("discovery-eval-dataset-root"),
  evalDiscoveryJson: document.getElementById("discovery-eval-discovery-json"),
  evalScenarioSplit: document.getElementById("discovery-eval-scenario-split"),
  evalSampleSeed: document.getElementById("discovery-eval-sample-seed"),
  evalWorkers: document.getElementById("discovery-eval-workers"),
  evalModalStatus: document.getElementById("discovery-eval-modal-status"),
  evalStartButton: document.getElementById("discovery-eval-start-button"),
  compareStartButton: document.getElementById("discovery-compare-start-button"),
  runDeleteModal: document.getElementById("discovery-run-delete-modal"),
  runDeleteClose: document.getElementById("discovery-run-delete-close"),
  runDeleteSubtitle: document.getElementById("discovery-run-delete-subtitle"),
  runDeletePhrase: document.getElementById("discovery-run-delete-phrase"),
  runDeleteInput: document.getElementById("discovery-run-delete-input"),
  runDeleteStatus: document.getElementById("discovery-run-delete-status"),
  runDeleteCancel: document.getElementById("discovery-run-delete-cancel"),
  runDeleteConfirm: document.getElementById("discovery-run-delete-confirm"),
  evalDrawer: document.getElementById("discovery-eval-drawer"),
  evalDrawerSubtitle: document.getElementById("discovery-eval-drawer-subtitle"),
  evalRerunButton: document.getElementById("discovery-eval-rerun-button"),
  evalSaveAll: document.getElementById("discovery-eval-save-all"),
  evalCloseDrawer: document.getElementById("discovery-eval-close-drawer"),
  evalSummary: document.getElementById("discovery-eval-summary"),
  evalStatusFilter: document.getElementById("discovery-eval-status-filter"),
  evalClassList: document.getElementById("discovery-eval-class-list"),
  evalClassLoadMore: document.getElementById("discovery-eval-class-load-more"),
  evalDetailMeta: document.getElementById("discovery-eval-detail-meta"),
  evalImageGrid: document.getElementById("discovery-eval-image-grid"),
  evalImagePrevious: document.getElementById("discovery-eval-image-previous"),
  evalImageExpected: document.getElementById("discovery-eval-image-expected"),
  evalImagePredicted: document.getElementById("discovery-eval-image-predicted"),
  evalImageComparison: document.getElementById("discovery-eval-image-comparison"),
  evalImageMetaPrevious: document.getElementById("discovery-eval-image-meta-previous"),
  evalImageMetaExpected: document.getElementById("discovery-eval-image-meta-expected"),
  evalImageMetaPredicted: document.getElementById("discovery-eval-image-meta-predicted"),
  evalImageMetaComparison: document.getElementById("discovery-eval-image-meta-comparison"),
};

elements.topLeftSlot = elements.topLeftTitle?.closest(".dashboard-slot") || null;
elements.middleLeftSlot = elements.middleLeftTitle?.closest(".dashboard-slot") || null;
elements.bottomLeftSlot = elements.bottomLeftTitle?.closest(".dashboard-slot") || null;
elements.projectionSlot = elements.projectionTitle?.closest(".dashboard-projection-slot") || null;
elements.probabilitiesSlot = elements.probabilitiesTitle?.closest(".dashboard-probability-slot") || null;

const chartState = {
  topLeft: null,
  middleLeft: null,
  bottomLeft: null,
  projection: null,
  probabilities: null,
};
const transitionGalleryNormalizationCache = new WeakMap();
const programVersionsNormalizationCache = new WeakMap();
const classRowsRenderSnapshotCache = new WeakMap();
const classRowGroupLookupCache = new WeakMap();
const projectionLegendSummaryCache = new WeakMap();
const PYTHON_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break",
  "case", "class", "continue", "def", "del", "elif", "else", "except",
  "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
  "match", "nonlocal", "not", "or", "pass", "raise", "return", "try",
  "while", "with", "yield",
]);
const PYTHON_BUILTINS = new Set([
  "Exception", "RuntimeError", "TypeError", "ValueError", "all", "any", "bool",
  "cls", "dict", "enumerate", "filter", "float", "getattr", "hasattr", "int",
  "isinstance", "len", "list", "map", "max", "min", "print", "range", "self",
  "set", "setattr", "str", "sum", "tuple", "zip",
]);
const PYTHON_TOKEN_RE = new RegExp(
  `(@[A-Za-z_][A-Za-z0-9_]*|\\b(?:${Array.from(PYTHON_KEYWORDS).join("|")})\\b|\\b(?:${Array.from(PYTHON_BUILTINS).join("|")})\\b|\\b(?:0[xX][0-9a-fA-F_]+|\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b)`,
  "g",
);

function randomIndex(length) {
  return Math.floor(Math.random() * length);
}

function waitMs(delayMs) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, Math.max(0, Number(delayMs) || 0));
  });
}

function chooseNextPageTitleIndex(previousIndex) {
  if (!DISCOVERY_PAGE_TITLES.length) return null;
  if (DISCOVERY_PAGE_TITLES.length === 1) return 0;
  let nextIndex = previousIndex;
  while (nextIndex === previousIndex) {
    nextIndex = randomIndex(DISCOVERY_PAGE_TITLES.length);
  }
  return nextIndex;
}

function chooseRandomPageTitleTransition() {
  if (!PAGE_TITLE_TRANSITIONS.length) return null;
  return PAGE_TITLE_TRANSITIONS[randomIndex(PAGE_TITLE_TRANSITIONS.length)] || null;
}

function setDiscoveryPageTitle(titleText) {
  const activeLayer = elements.pageTitleLayers?.[state.pageTitleActiveLayerIndex] || null;
  if (activeLayer) {
    activeLayer.textContent = titleText;
  }
  document.title = titleText;
}

function waitForAnimation(animation) {
  if (!animation?.finished) return Promise.resolve();
  return animation.finished.catch(() => {});
}

function copyDiscoveryPageTitleVisualStyle(source, target) {
  if (!source || !target) return;
  const computed = window.getComputedStyle(source);
  const styleProps = [
    "color",
    "font",
    "fontFamily",
    "fontSize",
    "fontStyle",
    "fontWeight",
    "letterSpacing",
    "lineHeight",
    "textAlign",
    "textTransform",
    "textShadow",
  ];
  for (const prop of styleProps) {
    target.style[prop] = computed[prop];
  }
}

function createDiscoveryPageTitleOverlay(layer, text) {
  if (!layer) return null;
  const rect = layer.getBoundingClientRect();
  const overlay = document.createElement("span");
  overlay.className = "discovery-page-title-overlay";
  overlay.textContent = text;
  copyDiscoveryPageTitleVisualStyle(layer, overlay);
  overlay.style.left = `${rect.left + window.scrollX}px`;
  overlay.style.top = `${rect.top + window.scrollY}px`;
  overlay.style.minWidth = `${Math.ceil(rect.width)}px`;
  overlay.style.height = `${Math.ceil(rect.height)}px`;
  overlay.style.transform = "translate3d(0, 0, 0)";
  overlay.style.opacity = "1";
  overlay.style.filter = "blur(0px)";
  document.body.append(overlay);
  return overlay;
}

function freezeDiscoveryPageTitleWidth(nextLayer) {
  if (!elements.pageTitle || !nextLayer) return;
  const currentWidth = Math.ceil(elements.pageTitle.getBoundingClientRect().width || 0);
  const nextWidth = Math.ceil(nextLayer.getBoundingClientRect().width || 0);
  const frozenWidth = Math.max(currentWidth, nextWidth);
  if (frozenWidth > 0) {
    elements.pageTitle.style.width = `${frozenWidth}px`;
  }
}

function clearDiscoveryPageTitleWidth() {
  if (elements.pageTitle) {
    elements.pageTitle.style.width = "";
  }
}

async function animateDiscoveryPageTitle(nextTitle) {
  const layers = elements.pageTitleLayers || [];
  if (layers.length < 2) {
    setDiscoveryPageTitle(nextTitle);
    return;
  }
  const currentLayer = layers[state.pageTitleActiveLayerIndex] || null;
  const nextLayerIndex = state.pageTitleActiveLayerIndex === 0 ? 1 : 0;
  const nextLayer = layers[nextLayerIndex] || null;
  if (!currentLayer || !nextLayer) {
    setDiscoveryPageTitle(nextTitle);
    return;
  }
  if (typeof currentLayer.animate !== "function" || typeof nextLayer.animate !== "function") {
    currentLayer.classList.remove("is-active");
    nextLayer.textContent = nextTitle;
    nextLayer.classList.add("is-active");
    state.pageTitleActiveLayerIndex = nextLayerIndex;
    document.title = nextTitle;
    return;
  }
  const transition = chooseRandomPageTitleTransition();
  if (!transition) {
    setDiscoveryPageTitle(nextTitle);
    return;
  }
  nextLayer.textContent = nextTitle;
  nextLayer.classList.add("is-active");
  nextLayer.style.opacity = "0";
  nextLayer.style.transform = "";
  nextLayer.style.filter = "";
  freezeDiscoveryPageTitleWidth(nextLayer);
  elements.pageTitle?.classList.add("is-animating");

  const outgoingOverlay = createDiscoveryPageTitleOverlay(currentLayer, currentLayer.textContent || "");
  const incomingOverlay = createDiscoveryPageTitleOverlay(nextLayer, nextTitle);
  if (!outgoingOverlay || !incomingOverlay) {
    elements.pageTitle?.classList.remove("is-animating");
    currentLayer.classList.remove("is-active");
    state.pageTitleActiveLayerIndex = nextLayerIndex;
    setDiscoveryPageTitle(nextTitle);
    nextLayer.style.opacity = "";
    nextLayer.style.transform = "";
    nextLayer.style.filter = "";
    clearDiscoveryPageTitleWidth();
    return;
  }

  const cleanupOverlays = () => {
    outgoingOverlay.remove();
    incomingOverlay.remove();
    elements.pageTitle?.classList.remove("is-animating");
  };

  const outgoing = outgoingOverlay.animate(transition.out, transition.outTiming);
  const incoming = incomingOverlay.animate(transition.in, transition.inTiming);
  document.title = nextTitle;

  try {
    await Promise.all([waitForAnimation(outgoing), waitForAnimation(incoming)]);
  } finally {
    cleanupOverlays();
    currentLayer.classList.remove("is-active");
    currentLayer.style.opacity = "";
    currentLayer.style.transform = "";
    currentLayer.style.filter = "";
    nextLayer.style.opacity = "";
    nextLayer.style.transform = "";
    nextLayer.style.filter = "";
    state.pageTitleActiveLayerIndex = nextLayerIndex;
    clearDiscoveryPageTitleWidth();
  }
}

async function rotateDiscoveryPageTitle({ animate } = { animate: true }) {
  if (!elements.pageTitle || !DISCOVERY_PAGE_TITLES.length) return;
  if (state.pageTitleAnimating) return;
  const nextIndex = chooseNextPageTitleIndex(state.pageTitleIndex);
  if (nextIndex == null) return;
  const nextTitle = DISCOVERY_PAGE_TITLES[nextIndex];
  state.pageTitleAnimating = true;
  try {
    if (animate && state.pageTitleIndex !== null) {
      await animateDiscoveryPageTitle(nextTitle);
    } else {
      setDiscoveryPageTitle(nextTitle);
    }
    state.pageTitleIndex = nextIndex;
  } finally {
    state.pageTitleAnimating = false;
  }
}

function scheduleDiscoveryPageTitleRotation(delayMs = PAGE_TITLE_ROTATION_INTERVAL_MS) {
  if (state.pageTitleRotationHandle) {
    window.clearTimeout(state.pageTitleRotationHandle);
    state.pageTitleRotationHandle = null;
  }
  state.pageTitleRotationHandle = window.setTimeout(async () => {
    try {
      await rotateDiscoveryPageTitle({ animate: true });
    } finally {
      scheduleDiscoveryPageTitleRotation(PAGE_TITLE_ROTATION_INTERVAL_MS);
    }
  }, delayMs);
}

function initializeDiscoveryPageTitleRotation() {
  if (!elements.pageTitle || !DISCOVERY_PAGE_TITLES.length) return;
  rotateDiscoveryPageTitle({ animate: false }).catch(() => {});
  scheduleDiscoveryPageTitleRotation(PAGE_TITLE_ROTATION_INTERVAL_MS);
}

function triggerDiscoveryPageTitleRotationNow() {
  if (!elements.pageTitle || !DISCOVERY_PAGE_TITLES.length) return;
  if (state.pageTitleAnimating) return;
  if (state.pageTitleRotationHandle) {
    window.clearTimeout(state.pageTitleRotationHandle);
    state.pageTitleRotationHandle = null;
  }
  rotateDiscoveryPageTitle({ animate: true })
    .catch(() => {})
    .finally(() => {
      scheduleDiscoveryPageTitleRotation(PAGE_TITLE_ROTATION_INTERVAL_MS);
    });
}

function echartsApi() {
  return window.echarts || null;
}

function ensureChart(host, key) {
  const api = echartsApi();
  if (!api || !host) return null;
  const existing = chartState[key];
  if (existing && existing.getDom() === host) {
    return existing;
  }
  if (existing) {
    existing.dispose();
  }
  const chart = api.init(host, null, { renderer: "canvas" });
  chartState[key] = chart;
  return chart;
}

function resizeMountedCharts() {
  for (const chart of Object.values(chartState)) {
    if (!chart) continue;
    try {
      chart.resize();
    } catch (_error) {
      // Ignore transient resize failures while the layout is settling.
    }
  }
}

function clearChartHost(host, key, message) {
  const existing = chartState[key];
  if (existing) {
    existing.dispose();
    chartState[key] = null;
  }
  if (host) {
    host.innerHTML = `<div class="chart-empty">${escapeHtml(message)}</div>`;
  }
}

function resolvePanelEnabled(panel, fallback = true) {
  if (panel === false) return false;
  if (typeof panel?.enabled === "boolean") {
    return panel.enabled !== false;
  }
  return Boolean(fallback);
}

function resolveDashboardPanelEnabled(dashboardPayload) {
  const layoutEnabled = dashboardPayload?.dashboardLayout?.panel_enabled || {};
  const dashboardSpec = dashboardPayload?.dashboardSpec || {};
  const display = dashboardPayload?.display || {};
  return {
    top_left: typeof layoutEnabled.top_left === "boolean"
      ? layoutEnabled.top_left
      : resolvePanelEnabled(dashboardSpec.top_left, true),
    bottom_left: typeof layoutEnabled.bottom_left === "boolean"
      ? layoutEnabled.bottom_left
      : resolvePanelEnabled(dashboardSpec.bottom_left, true),
    projection: typeof layoutEnabled.projection === "boolean"
      ? layoutEnabled.projection
      : resolvePanelEnabled(display.projection, true),
    class_probability: typeof layoutEnabled.class_probability === "boolean"
      ? layoutEnabled.class_probability
      : resolvePanelEnabled(display.class_probability, true),
  };
}

function setElementHidden(element, hidden) {
  if (!element) return;
  const shouldHide = Boolean(hidden);
  element.hidden = shouldHide;
  if (shouldHide) {
    element.style.display = "none";
    return;
  }
  element.style.removeProperty("display");
}

function resetDashboardPanelVisibility() {
  setElementHidden(elements.topLeftSlot, false);
  setElementHidden(elements.middleLeftSlot, false);
  setElementHidden(elements.bottomLeftSlot, false);
  setElementHidden(elements.projectionSlot, false);
  setElementHidden(elements.probabilitiesSlot, false);
  setElementHidden(elements.dashboardLeftStack, false);
  setElementHidden(elements.dashboardRightStack, false);
}

function resolveClassRowsEnabled(dashboardPayload) {
  const display = dashboardPayload?.display || {};
  if (typeof display.class_rows === "boolean") {
    return display.class_rows;
  }
  const rows = dashboardPayload?.classRows;
  return Array.isArray(rows) && rows.length > 0;
}

function applyLiveFooterLayout(dashboardPayload) {
  const classRowsEnabled = resolveClassRowsEnabled(dashboardPayload);
  if (elements.classRowsColumn) {
    setElementHidden(elements.classRowsColumn, !classRowsEnabled);
  }
  if (elements.classRowsTitle && shouldRenderPanel("classRowsTitle", classRowsEnabled ? "Dynamics Class Table" : "")) {
    elements.classRowsTitle.textContent = classRowsEnabled ? "Dynamics Class Table" : "";
  }
  if (elements.hudColumn) {
    setElementHidden(elements.hudColumn, false);
  }
  if (elements.liveFooter) {
    elements.liveFooter.classList.toggle("is-single-column", !classRowsEnabled);
  }
}

function syncDashboardPanelVisibility(dashboardPayload) {
  const panelEnabled = resolveDashboardPanelEnabled(dashboardPayload);
  const leftColumnEnabled = panelEnabled.top_left || panelEnabled.bottom_left;
  const rightColumnEnabled = panelEnabled.projection || panelEnabled.class_probability;
  const visibilitySignature = stableSignature({
    ...panelEnabled,
    leftColumnEnabled,
    rightColumnEnabled,
  });

  if (shouldRenderPanel("dashboardVisibility", visibilitySignature)) {
    invalidatePanelCache(
      "layout",
      "topLeftTitle",
      "middleLeftTitle",
      "bottomLeftTitle",
      "topLeftChart",
      "middleLeftChart",
      "bottomLeftChart",
      "projection",
      "probabilities",
      "prototypeLegend",
    );

    if (!panelEnabled.top_left) {
      clearChartHost(elements.topLeftChart, "topLeft", "");
    }
    if (!panelEnabled.bottom_left) {
      clearChartHost(elements.middleLeftChart, "middleLeft", "");
      clearChartHost(elements.bottomLeftChart, "bottomLeft", "");
    }
    if (!panelEnabled.projection) {
      clearChartHost(elements.projection, "projection", "");
      if (elements.projectionSummary) {
        elements.projectionSummary.textContent = "";
      }
      if (elements.prototypeLegend) {
        elements.prototypeLegend.innerHTML = "";
      }
    }
    if (!panelEnabled.class_probability) {
      clearChartHost(elements.probabilities, "probabilities", "");
      if (elements.probabilitiesSummary) {
        elements.probabilitiesSummary.textContent = "";
      }
    }
  }

  setElementHidden(elements.topLeftSlot, !panelEnabled.top_left);
  setElementHidden(elements.middleLeftSlot, !panelEnabled.bottom_left);
  setElementHidden(elements.bottomLeftSlot, !panelEnabled.bottom_left);
  setElementHidden(elements.projectionSlot, !panelEnabled.projection);
  setElementHidden(elements.probabilitiesSlot, !panelEnabled.class_probability);
  setElementHidden(elements.dashboardLeftStack, !leftColumnEnabled);
  setElementHidden(elements.dashboardRightStack, !rightColumnEnabled);

  return {
    ...panelEnabled,
    left_column: leftColumnEnabled,
    right_column: rightColumnEnabled,
  };
}

function activeRun() {
  return state.activeRunPayload;
}

function isPausedView() {
  return state.viewMode === "paused";
}

function liveViewRemainingMs() {
  if (isPausedView()) return 0;
  return Math.max(0, LIVE_VIEW_AUTO_PAUSE_MS - (Date.now() - state.liveViewStartedAtMs));
}

function isProjectionPaused() {
  return state.projectionMode === "paused";
}

function isProjectionSparseFilterEnabled() {
  return Boolean(state.projectionExcludeSparseClasses);
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function optionalInteger(value) {
  const parsed = optionalNumber(value);
  if (parsed === null) return null;
  return Number.isInteger(parsed) ? parsed : Math.trunc(parsed);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pathLeaf(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  const parts = text.split(/[\\/]+/).filter(Boolean);
  return parts[parts.length - 1] || text;
}

function normalizeSourceText(value) {
  if (typeof value !== "string") return null;
  const normalized = value.replace(/\r\n?/g, "\n");
  return normalized.trim() ? normalized : null;
}

function stableSignature(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function dashboardRevisionToken(dashboard, fallback = null) {
  const updatedAt = String(dashboard?.updatedAt || "").trim();
  if (updatedAt) return updatedAt;
  const fallbackText = String(fallback || "").trim();
  return fallbackText || "dashboard-static";
}

function transitionGalleryRevisionToken(runPayload) {
  const revision = String(
    runPayload?.transitionGallery?.revision
    || runPayload?.dashboard?.transitionGallery?.revision
    || ""
  ).trim();
  if (revision) return revision;
  return dashboardRevisionToken(runPayload?.dashboard, runPayload?.runId);
}

function transitionBundleMetaSignature(bundleMeta) {
  const selectedWitnesses = Array.isArray(bundleMeta?.selectedWitnesses)
    ? bundleMeta.selectedWitnesses
    : [];
  return stableSignature({
    action: bundleMeta?.action || null,
    previousTerminated: bundleMeta?.previousTerminated ?? null,
    expectedTerminated: bundleMeta?.expectedTerminated ?? null,
    predictedTerminated: bundleMeta?.predictedTerminated ?? null,
    worldIndex: bundleMeta?.worldIndex ?? null,
    mapName: bundleMeta?.mapName ?? null,
    predictionErrorMessage: bundleMeta?.predictionErrorMessage || null,
    predictionErrorPhase: bundleMeta?.predictionErrorPhase || null,
    predictedParseError: bundleMeta?.predictedParseError || null,
    rejectionReason: bundleMeta?.rejectionReason || null,
    rejectionReasons: Array.isArray(bundleMeta?.rejectionReasons) ? bundleMeta.rejectionReasons : [],
    qualifiedRegressionReject: Boolean(bundleMeta?.qualifiedRegressionReject),
    brokenGroupIds: Array.isArray(bundleMeta?.brokenGroupIds) ? bundleMeta.brokenGroupIds : [],
    selectedRegressionGroupIds: Array.isArray(bundleMeta?.selectedRegressionGroupIds)
      ? bundleMeta.selectedRegressionGroupIds
      : [],
    failGroups: Array.isArray(bundleMeta?.failGroups)
      ? bundleMeta.failGroups.map((group) => ({
        groupId: group.groupId || null,
        commitVersion: group.commitVersion || null,
        classId: group.classId ?? null,
        transitionCount: group.transitionCount ?? 0,
      }))
      : [],
    splitEvents: Array.isArray(bundleMeta?.splitEvents)
      ? bundleMeta.splitEvents.map((event) => ({
        splitGroupId: event.splitGroupId || null,
        sourceClassId: event.sourceClassId ?? null,
        brokenChildGroupId: event.brokenChildGroupId || null,
        brokenClassId: event.brokenClassId ?? null,
        brokenTransitionCount: event.brokenTransitionCount ?? 0,
        keptChildGroupId: event.keptChildGroupId || null,
        keptClassId: event.keptClassId ?? null,
        keptTransitionCount: event.keptTransitionCount ?? 0,
      }))
      : [],
    selectedWitnesses: selectedWitnesses.map((witness) => ({
      caseLabel: witness.caseLabel || null,
      transitionKey: witness.transitionKey || null,
      groupId: witness.groupId || null,
      commitVersion: witness.commitVersion || null,
      caseSource: witness.caseSource || null,
      worldIndex: witness.worldIndex ?? null,
      mapName: witness.mapName || null,
      previous: artifactKey(witness.previous),
      expected: artifactKey(witness.expected),
      predicted: artifactKey(witness.predicted),
      bundle: artifactKey(witness.bundle),
    })),
  });
}

function ensureRunPanelCache(runKey) {
  const normalizedRunKey = String(runKey || "none");
  if (state.renderCache.runKey === normalizedRunKey) {
    return;
  }
  state.renderCache = {
    runKey: normalizedRunKey,
    panels: {},
  };
}

function invalidatePanelCache(...panelKeys) {
  for (const panelKey of panelKeys) {
    delete state.renderCache.panels[panelKey];
  }
}

function shouldRenderPanel(panelKey, signature) {
  const normalizedSignature = stableSignature(signature);
  if (state.renderCache.panels[panelKey] === normalizedSignature) {
    return false;
  }
  state.renderCache.panels[panelKey] = normalizedSignature;
  return true;
}

function normalizeCommitVersion(value) {
  const text = String(value || "").trim().toLowerCase();
  if (!text) return null;
  const match = text.match(/v\d+/);
  return match ? match[0] : null;
}

function commitVersionSortIndex(value) {
  const normalizedCommitVersion = normalizeCommitVersion(value);
  if (!normalizedCommitVersion) return Number.MAX_SAFE_INTEGER;
  const parsed = Number.parseInt(normalizedCommitVersion.slice(1), 10);
  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
}

function prototypeLegendRowCommitVersion(row) {
  if (!row || typeof row !== "object") return null;
  return normalizeCommitVersion(
    row.commit_version
    || row.group_id
    || row.version_id
    || row.display_id
  );
}

function comparePrototypeLegendRows(left, right) {
  const versionDiff = (
    commitVersionSortIndex(prototypeLegendRowCommitVersion(left))
    - commitVersionSortIndex(prototypeLegendRowCommitVersion(right))
  );
  if (versionDiff) {
    return versionDiff;
  }
  const classDiff = Number(left?.classIndex || 0) - Number(right?.classIndex || 0);
  if (classDiff) {
    return classDiff;
  }
  return String(left?.group_id || left?.version_id || left?.display_id || "").localeCompare(
    String(right?.group_id || right?.version_id || right?.display_id || "")
  );
}

function parseSerializedStatePayload(value) {
  if (!value) return null;
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch (_error) {
      return null;
    }
  }
  return value && typeof value === "object" ? value : null;
}

function normalizeTargetWorlds(runPayload) {
  const rawTargetWorlds = runPayload?.dashboard?.agent?.target_worlds;
  if (!Array.isArray(rawTargetWorlds)) {
    return [];
  }
  return rawTargetWorlds
    .map((rawWorld) => {
      const worldIndex = optionalInteger(rawWorld?.world_index);
      if (worldIndex === null || worldIndex <= 0) {
        return null;
      }
      const worldSeed = optionalInteger(rawWorld?.world_seed);
      const roundRobinOrder = optionalInteger(rawWorld?.round_robin_order);
      const mapName = normalizeTransitionMapName(
        rawWorld?.map_name
        ?? rawWorld?.scenario_type
        ?? rawWorld?.world_label
      );
      const worldLabel = String(
        rawWorld?.world_label
        || mapName
        || `world ${worldIndex}`
      ).trim() || `world ${worldIndex}`;
      return {
        worldIndex,
        worldSeed,
        roundRobinOrder: (
          roundRobinOrder !== null && roundRobinOrder > 0
            ? roundRobinOrder
            : null
        ),
        mapName,
        worldLabel,
        scenarioType: normalizeTransitionMapName(rawWorld?.scenario_type),
        transitionCount: Math.max(0, optionalInteger(rawWorld?.transition_count) ?? 0),
        isActive: Boolean(rawWorld?.is_active),
        isCurrent: Boolean(rawWorld?.is_current),
        resumePending: Boolean(rawWorld?.resume_pending),
        previewStateJson: (
          typeof rawWorld?.preview_state_json === "string"
            ? String(rawWorld.preview_state_json)
            : null
        ),
      };
    })
    .filter(Boolean)
    .sort((left, right) => (
      Number(left.roundRobinOrder ?? Number.POSITIVE_INFINITY)
        - Number(right.roundRobinOrder ?? Number.POSITIVE_INFINITY)
      || Number(left.worldIndex) - Number(right.worldIndex)
      || String(left.worldLabel).localeCompare(String(right.worldLabel))
    ));
}

function targetWorldSelectionRunKey(runPayload) {
  return String(runPayload?.runId || activeRun()?.runId || "none");
}

function projectionTargetWorldPanelExpandedRunKey(runPayload) {
  return projectionSelectionRunKey(runPayload);
}

function isProjectionTargetWorldPanelExpanded(runPayload) {
  const runKey = projectionTargetWorldPanelExpandedRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.projectionTargetWorldPanelExpandedByRun, runKey)) {
    return false;
  }
  return Boolean(state.projectionTargetWorldPanelExpandedByRun[runKey]);
}

function setProjectionTargetWorldPanelExpanded(runPayload, expanded) {
  const runKey = projectionTargetWorldPanelExpandedRunKey(runPayload);
  state.projectionTargetWorldPanelExpandedByRun[runKey] = Boolean(expanded);
}

function toggleProjectionTargetWorldPanel(runPayload) {
  setProjectionTargetWorldPanelExpanded(runPayload, !isProjectionTargetWorldPanelExpanded(runPayload));
  invalidatePanelCache("projectionTargetWorlds");
  renderActive();
}

function targetWorldPanelExpandedRunKey(runPayload) {
  return targetWorldSelectionRunKey(runPayload);
}

function isTargetWorldPanelExpanded(runPayload) {
  const runKey = targetWorldPanelExpandedRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.targetWorldPanelExpandedByRun, runKey)) {
    return false;
  }
  return Boolean(state.targetWorldPanelExpandedByRun[runKey]);
}

function setTargetWorldPanelExpanded(runPayload, expanded) {
  const runKey = targetWorldPanelExpandedRunKey(runPayload);
  state.targetWorldPanelExpandedByRun[runKey] = Boolean(expanded);
}

function toggleTargetWorldPanel(runPayload) {
  setTargetWorldPanelExpanded(runPayload, !isTargetWorldPanelExpanded(runPayload));
  invalidatePanelCache("targetWorlds");
  renderActive();
}

function selectedTargetWorldIndex(runPayload) {
  const runKey = targetWorldSelectionRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.targetWorldSelectionByRun, runKey)) {
    return null;
  }
  const normalized = optionalInteger(state.targetWorldSelectionByRun[runKey]);
  return normalized !== null && normalized > 0 ? normalized : null;
}

function resolveSelectedTargetWorld(runPayload, targetWorlds) {
  if (!Array.isArray(targetWorlds) || !targetWorlds.length) {
    return null;
  }
  const selectedWorldIndex = selectedTargetWorldIndex(runPayload);
  if (selectedWorldIndex !== null) {
    const selectedEntry = targetWorlds.find((entry) => entry.worldIndex === selectedWorldIndex) || null;
    if (selectedEntry) {
      return selectedEntry;
    }
  }
  const liveCurrentWorldIndex = resolveLiveCurrentTargetWorldIndex(runPayload, targetWorlds);
  return (
    targetWorlds.find((entry) => entry.worldIndex === liveCurrentWorldIndex)
    || targetWorlds.find((entry) => entry.isCurrent)
    || targetWorlds[0]
    || null
  );
}

function selectTargetWorld(runPayload, worldIndex) {
  const normalizedWorldIndex = optionalInteger(worldIndex);
  if (normalizedWorldIndex === null || normalizedWorldIndex <= 0) {
    return;
  }
  const runKey = targetWorldSelectionRunKey(runPayload);
  if (selectedTargetWorldIndex(runPayload) === normalizedWorldIndex) {
    return;
  }
  state.targetWorldSelectionByRun[runKey] = normalizedWorldIndex;
  invalidatePanelCache("targetWorlds");
  renderActive();
}

function resolveLiveCurrentTargetWorldIndex(runPayload, targetWorlds) {
  const metricWorldIndex = normalizeTransitionWorldIndex(
    runPayload?.dashboard?.metrics?.world_index
    ?? runPayload?.dashboard?.metrics?.worldIndex,
  );
  if (
    metricWorldIndex !== null
    && Array.isArray(targetWorlds)
    && targetWorlds.some((entry) => entry.worldIndex === metricWorldIndex)
  ) {
    return metricWorldIndex;
  }
  const currentEntry = Array.isArray(targetWorlds)
    ? (targetWorlds.find((entry) => entry.isCurrent) || null)
    : null;
  return currentEntry?.worldIndex ?? null;
}

function normalizeVersionKey(value, fallbackCommitVersion = null) {
  return normalizeCommitVersion(value || fallbackCommitVersion);
}

function normalizeGroupVersionKey(value, fallbackCommitVersion = null) {
  const text = String(value || "").trim().toLowerCase();
  if (!text) {
    const commitVersion = normalizeCommitVersion(fallbackCommitVersion);
    return commitVersion ? `${commitVersion}:g0000` : null;
  }
  if (!text.includes(":")) {
    const commitVersion = normalizeCommitVersion(text || fallbackCommitVersion);
    return commitVersion ? `${commitVersion}:g0000` : null;
  }
  const [versionPart, groupPartRaw] = text.split(":", 2);
  const commitVersion = normalizeCommitVersion(versionPart || fallbackCommitVersion);
  const groupPart = String(groupPartRaw || "").trim().toLowerCase();
  if (!commitVersion) return null;
  if (!/^g\d+$/.test(groupPart)) return `${commitVersion}:g0000`;
  return `${commitVersion}:${groupPart}`;
}

function versionLabel(value) {
  return String(value || "-").trim().toUpperCase() || "-";
}

function versionDisplayParts(value) {
  const normalized = normalizeVersionKey(value);
  if (!normalized) {
    return {
      versionText: "V---",
      groupText: "",
    };
  }
  return {
    versionText: normalized.toUpperCase(),
    groupText: "",
  };
}

function groupVersionDisplayParts(value, fallbackCommitVersion = null) {
  const normalized = normalizeGroupVersionKey(value, fallbackCommitVersion);
  if (!normalized) {
    const fallbackVersion = normalizeVersionKey(fallbackCommitVersion);
    return {
      versionText: fallbackVersion ? fallbackVersion.toUpperCase() : "V---",
      groupText: "",
    };
  }
  const [versionPart, groupPart] = normalized.toUpperCase().split(":", 2);
  return {
    versionText: versionPart || "V---",
    groupText: groupPart || "G0000",
  };
}

function formatTerminatedLabel(value) {
  if (typeof value !== "boolean") return "-";
  return value ? "True" : "False";
}

function formatActionLabel(value) {
  const text = String(value || "").trim();
  return text || "-";
}

function normalizeTransitionWorldIndex(value) {
  const parsed = optionalInteger(value);
  return parsed !== null && parsed > 0 ? parsed : null;
}

function normalizeTransitionMapName(value) {
  return formatMapDisplayName(value);
}

function normalizeTransitionLocation(raw) {
  if (!raw || typeof raw !== "object") {
    return {
      worldIndex: null,
      mapName: null,
    };
  }
  return {
    worldIndex: normalizeTransitionWorldIndex(raw.worldIndex ?? raw.world_index),
    mapName: normalizeTransitionMapName(raw.mapName ?? raw.map_name),
  };
}

function formatTransitionLocationText(raw) {
  const { worldIndex, mapName } = normalizeTransitionLocation(raw);
  const parts = [];
  if (worldIndex !== null) {
    parts.push(`W${worldIndex}`);
  }
  if (mapName) {
    parts.push(mapName);
  }
  return parts.length ? parts.join(" · ") : null;
}

function appendTransitionLocationChips(items, raw) {
  if (!Array.isArray(items)) return;
  const { worldIndex, mapName } = normalizeTransitionLocation(raw);
  if (worldIndex !== null) {
    items.push({ label: "World", value: String(worldIndex) });
  }
  if (mapName) {
    items.push({ label: "Map", value: mapName });
  }
}

function formatFailureLabel(fail, prefix = "FAIL") {
  const failureId = String(fail?.failureId || "").trim();
  const folderMatch = failureId.match(/^fail(\d+)$/i);
  if (folderMatch) {
    return `${prefix} ${folderMatch[1]}`;
  }
  const failureIndex = optionalInteger(fail?.failureIndex);
  if (failureIndex !== null && failureIndex > 0) {
    return `${prefix} ${failureIndex}`;
  }
  return prefix;
}

function transitionVariantTone(variant) {
  if (variant === "success") return "success";
  if (variant === "fail") return "fail";
  return "expected";
}

function chooseTransitionMetadataBundle(review, selection, selectedFail) {
  if (selection?.activeVariant === "fail") {
    return selectedFail?.bundle || (review?.fails?.[0]?.bundle ?? null);
  }
  if (selection?.activeVariant === "success") {
    return review?.explained?.bundle || (review?.fails?.[0]?.bundle ?? null);
  }
  return review?.explained?.bundle || (review?.fails?.[0]?.bundle ?? null);
}

function artifactKey(artifact) {
  if (!artifact || typeof artifact !== "object") return null;
  const url = String(artifact.url || "").trim();
  return url || null;
}

function normalizeArtifact(artifact) {
  if (!artifact || typeof artifact !== "object") return null;
  const url = String(artifact.url || "").trim();
  if (!url) return null;
  const path = String(artifact.path || "").trim();
  return {
    url,
    path,
  };
}

function normalizeGeneratorAttempt(generatorAttempt) {
  if (!generatorAttempt || typeof generatorAttempt !== "object") return null;
  const generatorTry = optionalInteger(
    generatorAttempt.generatorTry ?? generatorAttempt.generator_try,
  );
  return {
    generatorTry: generatorTry !== null && generatorTry > 0 ? generatorTry : 1,
    totalAttempts: optionalInteger(
      generatorAttempt.totalAttempts ?? generatorAttempt.total_attempts,
    ) ?? 0,
    status: String(generatorAttempt.status || "").trim() || null,
    prompt: normalizeArtifact(generatorAttempt.prompt),
    output: normalizeArtifact(generatorAttempt.output),
    reasoning: normalizeArtifact(generatorAttempt.reasoning),
    error: normalizeArtifact(generatorAttempt.error),
  };
}

function normalizeFailAttempt(attempt) {
  if (!attempt || typeof attempt !== "object") return null;
  const unexpectedAttempt = optionalInteger(
    attempt.unexpectedAttempt ?? attempt.unexpected_attempt,
  );
  const generatorAttempts = Array.isArray(attempt.generatorAttempts)
    ? attempt.generatorAttempts
      .map((generatorAttempt) => normalizeGeneratorAttempt(generatorAttempt))
      .filter(Boolean)
    : [];
  generatorAttempts.sort((left, right) => (
    (left.generatorTry - right.generatorTry)
    || (left.totalAttempts - right.totalAttempts)
  ));
  return {
    unexpectedAttempt: unexpectedAttempt !== null && unexpectedAttempt > 0 ? unexpectedAttempt : 1,
    currentVersion: normalizeCommitVersion(attempt.currentVersion ?? attempt.current_version),
    status: String(attempt.status || "").trim() || null,
    errorMessage: String(attempt.errorMessage || attempt.error_message || "").trim() || null,
    elapsedText: String(attempt.elapsedText || attempt.elapsed_text || "").trim() || null,
    patchDigest: String(attempt.patchDigest || attempt.patch_digest || "").trim() || null,
    accepted: Boolean(attempt.accepted),
    summary: normalizeArtifact(attempt.summary),
    appliedProgram: normalizeArtifact(attempt.appliedProgram ?? attempt.applied_program),
    generatorAttempts,
  };
}

function normalizeBoolean(value) {
  return typeof value === "boolean" ? value : null;
}

function readBundleStepTerminated(payload, key) {
  if (!payload || typeof payload !== "object") return null;
  const statePayload = payload[key];
  if (!statePayload || typeof statePayload !== "object") return null;
  const stepPayload = statePayload.step;
  if (!stepPayload || typeof stepPayload !== "object") return null;
  return normalizeBoolean(stepPayload.terminated);
}

function encodeArtifactRelativePath(relativePath) {
  return String(relativePath || "")
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function buildArtifactUrlFromBundleUrl(bundleUrl, relativePath) {
  const baseUrl = String(bundleUrl || "").trim();
  const normalizedPath = encodeArtifactRelativePath(relativePath);
  if (!baseUrl || !normalizedPath) return null;
  const marker = "/artifacts/";
  const markerIndex = baseUrl.indexOf(marker);
  if (markerIndex < 0) return null;
  return `${baseUrl.slice(0, markerIndex)}${marker}${normalizedPath}`;
}

function artifactFromBundleRelativePath(bundleUrl, relativePath) {
  const normalizedPath = String(relativePath || "").trim().replace(/\\/g, "/");
  if (!normalizedPath) return null;
  const url = buildArtifactUrlFromBundleUrl(bundleUrl, normalizedPath);
  if (!url) return null;
  return {
    url,
    path: normalizedPath,
  };
}

function normalizeBundleStringList(values) {
  if (!Array.isArray(values)) return [];
  return values
    .map((value) => String(value || "").trim())
    .filter(Boolean);
}

function normalizeBundleFailGroup(rawGroup) {
  if (!rawGroup || typeof rawGroup !== "object") return null;
  return {
    groupId: String(rawGroup.group_id || "").trim() || null,
    commitVersion: normalizeCommitVersion(rawGroup.commit_version),
    classId: optionalInteger(rawGroup.class_id),
    transitionCount: optionalInteger(rawGroup.transition_count) ?? 0,
  };
}

function normalizeBundleSplitEvent(rawEvent) {
  if (!rawEvent || typeof rawEvent !== "object") return null;
  return {
    splitGroupId: String(rawEvent.split_group_id || "").trim() || null,
    sourceClassId: optionalInteger(rawEvent.source_class_id),
    brokenChildGroupId: String(rawEvent.broken_child_group_id || "").trim() || null,
    brokenClassId: optionalInteger(rawEvent.broken_class_id),
    brokenTransitionCount: optionalInteger(rawEvent.broken_transition_count) ?? 0,
    keptChildGroupId: String(rawEvent.kept_child_group_id || "").trim() || null,
    keptClassId: optionalInteger(rawEvent.kept_class_id),
    keptTransitionCount: optionalInteger(rawEvent.kept_transition_count) ?? 0,
  };
}

function normalizeBundleWitnessArtifact(rawWitness, bundleUrl) {
  if (!rawWitness || typeof rawWitness !== "object") return null;
  return {
    caseLabel: String(rawWitness.case_label || "").trim() || "Regression Witness",
    transitionKey: String(rawWitness.transition_key || "").trim() || null,
    groupId: String(rawWitness.group_id || "").trim() || null,
    commitVersion: normalizeCommitVersion(rawWitness.commit_version),
    caseSource: String(rawWitness.case_source || "").trim() || null,
    worldIndex: normalizeTransitionWorldIndex(rawWitness.world_index),
    mapName: normalizeTransitionMapName(rawWitness.map_name),
    previous: artifactFromBundleRelativePath(bundleUrl, rawWitness.previous_state_png),
    expected: artifactFromBundleRelativePath(bundleUrl, rawWitness.actual_next_state_png),
    predicted: artifactFromBundleRelativePath(bundleUrl, rawWitness.predicted_next_state_png),
    bundle: artifactFromBundleRelativePath(bundleUrl, rawWitness.transition_bundle_json),
  };
}

function extractTransitionBundleMetadata(payload, bundleUrl = "") {
  if (!payload || typeof payload !== "object") return null;
  const predictionError = payload.prediction_error;
  const rejection = payload.rejection && typeof payload.rejection === "object"
    ? payload.rejection
    : null;
  return {
    action: String(payload.action || "").trim() || null,
    synthesisAction: String(payload.action || "").trim() || null,
    previousStatePayload: payload.previous_state && typeof payload.previous_state === "object"
      ? payload.previous_state
      : null,
    expectedStatePayload: payload.actual_next_state && typeof payload.actual_next_state === "object"
      ? payload.actual_next_state
      : null,
    predictedStatePayload: payload.predicted_next_state && typeof payload.predicted_next_state === "object"
      ? payload.predicted_next_state
      : null,
    previousTerminated: readBundleStepTerminated(payload, "previous_state"),
    expectedTerminated: readBundleStepTerminated(payload, "actual_next_state"),
    predictedTerminated: readBundleStepTerminated(payload, "predicted_next_state"),
    worldIndex: normalizeTransitionWorldIndex(payload.world_index),
    mapName: normalizeTransitionMapName(payload.map_name),
    predictionErrorMessage: predictionError && typeof predictionError === "object"
      ? String(predictionError.message || "").trim() || null
      : null,
    predictionErrorPhase: predictionError && typeof predictionError === "object"
      ? String(predictionError.phase || "").trim() || null
      : null,
    predictedParseError: String(payload.predicted_parse_error || "").trim() || null,
    rejectionReason: rejection ? String(rejection.reason || "").trim() || null : null,
    rejectionReasons: normalizeBundleStringList(rejection?.rejection_reasons),
    qualifiedRegressionReject: Boolean(rejection?.qualified_regression_reject),
    brokenGroupIds: normalizeBundleStringList(rejection?.broken_group_ids),
    selectedRegressionGroupIds: normalizeBundleStringList(rejection?.selected_regression_group_ids),
    failGroups: Array.isArray(rejection?.fail_groups)
      ? rejection.fail_groups.map((group) => normalizeBundleFailGroup(group)).filter(Boolean)
      : [],
    splitEvents: Array.isArray(rejection?.split_events)
      ? rejection.split_events.map((event) => normalizeBundleSplitEvent(event)).filter(Boolean)
      : [],
    selectedWitnesses: Array.isArray(rejection?.selected_regression_witness_artifacts)
      ? rejection.selected_regression_witness_artifacts
        .map((witness) => normalizeBundleWitnessArtifact(witness, bundleUrl))
        .filter(Boolean)
      : [],
  };
}

function normalizeReviewFailRef(failRef) {
  if (!failRef || typeof failRef !== "object") return null;
  const reviewId = String(failRef.reviewId || "").trim();
  const failureId = String(failRef.failureId || "").trim();
  if (!reviewId) return null;
  return {
    reviewId,
    failureId: failureId || null,
    failureIndex: optionalInteger(failRef.failureIndex) ?? 0,
    stepIndex: optionalInteger(failRef.stepIndex) ?? 0,
    label: String(failRef.label || "").trim() || reviewId.toUpperCase(),
  };
}

function normalizeTransitionGallery(runPayload) {
  const topLevelGallery = runPayload?.transitionGallery;
  const dashboardGallery = runPayload?.dashboard?.transitionGallery;
  const gallery = (
    topLevelGallery
    && typeof topLevelGallery === "object"
    && (
      (Array.isArray(topLevelGallery.reviews) && topLevelGallery.reviews.length > 0)
      || (Array.isArray(topLevelGallery.versions) && topLevelGallery.versions.length > 0)
      || (topLevelGallery.live && typeof topLevelGallery.live === "object")
    )
  )
    ? topLevelGallery
    : (dashboardGallery || topLevelGallery || {});
  if (gallery && typeof gallery === "object") {
    const cached = transitionGalleryNormalizationCache.get(gallery);
    if (cached) {
      return cached;
    }
  }
  const reviews = Array.isArray(gallery.reviews)
    ? gallery.reviews
      .map((review) => {
        if (!review || typeof review !== "object") return null;
        const reviewId = String(review.reviewId || "").trim();
        if (!reviewId) return null;
        const fails = Array.isArray(review.fails)
          ? review.fails
            .map((fail) => {
              if (!fail || typeof fail !== "object") return null;
              const failureId = String(fail.failureId || "").trim();
              if (!failureId) return null;
              return {
                failureId,
                failureIndex: optionalInteger(fail.failureIndex) ?? 0,
                stepIndex: optionalInteger(fail.stepIndex) ?? optionalInteger(review.stepIndex) ?? 0,
                versionTag: normalizeCommitVersion(fail.versionTag),
                attempts: Array.isArray(fail.attempts)
                  ? fail.attempts
                    .map((attempt) => normalizeFailAttempt(attempt))
                    .filter(Boolean)
                  : [],
                worldIndex: normalizeTransitionWorldIndex(fail.worldIndex ?? fail.world_index),
                mapName: normalizeTransitionMapName(fail.mapName ?? fail.map_name),
                image: normalizeArtifact(fail.image),
                bundle: normalizeArtifact(fail.bundle),
                terminated: normalizeBoolean(fail.terminated),
              };
            })
            .filter(Boolean)
          : [];
        fails.sort((left, right) => (
          (left.failureIndex - right.failureIndex)
          || String(left.failureId).localeCompare(String(right.failureId))
        ));
        return {
          reviewId,
          stepId: String(review.stepId || reviewId).trim() || reviewId,
          stepIndex: optionalInteger(review.stepIndex) ?? 0,
          action: String(review.action || "").trim() || null,
          worldIndex: normalizeTransitionWorldIndex(review.worldIndex ?? review.world_index),
          mapName: normalizeTransitionMapName(review.mapName ?? review.map_name),
          previousTerminated: normalizeBoolean(review.previousTerminated),
          expectedTerminated: normalizeBoolean(review.expectedTerminated),
          previous: normalizeArtifact(review.previous),
          expected: normalizeArtifact(review.expected),
          explained: review.explained && typeof review.explained === "object"
            ? {
              versionTag: normalizeCommitVersion(review.explained.versionTag),
              worldIndex: normalizeTransitionWorldIndex(review.explained.worldIndex ?? review.explained.world_index),
              mapName: normalizeTransitionMapName(review.explained.mapName ?? review.explained.map_name),
              image: normalizeArtifact(review.explained.image),
              bundle: normalizeArtifact(review.explained.bundle),
              stepIndex: optionalInteger(review.explained.stepIndex) ?? optionalInteger(review.stepIndex) ?? 0,
              failureIndex: optionalInteger(review.explained.failureIndex) ?? 0,
              terminated: normalizeBoolean(review.explained.terminated),
            }
            : null,
          fails,
        };
      })
      .filter(Boolean)
    : [];
  reviews.sort((left, right) => left.stepIndex - right.stepIndex);
  const reviewsById = new Map(reviews.map((review) => [review.reviewId, review]));

  const versions = Array.isArray(gallery.versions)
    ? gallery.versions
      .map((version) => {
        if (!version || typeof version !== "object") return null;
        const commitVersion = normalizeCommitVersion(version.commitVersion || version.versionKey || version.label);
        const versionKey = normalizeVersionKey(version.versionKey || version.label, commitVersion);
        if (!versionKey || !commitVersion) return null;
        return {
          versionKey,
          label: versionLabel(version.label || commitVersion),
          commitVersion,
          sortIndex: optionalInteger(version.sortIndex) ?? 10 ** 6,
          sourcePath: String(version.sourcePath || "").trim() || null,
          source: normalizeSourceText(version.source),
          sourceDigest: String(version.sourceDigest || "").trim() || null,
          sourceLineCount: optionalInteger(version.sourceLineCount) ?? 0,
          isActive: Boolean(version.isActive),
          isCurrentProgram: Boolean(version.isCurrentProgram),
          isCurrentExplainer: Boolean(version.isCurrentExplainer),
          isLatestActive: Boolean(version.isLatestActive),
          isExplainer: Boolean(version.isExplainer),
          introReviewId: String(version.introReviewId || "").trim() || null,
        };
      })
      .filter(Boolean)
    : [];
  versions.sort((left, right) => (
    (left.sortIndex - right.sortIndex)
    || String(left.versionKey).localeCompare(String(right.versionKey))
  ));
  const versionsByKey = new Map(versions.map((version) => [version.versionKey, version]));

  const live = gallery.live && typeof gallery.live === "object"
    ? {
      currentVersionKey: normalizeVersionKey(gallery.live.currentVersionKey),
      currentGroupKey: normalizeGroupVersionKey(
        gallery.live.currentGroupKey,
        gallery.live.currentVersionKey,
      ),
      currentGroupLabel: String(gallery.live.currentGroupLabel || "").trim() || null,
      currentClassIndex: optionalInteger(gallery.live.currentClassIndex),
      isFail: Boolean(gallery.live.isFail),
      defaultReview: normalizeReviewFailRef(gallery.live.defaultReview),
    }
    : {
      currentVersionKey: null,
      currentGroupKey: null,
      currentGroupLabel: null,
      currentClassIndex: null,
      isFail: false,
      defaultReview: null,
    };

  const normalized = {
    versions,
    versionsByKey,
    reviews,
    reviewsById,
    live,
  };
  if (gallery && typeof gallery === "object") {
    transitionGalleryNormalizationCache.set(gallery, normalized);
  }
  return normalized;
}

function normalizeProgramVersions(runPayload) {
  const topLevelVersions = runPayload?.programVersions;
  const dashboardVersions = runPayload?.dashboard?.programVersions;
  const versionsInput = Array.isArray(topLevelVersions) && topLevelVersions.length > 0
    ? topLevelVersions
    : dashboardVersions;
  if (Array.isArray(versionsInput)) {
    const cached = programVersionsNormalizationCache.get(versionsInput);
    if (cached) {
      return cached;
    }
  }
  const versions = Array.isArray(versionsInput)
    ? versionsInput
      .map((version) => {
        if (!version || typeof version !== "object") return null;
        const commitVersion = normalizeCommitVersion(version.commitVersion || version.versionKey || version.label);
        const versionKey = normalizeVersionKey(version.versionKey || version.label, commitVersion);
        if (!versionKey || !commitVersion) return null;
        return {
          versionKey,
          label: versionLabel(version.label || commitVersion),
          commitVersion,
          sortIndex: optionalInteger(version.sortIndex) ?? 10 ** 6,
          sourcePath: String(version.sourcePath || "").trim() || null,
          source: normalizeSourceText(version.source),
          sourceDigest: String(version.sourceDigest || "").trim() || null,
          sourceLineCount: optionalInteger(version.sourceLineCount) ?? 0,
          isActive: Boolean(version.isActive),
          isCurrentProgram: Boolean(version.isCurrentProgram),
          isCurrentExplainer: Boolean(version.isCurrentExplainer),
          isLatestActive: Boolean(version.isLatestActive),
          isExplainer: Boolean(version.isExplainer),
          introReviewId: String(version.introReviewId || "").trim() || null,
          failReviews: Array.isArray(version.failReviews)
            ? version.failReviews
              .map((fail) => normalizeReviewFailRef(fail))
              .filter(Boolean)
            : [],
          defaultReview: normalizeReviewFailRef(version.defaultReview),
        };
      })
      .filter(Boolean)
    : [];
  versions.sort((left, right) => (
    (left.sortIndex - right.sortIndex)
    || String(left.versionKey).localeCompare(String(right.versionKey))
  ));
  const versionsByKey = new Map(versions.map((version) => [version.versionKey, version]));
  const normalized = {
    versions,
    versionsByKey,
  };
  if (Array.isArray(versionsInput)) {
    programVersionsNormalizationCache.set(versionsInput, normalized);
  }
  return normalized;
}

function resolveLiveNavigatorIdentity(runPayload, gallery) {
  const dashboard = runPayload?.dashboard || {};
  const heatmap = dashboard.visitationHeatmap || {};
  const metrics = dashboard.metrics || {};
  const currentClassIndex = optionalInteger(gallery?.live?.currentClassIndex)
    ?? optionalInteger(heatmap.current_group_class_index)
    ?? optionalInteger(heatmap.current_class_index)
    ?? optionalInteger(heatmap.current_transition?.class_index)
    ?? optionalInteger(metrics.current_dynamics_class);
  const groupSource = gallery?.live?.currentGroupKey
    || gallery?.live?.currentGroupLabel
    || heatmap.current_group_id
    || heatmap.current_group_label
    || heatmap.current_class_label
    || null;
  const groupParts = groupVersionDisplayParts(groupSource, gallery?.live?.currentVersionKey);
  const rawGroupLabel = String(
    gallery?.live?.currentGroupLabel
    || heatmap.current_group_label
    || heatmap.current_group_id
    || heatmap.current_class_label
    || "",
  ).trim();
  const classPrefix = currentClassIndex && currentClassIndex > 0 ? `C${currentClassIndex}: ` : "";
  if (groupParts.groupText) {
    return `${classPrefix}${groupParts.versionText} ${groupParts.groupText}`.trim();
  }
  if (groupParts.versionText && groupParts.versionText !== "V---") {
    return `${classPrefix}${groupParts.versionText}`.trim();
  }
  if (rawGroupLabel) {
    return `${classPrefix}${rawGroupLabel}`.trim();
  }
  return classPrefix ? `${classPrefix}-` : "-";
}

function formatOfflineEvalStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "runtime_error") return "Runtime Error";
  if (normalized === "compile_error") return "Compile Error";
  if (normalized === "mismatch") return "Mismatch";
  if (normalized === "pass") return "Pass";
  if (normalized === "compile_failed") return "Compile Failed";
  if (normalized === "running") return "Running";
  if (normalized === "queued") return "Queued";
  if (normalized === "completed") return "Completed";
  if (normalized === "failed") return "Failed";
  return normalized ? normalized.toUpperCase() : "-";
}

function formatOfflineEvalResultStatus(status, runSummary = null) {
  if (!isOfflineEvalClassPurity(runSummary)) {
    return formatOfflineEvalStatus(status);
  }
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "pass") return "Pure";
  if (normalized === "mismatch") return "Split";
  if (normalized === "runtime_error") return "Unassigned";
  return formatOfflineEvalStatus(status);
}

function offlineEvalSelectionMode(payload) {
  const explicitMode = String(payload?.selectionMode || "").trim();
  if (explicitMode) return explicitMode;
  return String(payload?.discoveryJson || "").trim() ? "heuristic_class" : "full_dataset";
}

function normalizeOfflineEvalRunMode(value) {
  const normalized = String(value || "accuracy").trim().toLowerCase().replaceAll("-", "_");
  if (["class", "classes", "class_purity", "dynamics_class", "dynamics_classes"].includes(normalized)) {
    return "class_purity";
  }
  return "accuracy";
}

function isOfflineEvalClassPurity(payload) {
  return normalizeOfflineEvalRunMode(payload?.runMode) === "class_purity"
    || offlineEvalSelectionMode(payload?.dataset || payload) === "class_purity";
}

function offlineEvalUsesHeuristicDiscovery(payload) {
  const selectionMode = offlineEvalSelectionMode(payload);
  return selectionMode === "heuristic_class" || selectionMode === "class_purity";
}

function offlineEvalResultLabelSingular(payload) {
  return offlineEvalUsesHeuristicDiscovery(payload) ? "class" : "transition";
}

function offlineEvalResultLabelPlural(payload) {
  return offlineEvalUsesHeuristicDiscovery(payload) ? "classes" : "transitions";
}

function offlineEvalResultPrefix(payload) {
  return offlineEvalUsesHeuristicDiscovery(payload) ? "C" : "T";
}

function offlineEvalRowPrefix(payload) {
  return isOfflineEvalClassPurity(payload) ? "H" : offlineEvalResultPrefix(payload);
}

function offlineEvalProgressLabelPlural(payload) {
  return offlineEvalSelectionMode(payload) === "class_purity"
    ? "transitions"
    : offlineEvalResultLabelPlural(payload);
}

function normalizeOfflineEvalProgress(summary) {
  const completedCount = Math.max(0, Number(summary?.progress?.completedCount || 0));
  const totalCount = Math.max(0, Number(summary?.progress?.totalCount || 0));
  const status = String(summary?.status || "").trim().toLowerCase();
  let percent = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;
  if (totalCount <= 0 && status && status !== "queued" && status !== "running") {
    percent = 100;
  }
  percent = Math.max(0, Math.min(100, percent));
  return {
    completedCount,
    totalCount,
    percent,
    percentText: `${Math.round(percent)}%`,
    countText: totalCount > 0 ? `${completedCount}/${totalCount} ${offlineEvalProgressLabelPlural(summary?.dataset)}` : null,
  };
}

function offlineEvalProgressSnapshotKey(summary) {
  const progress = normalizeOfflineEvalProgress(summary);
  const status = String(summary?.status || "").trim().toLowerCase();
  return `${status}:${progress.completedCount}:${progress.totalCount}`;
}

function renderOfflineEvalDrawerSubtitleMarkup(runSummary) {
  const statusKey = String(runSummary?.status || "").trim().toLowerCase();
  const statusText = formatOfflineEvalStatus(runSummary?.status);
  const sourceLabel = String(runSummary?.program?.label || "").trim();
  const sourcePath = String(runSummary?.program?.sourcePath || "").trim();
  const sourceText = pathLeaf(sourcePath) || sourceLabel || "selected source";
  const datasetRoot = String(runSummary?.dataset?.datasetRoot || "").trim();
  const datasetText = pathLeaf(datasetRoot);
  const scenarioSplit = normalizeOfflineEvalScenarioSplit(runSummary?.dataset?.scenarioSplit || "all");
  const scenarioSplitText = String(
    runSummary?.dataset?.scenarioSplitLabel
    || offlineEvalScenarioSplitLabel(scenarioSplit)
    || scenarioSplit,
  ).trim();
  const classCount = Number(runSummary?.dataset?.classCount ?? runSummary?.progress?.totalCount ?? 0);
  const workerCount = Number(runSummary?.options?.workers ?? 0);
  const resultLabelPlural = offlineEvalResultLabelPlural(runSummary?.dataset);
  const infoItems = [
    {
      label: "Source",
      value: sourceText,
    },
    {
      label: "Status",
      value: statusText,
      tone: statusKey ? `is-${statusKey}` : "",
    },
    datasetText
      ? {
        label: "Dataset",
        value: datasetText,
      }
      : null,
    scenarioSplitText
      ? {
        label: "Split",
        value: scenarioSplitText,
      }
      : null,
    Number.isFinite(classCount) && classCount > 0
      ? {
        label: "Scope",
        value: `${classCount} ${resultLabelPlural}`,
      }
      : null,
    Number.isFinite(workerCount) && workerCount > 0
      ? {
        label: "Workers",
        value: `${workerCount}`,
      }
      : null,
  ].filter(Boolean);
  return `
    <div class="discovery-eval-drawer-subtitle-grid">
      ${infoItems.map((item) => `
        <div class="discovery-eval-drawer-meta-card ${escapeHtml(item.tone || "")}">
          <span class="discovery-eval-drawer-meta-label">${escapeHtml(item.label)}</span>
          <span class="discovery-eval-drawer-meta-value">${escapeHtml(item.value)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderOfflineEvalSummaryMarkup(runSummary, notice = "", exportStatus = null) {
  const exportStatusKey = String(exportStatus?.status || "").trim().toLowerCase();
  const showExportProgress = Boolean(exportStatus && exportStatusKey && exportStatusKey !== "idle");
  const classPurityMode = isOfflineEvalClassPurity(runSummary);
  const accuracy = Number(
    classPurityMode
      ? (runSummary?.metrics?.purityHToR ?? runSummary?.metrics?.accuracy ?? 0)
      : (runSummary?.metrics?.accuracy ?? 0),
  );
  const completedCount = showExportProgress
    ? Number(exportStatus?.completedCount || 0)
    : Number(runSummary?.progress?.completedCount || 0);
  const totalCount = showExportProgress
    ? Number(exportStatus?.totalCount || 0)
    : Number(runSummary?.progress?.totalCount || 0);
  const percent = totalCount > 0 ? Math.max(0, Math.min(100, (completedCount / totalCount) * 100)) : 0;
  const summaryNotice = String(
    (showExportProgress ? exportStatus?.message : "")
    || notice
    || runSummary?.message
    || "",
  ).trim();
  const resultLabelPlural = showExportProgress
    ? "failure sets"
    : offlineEvalProgressLabelPlural(runSummary?.dataset);
  const progressLabel = showExportProgress
    ? "Export Progress"
    : classPurityMode
      ? "Class Progress"
      : "Eval Progress";
  const metricCards = classPurityMode
    ? [
      {
        label: "H->C Purity",
        value: `${(accuracy * 100).toFixed(2)}%`,
        tone: "is-accent",
      },
      {
        label: "Assigned",
        value: `${((Number(runSummary?.metrics?.assignmentCoverage ?? 0)) * 100).toFixed(1)}%`,
      },
      {
        label: "Split Classes",
        value: `${runSummary?.metrics?.splitHeuristicClassCount ?? runSummary?.metrics?.mismatchCount ?? 0}`,
        tone: Number(runSummary?.metrics?.splitHeuristicClassCount ?? runSummary?.metrics?.mismatchCount ?? 0) > 0 ? "is-failed" : "",
      },
    ]
    : [
      {
        label: "Accuracy",
        value: `${(accuracy * 100).toFixed(2)}%`,
        tone: "is-accent",
      },
      {
        label: "Correct",
        value: `${runSummary?.metrics?.correctCount ?? 0}`,
      },
      {
        label: "Failed",
        value: `${runSummary?.metrics?.failedCount ?? 0}`,
        tone: Number(runSummary?.metrics?.failedCount ?? 0) > 0 ? "is-failed" : "",
      },
    ];
  return `
    <div class="discovery-eval-summary-body">
      <div class="discovery-eval-summary-metrics">
        ${metricCards.map((card) => `
          <div class="discovery-eval-summary-card ${escapeHtml(card.tone || "")}">
            <div class="discovery-eval-summary-card-label">${escapeHtml(card.label)}</div>
            <div class="discovery-eval-summary-card-value">${escapeHtml(card.value)}</div>
          </div>
        `).join("")}
      </div>
      <div class="discovery-eval-summary-progress-block">
        <div class="discovery-eval-summary-progress-meta">
          <span class="discovery-eval-summary-progress-count">${escapeHtml(`${progressLabel} · ${completedCount}/${totalCount} ${resultLabelPlural}`)}</span>
          ${summaryNotice ? `
            <span class="discovery-eval-summary-progress-note" title="${escapeHtml(summaryNotice)}">
              ${escapeHtml(summaryNotice)}
            </span>
          ` : ""}
        </div>
        <div class="discovery-eval-progress-bar">
          <div class="discovery-eval-progress-fill" style="width: ${percent.toFixed(2)}%"></div>
        </div>
      </div>
    </div>
  `;
}

function renderOfflineEvalFailureMetaMarkup(failureDetail, runSummary = null) {
  const scenarioText = formatMapDisplayName(
    failureDetail?.scenarioType,
    failureDetail?.artifactStem,
  );
  const artifactText = String(failureDetail?.artifactStem || "").trim();
  const errorMessage = String(failureDetail?.error?.message || "").trim();
  const errorPhase = String(failureDetail?.error?.phase || "").trim();
  const comparisonLines = Array.isArray(failureDetail?.imageMeta?.comparison?.lines)
    ? failureDetail.imageMeta.comparison.lines
      .map((line) => String(line || "").trim())
      .filter(Boolean)
    : [];
  const diffLine = comparisonLines.find((line) => line.startsWith("Diff:")) || "";
  const diffValue = diffLine.startsWith("Diff:")
    ? diffLine.slice("Diff:".length).trim() || diffLine
    : diffLine;
  const usesHeuristicDiscovery = offlineEvalUsesHeuristicDiscovery(runSummary?.dataset || state.offlineEval.runSummary?.dataset);
  const primaryFields = [
    {
      label: usesHeuristicDiscovery ? "Class" : "Case",
      value: `${failureDetail?.classIdx ?? "-"}`,
    },
    {
      label: "Scenario",
      value: scenarioText || artifactText || "-",
    },
    {
      label: "Transition",
      value: failureDetail?.transitionIndex !== null && failureDetail?.transitionIndex !== undefined
        ? `#${failureDetail.transitionIndex}`
        : "-",
    },
  ];
  const secondaryFields = [];
  if (artifactText && artifactText !== scenarioText) {
    secondaryFields.push({
      label: "Artifact",
      value: artifactText,
    });
  }
  const detailFields = [...primaryFields, ...secondaryFields];
  return `
    <div class="discovery-eval-detail-grid">
      ${detailFields.map((field) => `
        <div class="discovery-eval-detail-card ${escapeHtml(field.tone || "")}">
          <span class="discovery-eval-detail-card-label">${escapeHtml(field.label)}</span>
          <span class="discovery-eval-detail-card-value">${escapeHtml(field.value)}</span>
        </div>
      `).join("")}
    </div>
    ${diffValue ? `
      <div class="discovery-eval-detail-diff" title="${escapeHtml(diffLine)}">
        <span class="discovery-eval-detail-diff-label">Diff</span>
        <span class="discovery-eval-detail-diff-value">${escapeHtml(diffValue)}</span>
      </div>
    ` : ""}
    ${errorMessage ? `
      <div class="discovery-eval-detail-error" title="${escapeHtml(`${errorPhase || "error"}: ${errorMessage}`)}">
        <span class="discovery-eval-detail-error-label">Error</span>
        <span class="discovery-eval-detail-error-value">${escapeHtml(`${errorPhase || "error"}: ${errorMessage}`)}</span>
      </div>
    ` : ""}
  `;
}

function offlineEvalClassResultRowKey(row) {
  return [
    String(row?.classIdx ?? ""),
    String(row?.heuristicClassIdx ?? ""),
    String(row?.status ?? ""),
    String(row?.artifactStem ?? ""),
    String(row?.transitionIndex ?? ""),
  ].join(":");
}

function resolveSelectedOfflineEvalClassRow() {
  const selectedKey = String(state.offlineEval.selectedClassRowKey || "").trim();
  if (!selectedKey) return null;
  return resolveOfflineEvalVisibleRows().find(
    (row) => offlineEvalClassResultRowKey(row) === selectedKey,
  ) || state.offlineEval.classResults.find(
    (row) => offlineEvalClassResultRowKey(row) === selectedKey,
  ) || null;
}

function resolveOfflineEvalClassRepresentatives(row) {
  return Array.isArray(row?.representativeTransitions)
    ? row.representativeTransitions.filter((item) => (
      item
      && typeof item === "object"
      && String(item.representativeId || "").trim()
    ))
    : [];
}

function defaultOfflineEvalClassRepresentativeId(row) {
  const representatives = resolveOfflineEvalClassRepresentatives(row);
  return String(representatives[0]?.representativeId || "").trim() || null;
}

function resolveOfflineEvalClassRepresentative(row, representativeId = null) {
  const representatives = resolveOfflineEvalClassRepresentatives(row);
  const safeRepresentativeId = String(representativeId || state.offlineEval.selectedClassRepresentativeId || "").trim();
  if (!safeRepresentativeId) return representatives[0] || null;
  return representatives.find(
    (item) => String(item?.representativeId || "").trim() === safeRepresentativeId,
  ) || representatives[0] || null;
}

function formatOfflineEvalClassRepresentativeRole(representative) {
  const role = String(representative?.role || "").trim().toLowerCase();
  if (role === "majority") return "Majority";
  if (role === "minority") return "Split Sample";
  if (role === "unassigned") return "Unassigned";
  if (role === "unexplainable") return "Unexplainable";
  return "Representative";
}

function renderOfflineEvalClassRepresentativesMarkup(row, selectedRepresentativeId = null) {
  const representatives = resolveOfflineEvalClassRepresentatives(row);
  if (!representatives.length) {
    return `
      <div class="discovery-eval-representative-empty">
        Representative transitions were not recorded for this CLASS run. Re-run CLASS analysis to inspect images.
      </div>
    `;
  }
  const safeSelectedId = String(selectedRepresentativeId || representatives[0]?.representativeId || "").trim();
  return `
    <div class="discovery-eval-representative-block">
      <div class="discovery-eval-representative-head">
        <span class="discovery-eval-representative-title">Representative Transitions</span>
        ${row?.hasMoreRepresentativeTransitions ? `
          <span class="discovery-eval-representative-note">Top ${representatives.length} shown</span>
        ` : ""}
      </div>
      <div class="discovery-eval-representative-list">
        ${representatives.map((representative) => {
          const representativeId = String(representative?.representativeId || "").trim();
          const repairClassLabel = representative?.repairClassId !== null && representative?.repairClassId !== undefined
            ? `C${representative.repairClassId}`
            : (representative?.explainable ? "Unassigned" : "Unexplainable");
          const transitionLabel = representative?.transitionIndex !== null && representative?.transitionIndex !== undefined
            ? `#${representative.transitionIndex}`
            : "-";
          const selectedClass = representativeId && representativeId === safeSelectedId ? "is-selected" : "";
          return `
            <button
              type="button"
              class="discovery-eval-representative-button ${selectedClass}"
              data-discovery-action="select-offline-eval-class-representative"
              data-representative-id="${escapeHtml(representativeId)}"
            >
              <span class="discovery-eval-representative-main">
                <span>${escapeHtml(formatOfflineEvalClassRepresentativeRole(representative))}</span>
                <strong>${escapeHtml(repairClassLabel)}</strong>
              </span>
              <span class="discovery-eval-representative-meta">${escapeHtml(`${transitionLabel} · ${String(representative?.action || "-").toUpperCase()}`)}</span>
            </button>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function renderOfflineEvalClassPurityDetailMarkup(row, selectedRepresentativeId = null) {
  const scenarioText = formatMapDisplayName(row?.scenarioType, row?.artifactStem);
  const artifactText = String(row?.artifactStem || "").trim();
  const repairDistribution = Array.isArray(row?.repairClassDistribution)
    ? row.repairClassDistribution
      .map((item) => {
        const repairClassId = item?.repairClassId;
        const transitionCount = Number(item?.transitionCount || 0);
        return repairClassId !== null && repairClassId !== undefined
          ? `C${repairClassId}: ${transitionCount}`
          : "";
      })
      .filter(Boolean)
    : [];
  const totalCount = Number(row?.transitionCount || 0);
  const explainableCount = Number(row?.explainableTransitionCount || 0);
  const assignedCount = Number(row?.assignedTransitionCount || 0);
  const unexplainableCount = Number(row?.unexplainableTransitionCount || 0);
  const unassignedCount = Number(row?.unassignedTransitionCount || 0);
  const majorityRepairClass = row?.majorityRepairClassId !== null && row?.majorityRepairClassId !== undefined
    ? `C${row.majorityRepairClassId}`
    : "-";
  const detailFields = [
    {
      label: "Heuristic Class",
      value: `${row?.heuristicClassIdx ?? row?.classIdx ?? "-"}`,
    },
    {
      label: "Status",
      value: formatOfflineEvalResultStatus(row?.status, state.offlineEval.runSummary),
      tone: `is-${String(row?.status || "unknown")}`,
    },
    {
      label: "Scenario",
      value: scenarioText || artifactText || "-",
    },
    {
      label: "Sample Transition",
      value: row?.transitionIndex !== null && row?.transitionIndex !== undefined
        ? `#${row.transitionIndex}`
        : "-",
    },
    {
      label: "Purity",
      value: row?.purity !== undefined ? formatPercent(row.purity) : "-",
      tone: Number(row?.purity ?? 0) >= 1 ? "is-status" : "",
    },
    {
      label: "Majority Class",
      value: `${majorityRepairClass} (${Number(row?.majorityRepairTransitionCount || 0)}/${assignedCount})`,
    },
    {
      label: "Assigned",
      value: `${assignedCount}/${totalCount}`,
    },
    {
      label: "Explainable",
      value: `${explainableCount}/${totalCount}`,
    },
    {
      label: "Unexplainable",
      value: `${unexplainableCount}`,
      tone: unexplainableCount > 0 ? "is-failed" : "",
    },
    {
      label: "Unassigned",
      value: `${unassignedCount}`,
      tone: unassignedCount > 0 ? "is-failed" : "",
    },
  ];
  return `
    <div class="discovery-eval-detail-grid">
      ${detailFields.map((field) => `
        <div class="discovery-eval-detail-card ${escapeHtml(field.tone || "")}">
          <span class="discovery-eval-detail-card-label">${escapeHtml(field.label)}</span>
          <span class="discovery-eval-detail-card-value">${escapeHtml(field.value)}</span>
        </div>
      `).join("")}
    </div>
    <div class="discovery-eval-detail-diff" title="${escapeHtml(repairDistribution.join(" · ") || "No dynamics class assignment")}">
      <span class="discovery-eval-detail-diff-label">Dynamics Class Distribution</span>
      <span class="discovery-eval-detail-diff-value">${escapeHtml(repairDistribution.join(" · ") || "No assigned dynamics class")}</span>
    </div>
    ${renderOfflineEvalClassRepresentativesMarkup(row, selectedRepresentativeId)}
  `;
}

function updateOfflineEvalRunList(runs) {
  const normalizedRuns = Array.isArray(runs)
    ? runs.filter((row) => row && typeof row === "object" && String(row.runId || "").trim())
    : [];
  normalizedRuns.sort(
    (left, right) => String(right?.startedAt || "").localeCompare(String(left?.startedAt || "")),
  );
  state.offlineEval.runs = normalizedRuns;
  if (state.activeRunPayload) {
    renderActive();
  }
  return normalizedRuns;
}

function upsertOfflineEvalRunSummary(summary) {
  const runId = String(summary?.runId || "").trim();
  if (!runId) return;
  const nextRuns = state.offlineEval.runs.filter(
    (row) => String(row?.runId || "").trim() !== runId,
  );
  nextRuns.push(summary);
  updateOfflineEvalRunList(nextRuns);
}

function offlineEvalRunMatchesSource(summary, source) {
  if (!summary || !source) return false;
  const runVersionKey = normalizeVersionKey(summary?.program?.versionKey);
  const sourceVersionKey = normalizeVersionKey(source?.versionKey);
  if (runVersionKey && sourceVersionKey && runVersionKey !== sourceVersionKey) {
    return false;
  }
  const sourceRunOutputDir = String(source?.runOutputDir || "").trim();
  const runOutputDir = String(summary?.program?.runOutputDir || "").trim();
  if (sourceRunOutputDir) {
    if (!runOutputDir || runOutputDir !== sourceRunOutputDir) {
      return false;
    }
  }
  const runDigest = String(summary?.program?.sourceDigest || "").trim();
  const sourceDigest = String(source?.sourceDigest || "").trim();
  if (runDigest && sourceDigest) {
    return runDigest === sourceDigest;
  }
  const runPath = String(summary?.program?.sourcePath || "").trim();
  const sourcePath = String(source?.sourcePath || "").trim();
  if (runPath && sourcePath) {
    return runPath === sourcePath;
  }
  return Boolean(runVersionKey && sourceVersionKey && runVersionKey === sourceVersionKey);
}

function offlineEvalRunModeMatches(summary, runMode = "accuracy") {
  return normalizeOfflineEvalRunMode(summary?.runMode) === normalizeOfflineEvalRunMode(runMode);
}

function findOfflineEvalLatestRunForSource(source, runMode = "accuracy") {
  if (!source) return null;
  const currentSummary = state.offlineEval.runSummary;
  if (offlineEvalRunModeMatches(currentSummary, runMode) && offlineEvalRunMatchesSource(currentSummary, source)) {
    return currentSummary;
  }
  return state.offlineEval.runs.find((row) => (
    offlineEvalRunModeMatches(row, runMode)
    && offlineEvalRunMatchesSource(row, source)
  )) || null;
}

function latestOfflineEvalRunSummary() {
  if (Array.isArray(state.offlineEval.runs) && state.offlineEval.runs.length > 0) {
    return state.offlineEval.runs[0] || null;
  }
  return state.offlineEval.runSummary || null;
}

function resolveOfflineEvalSourceForRunSummary(runSummary, runPayload = null) {
  const versionKey = String(runSummary?.program?.versionKey || "").trim();
  if (!versionKey) return null;
  return resolveOfflineEvalSource(runPayload || state.activeRunPayload, versionKey);
}

function resolveOfflineEvalVersionBadge(source, runMode = "accuracy") {
  const normalizedRunMode = normalizeOfflineEvalRunMode(runMode);
  const idleCaption = normalizedRunMode === "class_purity" ? "CLASS" : "RUN";
  const idleTitle = normalizedRunMode === "class_purity"
    ? "Run dynamics class analysis"
    : "Run offline heuristic eval";
  const summary = findOfflineEvalLatestRunForSource(
    typeof source === "string" ? { versionKey: source } : source,
    normalizedRunMode,
  );
  const latestSummary = latestOfflineEvalRunSummary();
  if (
    !summary
    || !latestSummary
    || String(summary?.runId || "").trim() !== String(latestSummary?.runId || "").trim()
  ) {
    return {
      tone: "idle",
      title: idleTitle,
      caption: idleCaption,
      progressPercent: 0,
    };
  }
  const runLabel = normalizedRunMode === "class_purity"
    ? "Dynamics class analysis"
    : "Offline heuristic eval";
  const status = String(summary.status || "").trim().toLowerCase();
  const progress = normalizeOfflineEvalProgress(summary);
  const progressTitle = [progress.countText, progress.percentText].filter(Boolean).join(" · ");
  if (status === "queued" || status === "running") {
    return {
      tone: "running",
      title: progressTitle
        ? `${runLabel} ${formatOfflineEvalStatus(status)} · ${progressTitle}`
        : `${runLabel} ${formatOfflineEvalStatus(status)}`,
      caption: progress.percentText,
      progressPercent: progress.percent,
    };
  }
  if (status === "completed") {
    const classPurity = isOfflineEvalClassPurity(summary);
    const completedCaption = classPurity && summary?.metrics?.purityHToR !== undefined
      ? `${Math.round(Number(summary.metrics.purityHToR || 0) * 100)}%`
      : progress.percentText;
    return {
      tone: "pass",
      title: progressTitle
        ? `${runLabel} completed · ${progressTitle}`
        : `${runLabel} completed`,
      caption: completedCaption,
      progressPercent: progress.percent,
    };
  }
  return {
    tone: "failed",
    title: status === "completed"
      ? (progressTitle
        ? `${runLabel} results · ${progressTitle}`
        : `${runLabel} results`)
      : (progressTitle
        ? `${runLabel} ${formatOfflineEvalStatus(status)} · ${progressTitle}`
        : `${runLabel} ${formatOfflineEvalStatus(status)}`),
    caption: progress.percentText,
    progressPercent: progress.percent,
  };
}

function isOfflineEvalRunActive(summary) {
  const status = String(summary?.status || "").trim().toLowerCase();
  return status === "queued" || status === "running";
}

async function loadOfflineEvalRuns(force = false) {
  if (!force && state.offlineEval.runs.length) {
    return state.offlineEval.runs;
  }
  if (!force && state.offlineEval.runsPromise) {
    return state.offlineEval.runsPromise;
  }
  const request = fetchJson("/api/eval/runs")
    .then((payload) => {
      state.offlineEval.runsPromise = null;
      return updateOfflineEvalRunList(payload?.runs);
    })
    .catch((error) => {
      state.offlineEval.runsPromise = null;
      throw error;
    });
  state.offlineEval.runsPromise = request;
  return request;
}

function stopOfflineEvalPoll() {
  if (state.offlineEval.pollHandle) {
    window.clearTimeout(state.offlineEval.pollHandle);
    state.offlineEval.pollHandle = null;
  }
}

function resetOfflineEvalClassResults() {
  state.offlineEval.classResults = [];
  state.offlineEval.classPageOffset = 0;
  state.offlineEval.classPageHasMore = false;
  state.offlineEval.classPageTotalCount = 0;
  state.offlineEval.classPageLoading = false;
  state.offlineEval.classPageHydrated = false;
  state.offlineEval.classPageUnavailable = false;
  state.offlineEval.classPageRequestToken += 1;
}

function clearOfflineEvalActiveState({ keepRuns = true } = {}) {
  stopOfflineEvalPoll();
  state.offlineEval.drawerOpen = false;
  state.offlineEval.runId = null;
  state.offlineEval.runSummary = null;
  resetOfflineEvalClassResults();
  state.offlineEval.selectedFailureId = null;
  state.offlineEval.selectedClassRowKey = null;
  state.offlineEval.selectedClassRepresentativeId = null;
  state.offlineEval.failureDetail = null;
  state.offlineEval.loadingFailureId = null;
  state.offlineEval.classRepresentativeDetail = null;
  state.offlineEval.loadingClassRepresentativeId = null;
  state.offlineEval.notice = "";
  state.offlineEval.exportBusy = false;
  state.offlineEval.exportStatus = null;
  state.offlineEval.exportToken += 1;
  state.offlineEval.pollInFlight = false;
  state.offlineEval.activeImageKind = "expected";
  if (!keepRuns) {
    state.offlineEval.runs = [];
  }
  clearOfflineEvalImages();
}

function scheduleOfflineEvalPoll(delayMs = OFFLINE_EVAL_SUMMARY_POLL_INTERVAL_MS) {
  stopOfflineEvalPoll();
  if (!state.offlineEval.runId) return;
  state.offlineEval.pollHandle = window.setTimeout(() => {
    refreshOfflineEvalRun().catch((error) => {
      const message = formatError(error);
      state.offlineEval.notice = message;
      renderOfflineEvalDrawer();
    });
  }, Math.max(
    OFFLINE_EVAL_SUMMARY_POLL_MIN_INTERVAL_MS,
    Number(delayMs) || OFFLINE_EVAL_SUMMARY_POLL_INTERVAL_MS,
  ));
}

function resolveOfflineEvalSource(runPayload, preferredVersionKey = null) {
  if (!runPayload) return null;
  const programVersions = normalizeProgramVersions(runPayload);
  if (!programVersions.versions.length) return null;
  const transitionResolved = resolveTransitionSelection(runPayload);
  const { selection, gallery } = transitionResolved;
  let selectedVersion = null;
  const normalizedPreferredKey = normalizeVersionKey(preferredVersionKey);
  if (normalizedPreferredKey) {
    selectedVersion = programVersions.versionsByKey.get(normalizedPreferredKey) || null;
  }
  if (!selectedVersion && selection.mode === "version") {
    selectedVersion = programVersions.versionsByKey.get(normalizeVersionKey(selection.versionKey)) || null;
  }
  if (!selectedVersion) {
    const currentVersionKey = resolveCurrentAcceptedVersionKey(
      runPayload,
      gallery,
      selection.versionKey,
    );
    selectedVersion = programVersions.versionsByKey.get(currentVersionKey)
      || programVersions.versions.find((version) => version.isCurrentExplainer)
      || programVersions.versions.find((version) => version.isCurrentProgram)
      || programVersions.versions[0]
      || null;
  }
  const source = normalizeSourceText(selectedVersion?.source);
  if (!selectedVersion || !source) {
    return null;
  }
  return {
    versionKey: selectedVersion.versionKey,
    label: selectedVersion.label || versionLabel(selectedVersion.versionKey),
    source,
    sourcePath: selectedVersion.sourcePath || null,
    sourceDigest: selectedVersion.sourceDigest || null,
    sourceLineCount: selectedVersion.sourceLineCount || sourceLineCount(source),
    runOutputDir: String(runPayload?.runOutputDir || "").trim() || null,
  };
}

async function ensureOfflineEvalDefaults(force = false) {
  if (!force && state.offlineEval.defaults) {
    return state.offlineEval.defaults;
  }
  if (!force && state.offlineEval.defaultsPromise) {
    return state.offlineEval.defaultsPromise;
  }
  const request = fetchJson("/api/eval/defaults")
    .then((payload) => {
      state.offlineEval.defaults = payload;
      state.offlineEval.defaultsPromise = null;
      return payload;
    })
    .catch((error) => {
      state.offlineEval.defaultsPromise = null;
      throw error;
    });
  state.offlineEval.defaultsPromise = request;
  return request;
}

function resolveOfflineEvalCatalog() {
  const catalog = state.offlineEval.defaults?.catalog;
  return Array.isArray(catalog?.datasetRoots) ? catalog : { datasetRoots: [] };
}

function normalizeOfflineEvalScenarioSplit(value) {
  const text = String(value || "").trim().toLowerCase().replace(/-/g, "_");
  if (!text || text === "none") return "all";
  if (text === "train") return "all_train";
  if (text === "test") return "all_test";
  return text;
}

function currentOfflineEvalScenarioSplit() {
  return normalizeOfflineEvalScenarioSplit(
    elements.evalScenarioSplit?.value
    || state.offlineEval.defaults?.scenarioSplit
    || state.offlineEval.defaults?.catalog?.defaultScenarioSplit
    || "all",
  );
}

function offlineEvalScenarioSplitOptions() {
  const catalog = resolveOfflineEvalCatalog();
  const options = Array.isArray(catalog?.scenarioSplits) ? catalog.scenarioSplits : [];
  return options.length
    ? options
    : [{ value: "all", label: "All maps", scenarioCount: null }];
}

function offlineEvalScenarioSplitLabel(value) {
  const normalized = normalizeOfflineEvalScenarioSplit(value);
  const option = offlineEvalScenarioSplitOptions().find(
    (row) => normalizeOfflineEvalScenarioSplit(row?.value) === normalized,
  );
  return String(option?.label || normalized.replace(/_/g, " ")).trim();
}

function offlineEvalScenarioSplitCount(value) {
  const normalized = normalizeOfflineEvalScenarioSplit(value);
  const option = offlineEvalScenarioSplitOptions().find(
    (row) => normalizeOfflineEvalScenarioSplit(row?.value) === normalized,
  );
  const count = Number(option?.scenarioCount);
  return Number.isFinite(count) ? count : null;
}

function offlineEvalCountForScenarioSplit(row, countField, countBySplitField, scenarioSplit) {
  const normalized = normalizeOfflineEvalScenarioSplit(scenarioSplit);
  const countsBySplit = row?.[countBySplitField];
  const splitValue = countsBySplit && typeof countsBySplit === "object"
    ? Number(countsBySplit[normalized])
    : Number.NaN;
  if (Number.isFinite(splitValue)) {
    return splitValue;
  }
  const fallback = Number(row?.[countField]);
  return Number.isFinite(fallback) ? fallback : null;
}

function populateOfflineEvalScenarioSplitOptions(defaults) {
  if (!elements.evalScenarioSplit) return;
  const options = offlineEvalScenarioSplitOptions();
  elements.evalScenarioSplit.innerHTML = "";
  for (const row of options) {
    const option = document.createElement("option");
    option.value = normalizeOfflineEvalScenarioSplit(row?.value);
    const count = Number(row?.scenarioCount);
    const countText = Number.isFinite(count) ? ` (${count} maps)` : "";
    option.textContent = `${row?.label || option.value}${countText}`;
    elements.evalScenarioSplit.append(option);
  }
  const preferred = normalizeOfflineEvalScenarioSplit(
    defaults?.scenarioSplit
    || defaults?.catalog?.defaultScenarioSplit
    || options[0]?.value
    || "all",
  );
  const available = new Set(options.map((row) => normalizeOfflineEvalScenarioSplit(row?.value)));
  elements.evalScenarioSplit.value = available.has(preferred)
    ? preferred
    : normalizeOfflineEvalScenarioSplit(options[0]?.value || "all");
}

function syncOfflineEvalDiscoveryJsonOptions(
  datasetRootPath,
  preferredDiscoveryJsonPath = null,
) {
  if (!elements.evalDiscoveryJson) return;
  const datasetRoots = resolveOfflineEvalCatalog().datasetRoots || [];
  const selectedDataset = datasetRoots.find(
    (row) => String(row?.path || "").trim() === String(datasetRootPath || "").trim(),
  ) || null;
  const discoveryJsons = Array.isArray(selectedDataset?.discoveryJsons)
    ? selectedDataset.discoveryJsons
    : [];
  const currentValue = String(
    preferredDiscoveryJsonPath !== null && preferredDiscoveryJsonPath !== undefined
      ? preferredDiscoveryJsonPath
      : (elements.evalDiscoveryJson.value || ""),
  ).trim();
  elements.evalDiscoveryJson.innerHTML = "";
  const scenarioSplit = currentOfflineEvalScenarioSplit();
  const fullDatasetOption = document.createElement("option");
  fullDatasetOption.value = "";
  const fullDatasetCount = offlineEvalCountForScenarioSplit(
    selectedDataset,
    "fullDatasetCount",
    "fullDatasetCountBySplit",
    scenarioSplit,
  );
  const fullDatasetCountText = Number.isFinite(Number(fullDatasetCount))
    ? ` (${fullDatasetCount} transitions)`
    : "";
  fullDatasetOption.textContent = `None (evaluate full dataset)${fullDatasetCountText}`;
  fullDatasetOption.title = "Evaluate every transition in the selected dataset root.";
  elements.evalDiscoveryJson.append(fullDatasetOption);
  for (const row of discoveryJsons) {
    const option = document.createElement("option");
    option.value = String(row.path || "");
    const classCount = offlineEvalCountForScenarioSplit(
      row,
      "classCount",
      "classCountBySplit",
      scenarioSplit,
    );
    const classCountText = Number.isFinite(Number(classCount))
      ? ` (${classCount} classes)`
      : "";
    option.textContent = `${row.label || row.displayPath || row.path}${classCountText}`;
    option.title = row.displayPath || row.path || "";
    elements.evalDiscoveryJson.append(option);
  }
  const nextValue = (
    currentValue === ""
    || discoveryJsons.some((row) => String(row?.path || "").trim() === currentValue)
  )
    ? currentValue
    : String(discoveryJsons[0]?.path || "").trim();
  elements.evalDiscoveryJson.value = nextValue;
}

function populateOfflineEvalCatalog(defaults) {
  if (!elements.evalDatasetRoot) return;
  const catalog = defaults?.catalog;
  const datasetRoots = Array.isArray(catalog?.datasetRoots) ? catalog.datasetRoots : [];
  elements.evalDatasetRoot.innerHTML = "";
  for (const row of datasetRoots) {
    const option = document.createElement("option");
    option.value = String(row.path || "");
    option.textContent = row.label || row.displayPath || row.path || "";
    option.title = row.displayPath || row.path || "";
    elements.evalDatasetRoot.append(option);
  }
  const preferredDatasetRoot = String(
    defaults?.datasetRoot
    || catalog?.defaultDatasetRoot
    || datasetRoots[0]?.path
    || "",
  ).trim();
  elements.evalDatasetRoot.value = preferredDatasetRoot;
  populateOfflineEvalScenarioSplitOptions(defaults);
  syncOfflineEvalDiscoveryJsonOptions(
    preferredDatasetRoot,
    String(defaults?.discoveryJson || catalog?.defaultDiscoveryJson || "").trim(),
  );
}

function updateOfflineEvalModalStatus() {
  const defaults = state.offlineEval.defaults;
  const datasetRoot = String(elements.evalDatasetRoot?.value || "").trim();
  const discoveryJson = String(elements.evalDiscoveryJson?.value || "").trim();
  const scenarioSplit = currentOfflineEvalScenarioSplit();
  const workerCount = Math.max(1, optionalInteger(elements.evalWorkers?.value) ?? defaults?.workers ?? 16);
  const datasetRoots = resolveOfflineEvalCatalog().datasetRoots || [];
  const selectedDataset = datasetRoots.find(
    (row) => String(row?.path || "").trim() === datasetRoot,
  ) || null;
  const selectedDiscovery = (Array.isArray(selectedDataset?.discoveryJsons) ? selectedDataset.discoveryJsons : []).find(
    (row) => String(row?.path || "").trim() === discoveryJson,
  ) || null;
  const runMode = normalizeOfflineEvalRunMode(state.offlineEval.modalRunMode);
  const usesHeuristicDiscovery = Boolean(selectedDiscovery);
  const selectionCount = usesHeuristicDiscovery
    ? offlineEvalCountForScenarioSplit(
      selectedDiscovery,
      "classCount",
      "classCountBySplit",
      scenarioSplit,
    )
    : offlineEvalCountForScenarioSplit(
      selectedDataset,
      "fullDatasetCount",
      "fullDatasetCountBySplit",
      scenarioSplit,
    );
  const splitMapCount = offlineEvalScenarioSplitCount(scenarioSplit);
  const splitText = `${offlineEvalScenarioSplitLabel(scenarioSplit)}${
    splitMapCount !== null ? ` (${splitMapCount} maps)` : ""
  }`;
  const matchingRun = findOfflineEvalLatestRunForSource(state.offlineEval.modalSource, runMode);
  const statusLines = [
    `mode: ${runMode === "class_purity" ? "H->C class purity" : "next-state accuracy"}`,
    `dataset root: ${selectedDataset?.displayPath || datasetRoot || "-"}`,
    `discovery json: ${selectedDiscovery?.displayPath || (discoveryJson || "none (evaluate full dataset)")}`,
    `map split: ${splitText}`,
    `workers: ${workerCount}`,
    usesHeuristicDiscovery
      ? `heuristic classes: ${selectionCount ?? defaults?.classCount ?? "-"}`
      : `full dataset transitions: ${selectionCount ?? "-"}`,
  ];
  if (runMode === "class_purity" && !usesHeuristicDiscovery) {
    statusLines.push("class analysis requires a heuristic discovery JSON");
  }
  if (matchingRun) {
    statusLines.push(
      `latest run: ${formatOfflineEvalStatus(matchingRun.status)}`
      + ` | failed ${matchingRun?.metrics?.failedCount ?? 0}`,
    );
  }
  elements.evalModalStatus.textContent = formatLines(statusLines);
}

function closeOfflineEvalModal() {
  state.offlineEval.modalRunId = null;
  state.offlineEval.modalRunMode = "accuracy";
  state.offlineEval.modalBusy = false;
  setElementHidden(elements.evalModal, true);
}

async function openOfflineEvalModal(source, options = {}) {
  const resolvedSource = source || resolveOfflineEvalSource(state.activeRunPayload);
  if (!resolvedSource) return;
  const runMode = normalizeOfflineEvalRunMode(options.runMode);
  state.offlineEval.modalSource = resolvedSource;
  state.offlineEval.modalRunMode = runMode;
  state.offlineEval.modalRunId = String(
    options.runId
    || state.activeRunPayload?.runId
    || "",
  ).trim() || null;
  setElementHidden(elements.evalModal, false);
  if (elements.evalModalTitle) {
    elements.evalModalTitle.textContent = runMode === "class_purity"
      ? "Dynamics Class Analysis"
      : "Offline Heuristic Eval";
  }
  if (elements.evalStartButton) {
    elements.evalStartButton.textContent = runMode === "class_purity"
      ? "Run Class Analysis"
      : "Run Eval";
  }
  elements.evalModalSubtitle.textContent = `${resolvedSource.label || versionLabel(resolvedSource.versionKey)} version tools`;
  elements.evalModalSource.textContent = formatLines([
    `version: ${resolvedSource.label || versionLabel(resolvedSource.versionKey)}`,
    `file: ${resolvedSource.sourcePath || "-"}`,
    `sha1: ${resolvedSource.sourceDigest || "-"}`,
    `lines: ${resolvedSource.sourceLineCount || sourceLineCount(resolvedSource.source)}`,
  ]);
  elements.evalModalStatus.textContent = "Loading eval defaults...";
  try {
    const defaults = await ensureOfflineEvalDefaults();
    populateOfflineEvalCatalog(defaults);
    elements.evalSampleSeed.value = String(defaults.sampleSeed ?? 42);
    elements.evalWorkers.value = String(defaults.workers ?? 16);
    updateOfflineEvalModalStatus();
  } catch (error) {
    elements.evalModalStatus.textContent = formatError(error);
  }
}

async function openOfflineEvalRun(runId, { forceDetailRefresh = true } = {}) {
  const safeRunId = String(runId || "").trim();
  if (!safeRunId) return;
  const currentRunId = String(state.offlineEval.runId || "").trim();
  state.offlineEval.notice = "";
  if (currentRunId !== safeRunId) {
    state.offlineEval.selectedFailureId = null;
    state.offlineEval.selectedClassRowKey = null;
    state.offlineEval.selectedClassRepresentativeId = null;
    state.offlineEval.failureDetail = null;
    state.offlineEval.loadingFailureId = null;
    state.offlineEval.classRepresentativeDetail = null;
    state.offlineEval.loadingClassRepresentativeId = null;
    resetOfflineEvalClassResults();
    clearOfflineEvalImages();
  }
  state.offlineEval.drawerOpen = true;
  state.offlineEval.runId = safeRunId;
  renderOfflineEvalDrawer();
  await refreshOfflineEvalRun(forceDetailRefresh);
  await syncOfflineEvalExportStatus(safeRunId, { pollIfRunning: true });
}

async function openOfflineEvalFromSource(source, options = {}) {
  const runMode = normalizeOfflineEvalRunMode(options.runMode);
  const resolvedSource = source || resolveOfflineEvalSource(state.activeRunPayload);
  if (!resolvedSource) return;
  await loadOfflineEvalRuns(true);
  const matchingRun = findOfflineEvalLatestRunForSource(resolvedSource, runMode);
  const currentRunId = String(state.offlineEval.runId || "").trim();
  const matchingRunId = String(matchingRun?.runId || "").trim();
  if (matchingRunId && (!state.offlineEval.drawerOpen || currentRunId !== matchingRunId)) {
    await openOfflineEvalRun(matchingRunId, { forceDetailRefresh: true });
    return;
  }
  await openOfflineEvalModal(resolvedSource, {
    runId: state.activeRunPayload?.runId,
    runMode,
  });
}

function resolveOfflineEvalFailureRows() {
  return state.offlineEval.classResults.filter((row) => String(row?.failureId || "").trim());
}

function resolveOfflineEvalVisibleRows() {
  return Array.isArray(state.offlineEval.classResults) ? state.offlineEval.classResults : [];
}

function resolveOfflineEvalEmptyRowsMessage(runSummary) {
  if (state.offlineEval.classPageUnavailable) {
    return "Eval rows are temporarily unavailable. Progress updates will continue.";
  }
  if (!state.offlineEval.classPageHydrated) {
    return "Loading eval rows...";
  }

  const statusFilter = String(state.offlineEval.statusFilter || "failed").trim() || "failed";
  const metrics = runSummary?.metrics && typeof runSummary.metrics === "object" ? runSummary.metrics : {};
  const progress = runSummary?.progress && typeof runSummary.progress === "object" ? runSummary.progress : {};
  const isActiveRun = isOfflineEvalRunActive(runSummary);

  if (statusFilter === "failed" && Number(metrics?.failedCount || 0) <= 0) {
    return isActiveRun ? "No failed results yet." : "No failed results.";
  }
  if (statusFilter === "pass" && Number(metrics?.correctCount || 0) <= 0) {
    return isActiveRun ? "No passing results yet." : "No passing results.";
  }
  if (statusFilter === "all" && Number(progress?.completedCount || 0) <= 0) {
    return isActiveRun ? "Waiting for the first eval rows..." : "No eval rows were recorded.";
  }
  return "No eval rows match the current filter.";
}

function buildOfflineEvalClassesUrl(runId, options = null) {
  const offset = Math.max(0, Number(options?.offset) || 0);
  const limit = Math.max(1, Number(options?.limit) || OFFLINE_EVAL_CLASS_PAGE_SIZE);
  const status = String(options?.status || "all").trim() || "all";
  const params = new URLSearchParams({
    offset: `${offset}`,
    limit: `${limit}`,
    status,
    sort: "desc",
  });
  return `/api/eval/runs/${encodeURIComponent(runId)}/classes?${params.toString()}`;
}

function mergeOfflineEvalClassRows(existingRows, nextRows) {
  const merged = [];
  const seen = new Set();
  for (const row of [...existingRows, ...nextRows]) {
    const rowKey = `${String(row?.classIdx ?? "")}:${String(row?.failureId ?? "")}:${String(row?.status ?? "")}`;
    if (seen.has(rowKey)) continue;
    seen.add(rowKey);
    merged.push(row);
  }
  return merged;
}

function resolveOfflineEvalAvailableImages(failureDetail) {
  const available = failureDetail?.availableImages;
  if (available && typeof available === "object") {
    return {
      previous: Boolean(available.previous),
      expected: Boolean(available.expected),
      predicted: Boolean(available.predicted),
      comparison: Boolean(available.comparison),
    };
  }
  const predictedAvailable = Boolean(String(failureDetail?.predictedCanonical || "").trim());
  return {
    previous: true,
    expected: true,
    predicted: predictedAvailable,
    comparison: predictedAvailable,
  };
}

function setOfflineEvalImageCardVisible(imageElement, visible) {
  const card = imageElement?.closest(".discovery-eval-image-card");
  if (card) {
    card.hidden = !visible;
  }
}

function setOfflineEvalImageCardState(imageElement, stateName, requestKey = "") {
  const card = imageElement?.closest(".discovery-eval-image-card");
  if (!card) return;
  if (stateName) {
    card.dataset.imageState = stateName;
  } else {
    delete card.dataset.imageState;
  }
  if (requestKey) {
    card.dataset.imageRequest = requestKey;
  } else {
    delete card.dataset.imageRequest;
  }
}

function normalizeOfflineEvalImageMetaLabel(label) {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized === "world") return "World";
  if (normalized === "map") return "Map";
  if (normalized === "action") return "Action";
  if (normalized === "terminated" || normalized === "done") return "Done";
  return null;
}

function normalizeOfflineEvalDoneValue(value) {
  if (typeof value === "boolean") {
    return formatTerminatedLabel(value);
  }
  const text = String(value || "").trim().toLowerCase();
  if (text === "true") {
    return formatTerminatedLabel(true);
  }
  if (text === "false") {
    return formatTerminatedLabel(false);
  }
  return String(value || "").trim();
}

function extractOfflineEvalImageMetaParts(metaPayload = null) {
  const items = [];
  const detailNotes = [];
  const seenKeys = new Set();
  const lines = Array.isArray(metaPayload?.lines)
    ? metaPayload.lines.map((line) => String(line || "").trim()).filter(Boolean)
    : [];
  for (const line of lines) {
    const segments = line.split("|").map((segment) => String(segment || "").trim()).filter(Boolean);
    for (const segment of segments) {
      const match = segment.match(/^(WORLD|MAP|ACTION|TERMINATED|DONE)\s+(.+)$/i);
      if (!match) {
        detailNotes.push(segment);
        continue;
      }
      const normalizedLabel = normalizeOfflineEvalImageMetaLabel(match[1]);
      const rawValueText = String(match[2] || "").trim();
      const valueText = normalizedLabel === "Done"
        ? normalizeOfflineEvalDoneValue(rawValueText)
        : rawValueText;
      if (!normalizedLabel || !valueText) {
        continue;
      }
      const dedupeKey = `${normalizedLabel}:${valueText}`;
      if (seenKeys.has(dedupeKey)) {
        continue;
      }
      seenKeys.add(dedupeKey);
      items.push({ label: normalizedLabel, value: valueText });
    }
  }

  const fieldEntries = [
    ["World", metaPayload?.worldIndex],
    ["Map", metaPayload?.mapName],
    ["Action", metaPayload?.action],
    ["Done", metaPayload?.terminated],
  ];
  for (const [label, rawValue] of fieldEntries) {
    if (rawValue === null || rawValue === undefined || rawValue === "") continue;
    const valueText = label === "Done"
      ? normalizeOfflineEvalDoneValue(rawValue)
      : String(rawValue).trim();
    if (!valueText) continue;
    const dedupeKey = `${label}:${valueText}`;
    if (seenKeys.has(dedupeKey)) continue;
    seenKeys.add(dedupeKey);
    items.push({ label, value: valueText });
  }

  return {
    items,
    detailNotes,
  };
}

function setOfflineEvalImageCardMeta(metaElement, metaPayload = null) {
  if (!metaElement) return;
  metaElement.innerHTML = "";
  const { items, detailNotes } = extractOfflineEvalImageMetaParts(metaPayload);
  const card = metaElement.closest(".discovery-eval-image-card");
  const headMetaElement = card?.querySelector(".discovery-eval-image-card-head-meta") || null;
  if (headMetaElement) {
    headMetaElement.innerHTML = "";
  }
  const kind = String(card?.dataset.kind || "").trim();
  const isActiveSelection = kind && kind === state.offlineEval.activeImageKind;
  const headMetaItems = isActiveSelection
    ? items.filter((item) => {
      const label = String(item?.label || "").trim();
      return label === "Action" || label === "Done";
    })
    : [];
  const filteredItems = items.filter((item) => {
    const label = String(item?.label || "").trim();
    return (
      label !== "World"
      && label !== "Map"
      && label !== "Action"
      && label !== "Done"
    );
  });
  const visibleDetailNotes = kind === "comparison" || isActiveSelection ? detailNotes : [];

  if (headMetaElement) {
    for (const item of headMetaItems) {
      headMetaElement.append(createHudItemChip(item));
    }
    headMetaElement.hidden = headMetaElement.childElementCount <= 0;
  }

  if (filteredItems.length) {
    const chipRow = document.createElement("div");
    chipRow.className = "discovery-eval-image-card-meta-chips";
    for (const item of filteredItems) {
      chipRow.append(createHudItemChip(item));
    }
    metaElement.append(chipRow);
  }

  if (visibleDetailNotes.length) {
    const noteBlock = document.createElement("div");
    noteBlock.className = "discovery-eval-image-card-meta-notes";
    for (const note of visibleDetailNotes) {
      const line = document.createElement("span");
      line.className = "discovery-eval-image-card-meta-line";
      line.textContent = note;
      noteBlock.append(line);
    }
    metaElement.append(noteBlock);
  }

  metaElement.hidden = metaElement.childElementCount <= 0;
}

function clearOfflineEvalImageMeta() {
  for (const element of [
    elements.evalImageMetaPrevious,
    elements.evalImageMetaExpected,
    elements.evalImageMetaPredicted,
    elements.evalImageMetaComparison,
  ]) {
    setOfflineEvalImageCardMeta(element, null);
  }
}

function applyOfflineEvalImageMeta(failureDetail) {
  const imageMeta = failureDetail?.imageMeta && typeof failureDetail.imageMeta === "object"
    ? failureDetail.imageMeta
    : {};
  setOfflineEvalImageCardMeta(elements.evalImageMetaPrevious, imageMeta.previous || null);
  setOfflineEvalImageCardMeta(elements.evalImageMetaExpected, imageMeta.expected || null);
  setOfflineEvalImageCardMeta(elements.evalImageMetaPredicted, imageMeta.predicted || null);
  setOfflineEvalImageCardMeta(elements.evalImageMetaComparison, imageMeta.comparison || null);
}

function resolveOfflineEvalImageCard(kind) {
  const host = elements.evalDrawer || document;
  return host.querySelector(`.discovery-eval-image-card[data-kind="${kind}"]`);
}

function pickOfflineEvalActiveImageKind(visibleImages = null, preferredKind = null) {
  const visibleMap = visibleImages && typeof visibleImages === "object"
    ? visibleImages
    : {
      previous: false,
      expected: false,
      predicted: false,
      comparison: false,
    };
  const normalizedPreferredKind = String(preferredKind || "").trim();
  const kindOrder = [
    ...(normalizedPreferredKind && OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS.includes(normalizedPreferredKind)
      ? [normalizedPreferredKind]
      : []),
    ...OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS.filter((kind) => kind !== normalizedPreferredKind),
  ];
  return kindOrder.find((kind) => Boolean(visibleMap[kind])) || null;
}

function updateOfflineEvalImageGridLayout(visibleImages = null) {
  const grid = elements.evalImageGrid;
  if (!grid) return;
  const visibleMap = visibleImages && typeof visibleImages === "object"
    ? visibleImages
    : {
      previous: false,
      expected: false,
      predicted: false,
      comparison: false,
    };
  const activeKind = pickOfflineEvalActiveImageKind(visibleMap, state.offlineEval.activeImageKind);
  state.offlineEval.activeImageKind = activeKind || "expected";

  const visibleKinds = OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS.filter((kind) => Boolean(visibleMap[kind]));
  const secondaryCardCount = Math.max(
    1,
    visibleKinds.filter((kind) => kind !== activeKind).length || 1,
  );
  grid.style.setProperty("--offline-eval-thumbnail-columns", String(Math.min(3, secondaryCardCount)));

  const orderedKinds = [
    ...(activeKind ? [activeKind] : []),
    ...OFFLINE_EVAL_SELECTABLE_IMAGE_KINDS.filter((kind) => kind !== activeKind),
  ];
  for (const kind of orderedKinds) {
    const card = resolveOfflineEvalImageCard(kind);
    if (!card) continue;
    const isVisible = Boolean(visibleMap[kind]);
    card.hidden = !isVisible;
    card.classList.toggle("is-active", Boolean(isVisible && activeKind === kind));
    card.setAttribute("aria-pressed", activeKind === kind ? "true" : "false");
    card.title = activeKind === kind
      ? "Currently focused. Click the image to open the PNG in a new tab."
      : "Click to focus this panel.";
    const stateBadge = card.querySelector(".discovery-eval-image-card-state");
    if (stateBadge) {
      stateBadge.textContent = activeKind === kind ? "Viewing" : "Click to focus";
    }
    grid.append(card);
  }

  const comparisonCard = resolveOfflineEvalImageCard("comparison");
  if (comparisonCard) {
    comparisonCard.hidden = !Boolean(visibleMap.comparison);
    comparisonCard.classList.remove("is-active");
  }
}

function showOfflineEvalImageLoadingState(visibleImages = null) {
  state.offlineEval.imageFailureKey = null;
  clearOfflineEvalImageMeta();
  const visibleMap = visibleImages && typeof visibleImages === "object"
    ? visibleImages
    : {
      previous: true,
      expected: true,
      predicted: true,
      comparison: true,
    };
  updateOfflineEvalImageGridLayout(visibleMap);
  for (const [imageElement, kind] of [
    [elements.evalImagePrevious, "previous"],
    [elements.evalImageExpected, "expected"],
    [elements.evalImagePredicted, "predicted"],
    [elements.evalImageComparison, "comparison"],
  ]) {
    if (!imageElement) continue;
    const isVisible = Boolean(visibleMap?.[kind]);
    setOfflineEvalImageCardVisible(imageElement, isVisible);
    const link = imageElement.closest("a");
    imageElement.onload = null;
    imageElement.onerror = null;
    imageElement.removeAttribute("src");
    if (link) {
      link.removeAttribute("href");
      link.removeAttribute("title");
    }
    setOfflineEvalImageCardState(
      imageElement,
      isVisible ? "loading" : "hidden",
      isVisible ? `${Date.now()}:${kind}` : "",
    );
  }
}

function applyOfflineEvalImageSourcesFromBase(imageKey, basePath, availableImages = null) {
  updateOfflineEvalImageGridLayout(availableImages);
  if (state.offlineEval.imageFailureKey === imageKey) {
    for (const [imageElement, kind] of [
      [elements.evalImagePrevious, "previous"],
      [elements.evalImageExpected, "expected"],
      [elements.evalImagePredicted, "predicted"],
      [elements.evalImageComparison, "comparison"],
    ]) {
      if (!imageElement) continue;
      setOfflineEvalImageCardVisible(imageElement, Boolean(availableImages?.[kind]));
    }
    return;
  }
  state.offlineEval.imageFailureKey = imageKey;
  const revision = encodeURIComponent(imageKey);
  for (const [imageElement, kind] of [
    [elements.evalImagePrevious, "previous"],
    [elements.evalImageExpected, "expected"],
    [elements.evalImagePredicted, "predicted"],
    [elements.evalImageComparison, "comparison"],
  ]) {
    if (!imageElement) continue;
    const isVisible = Boolean(availableImages?.[kind]);
    setOfflineEvalImageCardVisible(imageElement, isVisible);
    const link = imageElement.closest("a");
    imageElement.onload = null;
    imageElement.onerror = null;
    if (!isVisible) {
      imageElement.removeAttribute("src");
      if (link) {
        link.removeAttribute("href");
        link.removeAttribute("title");
      }
      setOfflineEvalImageCardState(imageElement, "hidden");
      continue;
    }
    const url = `${basePath}/${kind}?rev=${revision}`;
    const requestKey = `${imageKey}:${kind}`;
    setOfflineEvalImageCardState(imageElement, "loading", requestKey);
    imageElement.removeAttribute("src");
    if (link) {
      link.removeAttribute("href");
      link.removeAttribute("title");
    }
    imageElement.onload = () => {
      const card = imageElement.closest(".discovery-eval-image-card");
      if (!card || card.dataset.imageRequest !== requestKey) return;
      setOfflineEvalImageCardState(imageElement, "ready", requestKey);
      if (link) {
        link.href = url;
        link.title = `${kind} image`;
      }
    };
    imageElement.onerror = () => {
      const card = imageElement.closest(".discovery-eval-image-card");
      if (!card || card.dataset.imageRequest !== requestKey) return;
      setOfflineEvalImageCardState(imageElement, "error", requestKey);
      imageElement.removeAttribute("src");
      if (link) {
        link.removeAttribute("href");
        link.removeAttribute("title");
      }
    };
    imageElement.src = url;
  }
}

function applyOfflineEvalImageSources(runId, failureId, availableImages = null) {
  const basePath = `/api/eval/runs/${encodeURIComponent(runId)}/failures/${encodeURIComponent(failureId)}/render`;
  applyOfflineEvalImageSourcesFromBase(`${runId}:failure:${failureId}`, basePath, availableImages);
}

function applyOfflineEvalClassRepresentativeImageSources(runId, representativeId, availableImages = null) {
  const basePath = `/api/eval/runs/${encodeURIComponent(runId)}/class-representatives/${encodeURIComponent(representativeId)}/render`;
  applyOfflineEvalImageSourcesFromBase(`${runId}:class-representative:${representativeId}`, basePath, availableImages);
}

function currentOfflineEvalImageDetail() {
  return isOfflineEvalClassPurity(state.offlineEval.runSummary)
    ? state.offlineEval.classRepresentativeDetail
    : state.offlineEval.failureDetail;
}

function clearOfflineEvalImages() {
  state.offlineEval.imageFailureKey = null;
  state.offlineEval.activeImageKind = "expected";
  clearOfflineEvalImageMeta();
  updateOfflineEvalImageGridLayout({
    previous: false,
    expected: false,
    predicted: false,
    comparison: false,
  });
  for (const element of [
    elements.evalImagePrevious,
    elements.evalImageExpected,
    elements.evalImagePredicted,
    elements.evalImageComparison,
  ]) {
    if (element) {
      setOfflineEvalImageCardVisible(element, false);
      setOfflineEvalImageCardState(element, "hidden");
      element.onload = null;
      element.onerror = null;
      element.removeAttribute("src");
      const link = element.closest("a");
      if (link) {
        link.removeAttribute("href");
        link.removeAttribute("title");
      }
    }
  }
}

function renderOfflineEvalDrawer() {
  const {
    drawerOpen,
    runSummary,
    failureDetail,
    selectedFailureId,
    loadingFailureId,
    classRepresentativeDetail,
    selectedClassRepresentativeId,
    loadingClassRepresentativeId,
  } = state.offlineEval;
  setElementHidden(elements.evalDrawer, !drawerOpen);
  const rerunSource = resolveOfflineEvalSourceForRunSummary(runSummary);
  if (elements.evalRerunButton) {
    elements.evalRerunButton.disabled = !rerunSource;
    elements.evalRerunButton.title = rerunSource
      ? `Re-evaluate ${rerunSource.label || versionLabel(rerunSource.versionKey)}`
      : "This version source is not available in the current discovery run.";
  }
  if (elements.evalStatusFilter && elements.evalStatusFilter.value !== state.offlineEval.statusFilter) {
    elements.evalStatusFilter.value = state.offlineEval.statusFilter;
  }
  if (!drawerOpen) {
    return;
  }
  if (!runSummary) {
    elements.evalDrawerSubtitle.textContent = "No offline eval run selected.";
    elements.evalSummary.innerHTML = `<div class="discovery-eval-summary-empty">Run an eval from Version Navigator.</div>`;
    elements.evalClassList.innerHTML = "";
    elements.evalDetailMeta.innerHTML = `<div class="discovery-eval-detail-empty">No failure selected.</div>`;
    clearOfflineEvalImages();
    return;
  }

  const classCount = Number(runSummary?.dataset?.classCount ?? runSummary?.progress?.totalCount ?? 0);
  const resultLabelPlural = offlineEvalResultLabelPlural(runSummary?.dataset);
  const resultPrefix = offlineEvalRowPrefix(runSummary);
  const classPurityMode = isOfflineEvalClassPurity(runSummary);
  const selectedClassRow = classPurityMode ? resolveSelectedOfflineEvalClassRow() : null;
  elements.evalDrawerSubtitle.innerHTML = renderOfflineEvalDrawerSubtitleMarkup(runSummary);
  elements.evalSummary.innerHTML = renderOfflineEvalSummaryMarkup(
    runSummary,
    state.offlineEval.notice,
    state.offlineEval.exportStatus,
  );

  const visibleRows = resolveOfflineEvalVisibleRows();
  elements.evalClassList.innerHTML = "";
  if (!visibleRows.length) {
    const empty = document.createElement("div");
    empty.className = "meta-block";
    empty.textContent = resolveOfflineEvalEmptyRowsMessage(runSummary);
    elements.evalClassList.append(empty);
  } else {
    for (const row of visibleRows) {
      const failureId = String(row.failureId || "").trim() || null;
      const rowKey = offlineEvalClassResultRowKey(row);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `discovery-eval-class-item is-${String(row.status || "unknown")}`;
      button.dataset.discoveryAction = "select-offline-eval-class";
      button.dataset.rowKey = rowKey;
      button.dataset.failureId = failureId || "";
      button.classList.toggle(
        "is-selected",
        classPurityMode
          ? Boolean(rowKey && rowKey === state.offlineEval.selectedClassRowKey)
          : Boolean(failureId && failureId === selectedFailureId),
      );
      const statusLabel = formatOfflineEvalResultStatus(row.status, runSummary);
      const scenarioText = formatMapDisplayName(row.scenarioType, row.artifactStem) || "-";
      const transitionText = row.transitionIndex !== null && row.transitionIndex !== undefined
        ? `#${row.transitionIndex}`
        : "";
      const purityText = isOfflineEvalClassPurity(runSummary) && row.purity !== undefined
        ? `purity ${formatPercent(row.purity)}`
        : "";
      const repairClassText = isOfflineEvalClassPurity(runSummary) && row.majorityRepairClassId
        ? `C${row.majorityRepairClassId}`
        : "";
      const metaParts = [
        scenarioText,
        transitionText,
        repairClassText,
        purityText,
      ].filter(Boolean);
      button.innerHTML = `
        <span class="discovery-eval-class-main">
          <span class="discovery-eval-class-primary">${resultPrefix}${String(row.classIdx || "-").padStart(3, "0")} · ${String(row.action || "-").toUpperCase()}</span>
          <span class="discovery-eval-status-badge is-${String(row.status || "unknown")}">${statusLabel}</span>
        </span>
        <span class="discovery-eval-class-meta">${escapeHtml(metaParts.join(" · "))}</span>
      `;
      elements.evalClassList.append(button);
    }
  }
  if (elements.evalClassLoadMore) {
    const shownCount = visibleRows.length;
    const totalCount = Math.max(0, Number(state.offlineEval.classPageTotalCount || shownCount));
    const hasMore = Boolean(state.offlineEval.classPageHasMore);
    const isLoadingMore = Boolean(state.offlineEval.classPageLoading && shownCount > 0);
    elements.evalClassLoadMore.hidden = !hasMore && !isLoadingMore;
    elements.evalClassLoadMore.disabled = !hasMore || isLoadingMore;
    elements.evalClassLoadMore.textContent = isLoadingMore
      ? "Loading..."
      : `Load More (${shownCount}/${totalCount})`;
  }

  const failureRows = resolveOfflineEvalFailureRows();
  const hasFailureDetail = Boolean(failureDetail && selectedFailureId);
  const isFailureLoading = Boolean(
    selectedFailureId
    && String(loadingFailureId || "").trim() === String(selectedFailureId || "").trim()
    && (!failureDetail || String(failureDetail.failureId || "").trim() !== String(selectedFailureId || "").trim()),
  );
  if (elements.evalSaveAll) {
    const exportBusy = Boolean(state.offlineEval.exportBusy);
    const exportStatus = state.offlineEval.exportStatus;
    const exportStatusKey = String(exportStatus?.status || "").trim().toLowerCase();
    const exportIsCancelling = exportStatusKey === "cancelling";
    const exportButtonText = exportBusy
      ? exportIsCancelling
        ? "Cancelling..."
        : "Cancel Export"
      : "Export Images";
    elements.evalSaveAll.disabled = exportIsCancelling || (!exportBusy && failureRows.length <= 0);
    elements.evalSaveAll.textContent = exportButtonText;
    elements.evalSaveAll.title = exportBusy
      ? exportIsCancelling
        ? String(exportStatus?.message || "Cancellation request is being processed.")
        : "Cancel the current export job"
      : failureRows.length > 0
        ? "Save all failure images"
        : "No failure images to save.";
  }

  if (classPurityMode) {
    if (!selectedClassRow) {
      elements.evalDetailMeta.innerHTML = `<div class="discovery-eval-detail-empty">Select a class result to inspect.</div>`;
      clearOfflineEvalImages();
      return;
    }
    const selectedRepresentative = resolveOfflineEvalClassRepresentative(
      selectedClassRow,
      selectedClassRepresentativeId,
    );
    const representativeId = String(selectedRepresentative?.representativeId || selectedClassRepresentativeId || "").trim();
    const hasRepresentativeDetail = Boolean(
      representativeId
      && classRepresentativeDetail
      && String(classRepresentativeDetail.representativeId || "").trim() === representativeId
    );
    const isRepresentativeLoading = Boolean(
      representativeId
      && String(loadingClassRepresentativeId || "").trim() === representativeId
      && !hasRepresentativeDetail
    );
    elements.evalDetailMeta.innerHTML = renderOfflineEvalClassPurityDetailMarkup(
      selectedClassRow,
      representativeId,
    );
    if (isRepresentativeLoading) {
      showOfflineEvalImageLoadingState();
      return;
    }
    if (hasRepresentativeDetail) {
      const availableImages = resolveOfflineEvalAvailableImages(classRepresentativeDetail);
      state.offlineEval.activeImageKind = pickOfflineEvalActiveImageKind(
        availableImages,
        state.offlineEval.activeImageKind,
      ) || "expected";
      applyOfflineEvalImageMeta(classRepresentativeDetail);
      applyOfflineEvalClassRepresentativeImageSources(
        runSummary.runId,
        representativeId,
        availableImages,
      );
      return;
    }
    clearOfflineEvalImages();
    return;
  }

  if (!hasFailureDetail) {
    if (isFailureLoading) {
      elements.evalDetailMeta.innerHTML = `<div class="discovery-eval-detail-empty">Rendering selected failure...</div>`;
      showOfflineEvalImageLoadingState();
    } else {
      elements.evalDetailMeta.innerHTML = `<div class="discovery-eval-detail-empty">Select a failed result to inspect.</div>`;
      clearOfflineEvalImages();
    }
    return;
  }

  elements.evalDetailMeta.innerHTML = renderOfflineEvalFailureMetaMarkup(failureDetail, runSummary);
  state.offlineEval.activeImageKind = pickOfflineEvalActiveImageKind(
    resolveOfflineEvalAvailableImages(failureDetail),
    state.offlineEval.activeImageKind,
  ) || "expected";
  applyOfflineEvalImageMeta(failureDetail);
  applyOfflineEvalImageSources(
    runSummary.runId,
    failureDetail.failureId,
    resolveOfflineEvalAvailableImages(failureDetail),
  );
}

function selectOfflineEvalClassRow(rowKey) {
  const safeRowKey = String(rowKey || "").trim();
  state.offlineEval.selectedClassRowKey = safeRowKey || null;
  state.offlineEval.selectedFailureId = null;
  state.offlineEval.failureDetail = null;
  state.offlineEval.loadingFailureId = null;
  const selectedRow = resolveSelectedOfflineEvalClassRow();
  const representativeId = defaultOfflineEvalClassRepresentativeId(selectedRow);
  state.offlineEval.selectedClassRepresentativeId = representativeId;
  state.offlineEval.classRepresentativeDetail = null;
  state.offlineEval.loadingClassRepresentativeId = null;
  renderOfflineEvalDrawer();
  if (representativeId) {
    selectOfflineEvalClassRepresentative(representativeId).catch((error) => {
      state.offlineEval.notice = formatError(error);
      renderOfflineEvalDrawer();
    });
  }
}

async function selectOfflineEvalClassRepresentative(representativeId, { force = false } = {}) {
  const runId = state.offlineEval.runId;
  const safeRepresentativeId = String(representativeId || "").trim();
  state.offlineEval.selectedClassRepresentativeId = safeRepresentativeId || null;
  state.offlineEval.selectedFailureId = null;
  state.offlineEval.failureDetail = null;
  state.offlineEval.loadingFailureId = null;
  if (!runId || !safeRepresentativeId) {
    state.offlineEval.classRepresentativeDetail = null;
    state.offlineEval.loadingClassRepresentativeId = null;
    renderOfflineEvalDrawer();
    return;
  }
  if (!force && state.offlineEval.classRepresentativeDetail?.representativeId === safeRepresentativeId) {
    renderOfflineEvalDrawer();
    return;
  }
  const currentRepresentativeId = String(state.offlineEval.classRepresentativeDetail?.representativeId || "").trim();
  if (currentRepresentativeId !== safeRepresentativeId) {
    state.offlineEval.classRepresentativeDetail = null;
  }
  state.offlineEval.loadingClassRepresentativeId = safeRepresentativeId;
  renderOfflineEvalDrawer();
  try {
    const payload = await fetchJson(
      `/api/eval/runs/${encodeURIComponent(runId)}/class-representatives/${encodeURIComponent(safeRepresentativeId)}`,
    );
    if (String(state.offlineEval.runId || "").trim() !== String(runId || "").trim()) {
      return;
    }
    if (String(state.offlineEval.selectedClassRepresentativeId || "").trim() !== safeRepresentativeId) {
      return;
    }
    state.offlineEval.loadingClassRepresentativeId = null;
    state.offlineEval.classRepresentativeDetail = payload;
    renderOfflineEvalDrawer();
  } catch (error) {
    if (
      String(state.offlineEval.runId || "").trim() === String(runId || "").trim()
      && String(state.offlineEval.selectedClassRepresentativeId || "").trim() === safeRepresentativeId
    ) {
      state.offlineEval.loadingClassRepresentativeId = null;
    }
    throw error;
  }
}

async function selectOfflineEvalFailure(failureId, { force = false } = {}) {
  const runId = state.offlineEval.runId;
  const safeFailureId = String(failureId || "").trim();
  state.offlineEval.selectedFailureId = safeFailureId || null;
  state.offlineEval.selectedClassRowKey = null;
  state.offlineEval.selectedClassRepresentativeId = null;
  state.offlineEval.classRepresentativeDetail = null;
  state.offlineEval.loadingClassRepresentativeId = null;
  if (!runId || !safeFailureId) {
    state.offlineEval.failureDetail = null;
    state.offlineEval.loadingFailureId = null;
    renderOfflineEvalDrawer();
    return;
  }
  if (!force && state.offlineEval.failureDetail?.failureId === safeFailureId) {
    renderOfflineEvalDrawer();
    return;
  }
  const currentFailureId = String(state.offlineEval.failureDetail?.failureId || "").trim();
  if (currentFailureId !== safeFailureId) {
    state.offlineEval.failureDetail = null;
  }
  state.offlineEval.loadingFailureId = safeFailureId;
  renderOfflineEvalDrawer();
  try {
    const payload = await fetchJson(
      `/api/eval/runs/${encodeURIComponent(runId)}/failures/${encodeURIComponent(safeFailureId)}`,
    );
    if (String(state.offlineEval.runId || "").trim() !== String(runId || "").trim()) {
      return;
    }
    if (String(state.offlineEval.selectedFailureId || "").trim() !== safeFailureId) {
      return;
    }
    state.offlineEval.loadingFailureId = null;
    state.offlineEval.failureDetail = payload;
    renderOfflineEvalDrawer();
  } catch (error) {
    if (
      String(state.offlineEval.runId || "").trim() === String(runId || "").trim()
      && String(state.offlineEval.selectedFailureId || "").trim() === safeFailureId
    ) {
      state.offlineEval.loadingFailureId = null;
    }
    throw error;
  }
}

async function loadOfflineEvalClassResults(options = null) {
  let forceDetailRefresh = Boolean(options?.forceDetailRefresh);
  const reset = Boolean(options?.reset);
  const runId = String(state.offlineEval.runId || "").trim();
  if (!runId) return null;
  const shouldRenderLoadingState = reset && !state.offlineEval.classPageHydrated;
  const requestToken = state.offlineEval.classPageRequestToken + 1;
  state.offlineEval.classPageRequestToken = requestToken;
  state.offlineEval.classPageLoading = true;
  const offset = reset ? 0 : Number(state.offlineEval.classPageOffset || 0);
  const limit = Math.max(1, Number(state.offlineEval.classPageLimit || OFFLINE_EVAL_CLASS_PAGE_SIZE));
  const statusFilter = String(state.offlineEval.statusFilter || "failed").trim() || "failed";
  if (shouldRenderLoadingState) {
    renderOfflineEvalDrawer();
  }
  try {
    const classPayload = await fetchJson(
      buildOfflineEvalClassesUrl(runId, {
        offset,
        limit,
        status: statusFilter,
      }),
    );
    if (
      String(state.offlineEval.runId || "").trim() !== runId
      || state.offlineEval.classPageRequestToken !== requestToken
      || String(state.offlineEval.statusFilter || "failed").trim() !== statusFilter
    ) {
      return null;
    }

    const rows = Array.isArray(classPayload?.classes) ? classPayload.classes : [];
    const page = classPayload?.page || {};
    if (Boolean(page?.temporarilyUnavailable)) {
      state.offlineEval.classPageHydrated = true;
      state.offlineEval.classPageUnavailable = true;
      state.offlineEval.classPageHasMore = false;
      state.offlineEval.classPageTotalCount = Math.max(
        Number(page?.totalCount || 0),
        state.offlineEval.classResults.length,
      );
      if (page?.message) {
        state.offlineEval.notice = String(page.message).trim();
      }
      renderOfflineEvalDrawer();
      return classPayload;
    }
    state.offlineEval.classPageHydrated = true;
    state.offlineEval.classPageUnavailable = false;
    state.offlineEval.classResults = reset
      ? rows
      : mergeOfflineEvalClassRows(state.offlineEval.classResults, rows);
    state.offlineEval.classPageOffset = offset + rows.length;
    state.offlineEval.classPageHasMore = Boolean(page?.hasMore);
    state.offlineEval.classPageTotalCount = Math.max(
      Number(page?.totalCount || 0),
      state.offlineEval.classResults.length,
    );

    if (isOfflineEvalClassPurity(state.offlineEval.runSummary)) {
      const visibleRows = resolveOfflineEvalVisibleRows();
      const hasSelectedClassRow = visibleRows.some(
        (row) => offlineEvalClassResultRowKey(row) === String(state.offlineEval.selectedClassRowKey || "").trim(),
      );
      if (!hasSelectedClassRow) {
        state.offlineEval.selectedClassRowKey = visibleRows.length
          ? offlineEvalClassResultRowKey(visibleRows[0])
          : null;
      }
      state.offlineEval.selectedFailureId = null;
      state.offlineEval.failureDetail = null;
      state.offlineEval.loadingFailureId = null;
      const selectedRow = resolveSelectedOfflineEvalClassRow();
      const representativeId = defaultOfflineEvalClassRepresentativeId(selectedRow);
      if (representativeId && state.offlineEval.selectedClassRepresentativeId !== representativeId) {
        state.offlineEval.selectedClassRepresentativeId = representativeId;
        state.offlineEval.classRepresentativeDetail = null;
        state.offlineEval.loadingClassRepresentativeId = null;
      } else if (!representativeId) {
        state.offlineEval.selectedClassRepresentativeId = null;
        state.offlineEval.classRepresentativeDetail = null;
        state.offlineEval.loadingClassRepresentativeId = null;
      }
      renderOfflineEvalDrawer();
      if (representativeId) {
        selectOfflineEvalClassRepresentative(representativeId, { force: forceDetailRefresh }).catch((error) => {
          state.offlineEval.notice = formatError(error);
          renderOfflineEvalDrawer();
        });
      }
    } else {
      const failureRows = resolveOfflineEvalFailureRows();
      const hasSelectedFailure = failureRows.some(
        (row) => String(row.failureId || "").trim() === String(state.offlineEval.selectedFailureId || "").trim(),
      );
      if (!hasSelectedFailure) {
        state.offlineEval.selectedFailureId = String(
          failureRows[0]?.failureId
          || "",
        ).trim() || null;
        state.offlineEval.failureDetail = null;
        state.offlineEval.loadingFailureId = null;
        forceDetailRefresh = true;
      }
      if (state.offlineEval.selectedFailureId) {
        await selectOfflineEvalFailure(state.offlineEval.selectedFailureId, { force: forceDetailRefresh });
      } else {
        state.offlineEval.failureDetail = null;
        state.offlineEval.loadingFailureId = null;
      }
    }
    return classPayload;
  } finally {
    if (state.offlineEval.classPageRequestToken === requestToken) {
      state.offlineEval.classPageLoading = false;
      renderOfflineEvalDrawer();
    }
  }
}

async function refreshOfflineEvalRun(forceDetailRefresh = false) {
  const runId = String(state.offlineEval.runId || "").trim();
  if (!runId || state.offlineEval.pollInFlight) return;
  state.offlineEval.pollInFlight = true;
  try {
    const previousProgressSnapshotKey = offlineEvalProgressSnapshotKey(state.offlineEval.runSummary);
    const runSummary = await fetchJson(`/api/eval/runs/${encodeURIComponent(runId)}`);
    if (String(state.offlineEval.runId || "").trim() !== runId) {
      return;
    }
    state.offlineEval.runSummary = runSummary;
    upsertOfflineEvalRunSummary(runSummary);
    const nextProgressSnapshotKey = offlineEvalProgressSnapshotKey(runSummary);
    const shouldRefreshClasses = forceDetailRefresh
      || state.offlineEval.classResults.length <= 0
      || previousProgressSnapshotKey !== nextProgressSnapshotKey;
    if (shouldRefreshClasses) {
      try {
        await loadOfflineEvalClassResults({ reset: true, forceDetailRefresh });
      } catch (error) {
        state.offlineEval.notice = formatError(error);
      }
    }
    renderOfflineEvalDrawer();
    if (isOfflineEvalRunActive(runSummary)) {
      scheduleOfflineEvalPoll();
    } else {
      stopOfflineEvalPoll();
    }
  } finally {
    state.offlineEval.pollInFlight = false;
  }
}

async function startOfflineEvalRun() {
  if (state.offlineEval.modalBusy || !state.offlineEval.modalSource) return;
  const runMode = normalizeOfflineEvalRunMode(state.offlineEval.modalRunMode);
  state.offlineEval.modalBusy = true;
  elements.evalModalStatus.textContent = runMode === "class_purity"
    ? "Starting dynamics class analysis worker..."
    : "Starting offline eval worker...";
  try {
    const payload = await fetchJson("/api/eval/runs", {
      method: "POST",
      body: JSON.stringify({
        runId: state.offlineEval.modalRunId || state.activeRunPayload?.runId || null,
        runMode,
        source: state.offlineEval.modalSource.source,
        program: {
          versionKey: state.offlineEval.modalSource.versionKey,
          label: state.offlineEval.modalSource.label,
          sourcePath: state.offlineEval.modalSource.sourcePath,
          sourceDigest: state.offlineEval.modalSource.sourceDigest,
          sourceLineCount: state.offlineEval.modalSource.sourceLineCount,
          runOutputDir: state.activeRunPayload?.runOutputDir || null,
        },
        datasetRoot: String(elements.evalDatasetRoot.value || "").trim(),
        discoveryJson: String(elements.evalDiscoveryJson.value || "").trim(),
        scenarioSplit: currentOfflineEvalScenarioSplit(),
        sampleSeed: optionalInteger(elements.evalSampleSeed.value) ?? 42,
        workers: Math.max(1, optionalInteger(elements.evalWorkers.value) ?? 16),
      }),
    });
    state.offlineEval.notice = "";
    state.offlineEval.drawerOpen = true;
    state.offlineEval.runId = payload.runId;
    state.offlineEval.runSummary = payload;
    upsertOfflineEvalRunSummary(payload);
    resetOfflineEvalClassResults();
    state.offlineEval.selectedFailureId = null;
    state.offlineEval.selectedClassRowKey = null;
    state.offlineEval.selectedClassRepresentativeId = null;
    state.offlineEval.failureDetail = null;
    state.offlineEval.loadingFailureId = null;
    state.offlineEval.classRepresentativeDetail = null;
    state.offlineEval.loadingClassRepresentativeId = null;
    state.offlineEval.activeImageKind = "expected";
    await loadOfflineEvalRuns(true);
    closeOfflineEvalModal();
    renderOfflineEvalDrawer();
    await refreshOfflineEvalRun(true);
  } catch (error) {
    elements.evalModalStatus.textContent = formatError(error);
  } finally {
    state.offlineEval.modalBusy = false;
  }
}

async function loadLatestOfflineEvalRunMaybe() {
  if (state.offlineEval.restoreAttempted) return;
  state.offlineEval.restoreAttempted = true;
  try {
    const runs = await loadOfflineEvalRuns(true);
    const storedExportRunId = loadStoredOfflineEvalExportRunId();
    const latestRun = runs[0] || null;
    const storedExportRun = storedExportRunId
      ? runs.find((row) => String(row?.runId || "").trim() === storedExportRunId) || null
      : null;
    if (storedExportRunId && !storedExportRun) {
      clearStoredOfflineEvalExportRunId(storedExportRunId);
    }
    const restoreRun = storedExportRun || latestRun;
    if (!restoreRun) return;
    state.offlineEval.runId = restoreRun.runId || null;
    state.offlineEval.runSummary = restoreRun;
    const exportStatus = await syncOfflineEvalExportStatus(restoreRun.runId, {
      pollIfRunning: Boolean(storedExportRun),
    });
    const exportStatusKey = String(exportStatus?.status || "").trim().toLowerCase();
    state.offlineEval.drawerOpen = Boolean(
      storedExportRun
      && exportStatusKey
      && exportStatusKey !== "idle",
    );
    if (!state.offlineEval.drawerOpen) {
      state.offlineEval.drawerOpen = isOfflineEvalRunActive(restoreRun);
    }
    if (state.offlineEval.drawerOpen) {
      await refreshOfflineEvalRun(true);
    }
  } catch (_error) {
    // Ignore restore failures during initial page load.
  }
}

function isOfflineEvalExportActive(status) {
  const statusKey = String(status?.status || "").trim().toLowerCase();
  return statusKey === "running" || statusKey === "cancelling";
}

function applyOfflineEvalExportStatus(runId, status) {
  const safeRunId = String(runId || status?.runId || "").trim();
  const statusKey = String(status?.status || "").trim().toLowerCase();
  if (!status || statusKey === "idle") {
    state.offlineEval.exportBusy = false;
    state.offlineEval.exportStatus = null;
    state.offlineEval.notice = "";
    clearStoredOfflineEvalExportRunId(safeRunId || null);
    return "idle";
  }
  state.offlineEval.exportStatus = status;
  state.offlineEval.exportBusy = isOfflineEvalExportActive(status);
  if (state.offlineEval.exportBusy) {
    persistOfflineEvalExportRunId(safeRunId);
  } else {
    clearStoredOfflineEvalExportRunId(safeRunId || null);
  }
  state.offlineEval.notice = String(status?.message || "").trim();
  return statusKey;
}

function resumeOfflineEvalExportPolling(runId) {
  const safeRunId = String(runId || "").trim();
  if (!safeRunId) return;
  const exportToken = state.offlineEval.exportToken + 1;
  state.offlineEval.exportToken = exportToken;
  state.offlineEval.exportBusy = true;
  pollOfflineEvalExportStatus(safeRunId, exportToken).catch((error) => {
    if (state.offlineEval.exportToken !== exportToken) {
      return;
    }
    state.offlineEval.notice = formatError(error);
    state.offlineEval.exportBusy = isOfflineEvalExportActive(state.offlineEval.exportStatus);
    renderOfflineEvalDrawer();
  });
}

async function syncOfflineEvalExportStatus(runId, { pollIfRunning = false } = {}) {
  const safeRunId = String(runId || "").trim();
  if (!safeRunId) return null;
  const wasTrackingSameRun = Boolean(
    state.offlineEval.exportBusy
    && isOfflineEvalExportActive(state.offlineEval.exportStatus)
    && String(state.offlineEval.exportStatus?.runId || "").trim() === safeRunId,
  );
  const status = await fetchJson(`/api/eval/runs/${encodeURIComponent(safeRunId)}/export/status`);
  if (String(state.offlineEval.runId || "").trim() !== safeRunId) {
    return status;
  }
  const statusKey = applyOfflineEvalExportStatus(safeRunId, status);
  renderOfflineEvalDrawer();
  if (pollIfRunning && statusKey === "running" && !wasTrackingSameRun) {
    resumeOfflineEvalExportPolling(safeRunId);
  }
  return status;
}

async function pollOfflineEvalExportStatus(runId, exportToken) {
  while (true) {
    if (state.offlineEval.exportToken !== exportToken) {
      return null;
    }
    const status = await fetchJson(`/api/eval/runs/${encodeURIComponent(runId)}/export/status`);
    if (state.offlineEval.exportToken !== exportToken) {
      return null;
    }
    const statusKey = applyOfflineEvalExportStatus(runId, status);
    renderOfflineEvalDrawer();
    if (!isOfflineEvalExportActive(status)) {
      return status;
    }
    await waitMs(250);
  }
}

async function exportOfflineEvalFailures() {
  const runId = String(state.offlineEval.runId || "").trim();
  if (!runId || state.offlineEval.exportBusy) return;
  const preferredSource = resolveOfflineEvalSourceForRunSummary(state.offlineEval.runSummary);
  const preferredDir = String(
    preferredSource?.sourcePath
    || state.offlineEval.runSummary?.program?.sourcePath
    || "",
  ).trim() || null;
  const exportToken = state.offlineEval.exportToken + 1;
  state.offlineEval.exportToken = exportToken;
  state.offlineEval.exportBusy = true;
  state.offlineEval.exportStatus = {
    completedCount: 0,
    totalCount: resolveOfflineEvalFailureRows().length,
    message: "Choosing export folder...",
  };
  state.offlineEval.notice = "Choosing export folder...";
  renderOfflineEvalDrawer();
  try {
    const picked = await fetchJson(`/api/eval/runs/${encodeURIComponent(runId)}/export/pick-directory`, {
      method: "POST",
      body: JSON.stringify({
        scope: "all_failures",
        preferredDir,
      }),
    });
    if (state.offlineEval.exportToken !== exportToken) {
      return;
    }
    if (picked?.cancelled) {
      state.offlineEval.exportStatus = null;
      state.offlineEval.exportBusy = false;
      state.offlineEval.notice = "Export cancelled.";
      clearStoredOfflineEvalExportRunId(runId);
      renderOfflineEvalDrawer();
      return;
    }
    state.offlineEval.exportStatus = {
      completedCount: 0,
      totalCount: resolveOfflineEvalFailureRows().length,
      message: `Saving 0/${resolveOfflineEvalFailureRows().length} failure image sets...`,
    };
    state.offlineEval.notice = String(state.offlineEval.exportStatus.message);
    renderOfflineEvalDrawer();
    persistOfflineEvalExportRunId(runId);
    const startedStatus = await fetchJson(`/api/eval/runs/${encodeURIComponent(runId)}/export/start`, {
      method: "POST",
      body: JSON.stringify({
        scope: "all_failures",
        destinationDir: picked?.directory || null,
      }),
    });
    if (state.offlineEval.exportToken !== exportToken) {
      return;
    }
    applyOfflineEvalExportStatus(runId, startedStatus);
    renderOfflineEvalDrawer();
    const finalStatus = await pollOfflineEvalExportStatus(runId, exportToken);
    if (state.offlineEval.exportToken !== exportToken || !finalStatus) {
      return;
    }
    if (String(finalStatus?.status || "").trim() === "failed") {
      throw new Error(String(finalStatus?.message || "Export failed."));
    }
    state.offlineEval.notice = String(
      finalStatus?.message || `Exported ${finalStatus?.completedCount || 0} case(s).`,
    ).trim();
  } finally {
    if (state.offlineEval.exportToken === exportToken) {
      state.offlineEval.exportBusy = isOfflineEvalExportActive(state.offlineEval.exportStatus);
      renderOfflineEvalDrawer();
    }
  }
}

async function cancelOfflineEvalExport() {
  const runId = String(state.offlineEval.runId || "").trim();
  if (!runId || !isOfflineEvalExportActive(state.offlineEval.exportStatus)) return;
  const currentStatus = state.offlineEval.exportStatus || {};
  const currentStatusKey = String(currentStatus?.status || "").trim().toLowerCase();
  if (currentStatusKey === "cancelling") return;
  state.offlineEval.exportStatus = {
    ...currentStatus,
    runId,
    status: "cancelling",
    cancelRequested: true,
    message: `Cancelling export after current item (${currentStatus?.completedCount ?? 0}/${currentStatus?.totalCount ?? 0} saved)...`,
  };
  state.offlineEval.exportBusy = true;
  state.offlineEval.notice = String(state.offlineEval.exportStatus.message);
  renderOfflineEvalDrawer();
  const cancelledStatus = await fetchJson(`/api/eval/runs/${encodeURIComponent(runId)}/export/cancel`, {
    method: "POST",
  });
  if (String(state.offlineEval.runId || "").trim() !== runId) {
    return;
  }
  applyOfflineEvalExportStatus(runId, cancelledStatus);
  renderOfflineEvalDrawer();
}

function transitionSelectionRunKey(runPayload) {
  return String(runPayload?.runId || "none");
}

function transitionWitnessSummaryRunKey(runPayload) {
  return transitionSelectionRunKey(runPayload);
}

function defaultRejectFailGroupsExpanded(failGroupCount = 0) {
  return (optionalInteger(failGroupCount) ?? 0) <= 4;
}

function isRejectFailGroupsExpanded(runPayload, failGroupCount = 0) {
  const runKey = transitionWitnessSummaryRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.transitionWitnessFailGroupsExpandedByRun, runKey)) {
    return defaultRejectFailGroupsExpanded(failGroupCount);
  }
  return Boolean(state.transitionWitnessFailGroupsExpandedByRun[runKey]);
}

function setRejectFailGroupsExpanded(runPayload, expanded) {
  const runKey = transitionWitnessSummaryRunKey(runPayload);
  state.transitionWitnessFailGroupsExpandedByRun[runKey] = Boolean(expanded);
}

function toggleRejectFailGroups(runPayload, failGroupCount = 0) {
  setRejectFailGroupsExpanded(
    runPayload,
    !isRejectFailGroupsExpanded(runPayload, failGroupCount),
  );
  invalidatePanelCache("transitionStageWitnesses");
  renderActive();
}

function defaultRejectSplitsExpanded(splitCount = 0) {
  return (optionalInteger(splitCount) ?? 0) <= 4;
}

function isRejectSplitsExpanded(runPayload, splitCount = 0) {
  const runKey = transitionWitnessSummaryRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.transitionWitnessSplitsExpandedByRun, runKey)) {
    return defaultRejectSplitsExpanded(splitCount);
  }
  return Boolean(state.transitionWitnessSplitsExpandedByRun[runKey]);
}

function setRejectSplitsExpanded(runPayload, expanded) {
  const runKey = transitionWitnessSummaryRunKey(runPayload);
  state.transitionWitnessSplitsExpandedByRun[runKey] = Boolean(expanded);
}

function toggleRejectSplits(runPayload, splitCount = 0) {
  setRejectSplitsExpanded(
    runPayload,
    !isRejectSplitsExpanded(runPayload, splitCount),
  );
  invalidatePanelCache("transitionStageWitnesses");
  renderActive();
}

function programInspectorRunKey(runPayload) {
  return transitionSelectionRunKey(runPayload);
}

function isProgramInspectorOpen(runPayload) {
  const runKey = programInspectorRunKey(runPayload);
  return Boolean(state.programInspectorByRun[runKey]?.open);
}

function openProgramInspector(runPayload, nextState = null) {
  const runKey = programInspectorRunKey(runPayload);
  state.programInspectorByRun[runKey] = {
    ...(state.programInspectorByRun[runKey] || {}),
    ...(nextState && typeof nextState === "object" ? nextState : {}),
    open: true,
  };
}

function closeProgramInspector(runPayload) {
  const runKey = programInspectorRunKey(runPayload);
  state.programInspectorByRun[runKey] = {
    ...(state.programInspectorByRun[runKey] || {}),
    open: false,
  };
}

function openAcceptedProgramInspector(runPayload) {
  openProgramInspector(runPayload, {
    mode: "program",
    attemptIndex: 0,
    generatorIndex: 0,
    artifactKey: null,
  });
}

function openAttemptTraceInspector(runPayload) {
  openProgramInspector(runPayload, {
    mode: "attempt",
    attemptIndex: 0,
    generatorIndex: 0,
    artifactKey: "prompt",
  });
}

function updateProgramInspector(runPayload, updates) {
  const runKey = programInspectorRunKey(runPayload);
  state.programInspectorByRun[runKey] = {
    ...(state.programInspectorByRun[runKey] || {}),
    ...(updates && typeof updates === "object" ? updates : {}),
  };
}

function resolveCurrentAcceptedVersionKey(runPayload, gallery = null, fallbackVersionKey = null) {
  const metrics = runPayload?.dashboard?.metrics || {};
  return normalizeVersionKey(
    runPayload?.dashboard?.agent?.current_version_id
    || metrics.current_patch_current_version
    || metrics.current_patch_version
    || metrics.program_version
    || gallery?.live?.currentVersionKey
    || fallbackVersionKey,
  );
}

function resolveCurrentLiveVersion(runPayload, gallery, programVersions = null) {
  const versions = programVersions || normalizeProgramVersions(runPayload);
  const currentVersionKey = resolveCurrentAcceptedVersionKey(runPayload, gallery);
  return versions.versionsByKey.get(currentVersionKey)
    || gallery.versionsByKey.get(currentVersionKey)
    || versions.versions.find((version) => version.isCurrentExplainer)
    || versions.versions.find((version) => version.isCurrentProgram)
    || gallery.versions.find((version) => version.isCurrentExplainer)
    || gallery.versions.find((version) => version.isCurrentProgram)
    || versions.versions[0]
    || gallery.versions[0]
    || null;
}

function resolvePrimaryReviewId(version, gallery) {
  const introReviewId = String(version?.introReviewId || "").trim();
  if (introReviewId && gallery.reviewsById.has(introReviewId)) {
    return introReviewId;
  }
  const defaultReviewId = String(version?.defaultReview?.reviewId || "").trim();
  return defaultReviewId && gallery.reviewsById.has(defaultReviewId) ? defaultReviewId : null;
}

function resolvePrimaryReview(version, gallery) {
  const reviewId = resolvePrimaryReviewId(version, gallery);
  return reviewId ? gallery.reviewsById.get(reviewId) || null : null;
}

function invalidateTransitionPanels() {
  invalidatePanelCache(
    "transitionVersionSummary",
    "transitionVersionList",
    "transitionStageMode",
    "transitionStageToolbar",
    "transitionStageFrames",
    "transitionStageWitnesses",
  );
}

async function ensureTransitionBundleMetadata(bundleArtifact, revisionToken = null) {
  const bundleUrl = String(bundleArtifact?.url || "").trim();
  if (!bundleUrl) return null;
  const normalizedRevisionToken = String(revisionToken || "").trim() || null;
  const cachedRevisionToken = state.transitionBundleMetaRevisionByUrl[bundleUrl] || null;
  if (
    Object.prototype.hasOwnProperty.call(state.transitionBundleMetaByUrl, bundleUrl)
    && cachedRevisionToken === normalizedRevisionToken
  ) {
    return state.transitionBundleMetaByUrl[bundleUrl];
  }
  if (state.transitionBundleMetaRequestByUrl[bundleUrl] === normalizedRevisionToken) {
    return null;
  }
  state.transitionBundleMetaRequestByUrl[bundleUrl] = normalizedRevisionToken;
  try {
    const payload = await fetchJson(bundleUrl);
    if (state.transitionBundleMetaRequestByUrl[bundleUrl] === normalizedRevisionToken) {
      state.transitionBundleMetaByUrl[bundleUrl] = extractTransitionBundleMetadata(payload, bundleUrl);
      state.transitionBundleMetaRevisionByUrl[bundleUrl] = normalizedRevisionToken;
    }
  } catch (_error) {
    if (state.transitionBundleMetaRequestByUrl[bundleUrl] === normalizedRevisionToken) {
      state.transitionBundleMetaByUrl[bundleUrl] = null;
      state.transitionBundleMetaRevisionByUrl[bundleUrl] = normalizedRevisionToken;
    }
  } finally {
    if (state.transitionBundleMetaRequestByUrl[bundleUrl] === normalizedRevisionToken) {
      delete state.transitionBundleMetaRequestByUrl[bundleUrl];
      invalidatePanelCache("transitionStageFrames");
      invalidatePanelCache("transitionStageWitnesses");
      renderActive();
    }
  }
  return state.transitionBundleMetaByUrl[bundleUrl];
}

function coerceTransitionWitnessIndex(value) {
  const witnessIndex = optionalInteger(value);
  return witnessIndex !== null && witnessIndex >= 0 ? witnessIndex : null;
}

function ensureTransitionSelection(runPayload) {
  const runKey = transitionSelectionRunKey(runPayload);
  const gallery = normalizeTransitionGallery(runPayload);
  const programVersions = normalizeProgramVersions(runPayload);
  const existing = state.transitionSelectionByRun[runKey] || {
    mode: "live",
    railSelection: "live",
    versionKey: null,
    reviewId: null,
    failureId: null,
    activeVariant: "expected",
    focusMode: "target",
    witnessIndex: null,
  };
  const currentPatchVisible = isCurrentPatchVisible(runPayload, gallery);
  existing.focusMode = existing.focusMode === "witness" ? "witness" : "target";
  existing.witnessIndex = existing.focusMode === "witness"
    ? coerceTransitionWitnessIndex(existing.witnessIndex)
    : null;
  if (existing.mode === "live") {
    const currentVersion = resolveCurrentLiveVersion(runPayload, gallery, programVersions);
    Object.assign(existing, {
      mode: "live",
      railSelection: "live",
      versionKey: currentVersion?.versionKey || null,
      reviewId: resolvePrimaryReviewId(currentVersion, gallery),
      failureId: null,
      activeVariant: "expected",
      focusMode: "target",
      witnessIndex: null,
    });
  } else {
    const liveVersion = resolveCurrentLiveVersion(runPayload, gallery, programVersions);
    const liveReviewId = String(gallery.live?.defaultReview?.reviewId || "").trim() || null;
    const liveFailureId = String(gallery.live?.defaultReview?.failureId || "").trim() || null;
    let selectedVersion = programVersions.versionsByKey.get(normalizeVersionKey(existing.versionKey))
      || gallery.versionsByKey.get(normalizeVersionKey(existing.versionKey))
      || null;
    if (existing.railSelection === "current" && liveVersion) {
      selectedVersion = liveVersion;
    }
    if (!selectedVersion) {
      selectedVersion = liveVersion;
    }
    if (!selectedVersion) {
      if (existing.railSelection === "current" && currentPatchVisible) {
        existing.mode = "version";
        existing.railSelection = "current";
        existing.versionKey = null;
        existing.reviewId = liveReviewId;
        if (existing.activeVariant !== "expected" && existing.activeVariant !== "fail") {
          existing.activeVariant = "expected";
        }
        existing.failureId = existing.activeVariant === "fail" ? liveFailureId : null;
        existing.focusMode = "target";
        existing.witnessIndex = null;
      } else {
        existing.mode = "live";
        existing.railSelection = "live";
        existing.versionKey = null;
        existing.reviewId = null;
        existing.failureId = null;
        existing.activeVariant = "expected";
        existing.focusMode = "target";
        existing.witnessIndex = null;
      }
    } else {
      const primaryReviewId = resolvePrimaryReviewId(selectedVersion, gallery);
      const isCurrentPatchStillValid = Boolean(
        existing.railSelection === "current"
        && currentPatchVisible
        && liveVersion
        && selectedVersion.versionKey === liveVersion.versionKey
      );
      if (existing.railSelection === "current" && !isCurrentPatchStillValid) {
        existing.railSelection = "version";
      }
      existing.mode = "version";
      if (existing.railSelection !== "current" && existing.railSelection !== "version") {
        existing.railSelection = "version";
      }
      existing.versionKey = selectedVersion.versionKey;
      if (existing.activeVariant !== "expected" && existing.activeVariant !== "success" && existing.activeVariant !== "fail") {
        existing.activeVariant = "expected";
      }
      const resolvedReviewId = (
        existing.railSelection === "current" && currentPatchVisible
          ? liveReviewId
          : primaryReviewId
      );
      const review = resolvedReviewId ? gallery.reviewsById.get(resolvedReviewId) || null : null;
      const failOptions = Array.isArray(review?.fails) ? review.fails : [];
      const selectedFail = failOptions.find((fail) => fail.failureId === existing.failureId) || null;
      if (existing.activeVariant === "fail" && !selectedFail) {
        existing.activeVariant = "expected";
      }
      existing.reviewId = resolvedReviewId;
      existing.failureId = selectedFail?.failureId || null;
      if (existing.activeVariant === "success" && !review?.explained?.image?.url && !review?.explained?.bundle?.url) {
        existing.activeVariant = "expected";
      }
    }
  }

  if (existing.activeVariant !== "fail") {
    existing.focusMode = "target";
    existing.witnessIndex = null;
  } else {
    existing.focusMode = existing.focusMode === "witness" ? "witness" : "target";
    existing.witnessIndex = existing.focusMode === "witness"
      ? coerceTransitionWitnessIndex(existing.witnessIndex)
      : null;
  }

  state.transitionSelectionByRun[runKey] = existing;
  return existing;
}

function resolveTransitionSelection(runPayload) {
  const gallery = normalizeTransitionGallery(runPayload);
  const programVersions = normalizeProgramVersions(runPayload);
  const selection = ensureTransitionSelection(runPayload);
  const selectedVersion = programVersions.versionsByKey.get(normalizeVersionKey(selection.versionKey))
    || gallery.versionsByKey.get(normalizeVersionKey(selection.versionKey))
    || resolveCurrentLiveVersion(runPayload, gallery, programVersions)
    || programVersions.versions.find((version) => version.isCurrentExplainer)
    || programVersions.versions.find((version) => version.isCurrentProgram)
    || programVersions.versions[0]
    || gallery.versions[0]
    || null;
  const review = gallery.reviewsById.get(selection.reviewId)
    || resolvePrimaryReview(selectedVersion, gallery)
    || null;
  const failOptions = Array.isArray(review?.fails) ? review.fails : [];
  const selectedFail = failOptions.find((fail) => fail.failureId === selection.failureId) || null;
  const successArtifact = review?.explained?.image || null;
  const successBundle = review?.explained?.bundle || null;
  const reviewPrevious = review?.previous || null;
  const expected = review?.expected || null;
  if (selection.activeVariant === "success" && !successArtifact?.url && !successBundle?.url) {
    selection.activeVariant = "expected";
  }

  const metadataBundle = chooseTransitionMetadataBundle(review, selection, selectedFail);
  const bundleMeta = metadataBundle?.url
    ? state.transitionBundleMetaByUrl[String(metadataBundle.url).trim()] ?? null
    : null;
  const selectedWitnesses = Array.isArray(bundleMeta?.selectedWitnesses)
    ? bundleMeta.selectedWitnesses
    : [];
  const requestedWitnessIndex = coerceTransitionWitnessIndex(selection.witnessIndex);
  const focusedWitness = (
    selection.activeVariant === "fail"
    && selection.focusMode === "witness"
    && requestedWitnessIndex !== null
    && requestedWitnessIndex < selectedWitnesses.length
  )
    ? selectedWitnesses[requestedWitnessIndex]
    : null;
  const focusedWitnessBundle = focusedWitness?.bundle || null;
  const focusedWitnessBundleUrl = String(focusedWitnessBundle?.url || "").trim();
  const focusedWitnessBundleMeta = focusedWitnessBundleUrl
    ? state.transitionBundleMetaByUrl[focusedWitnessBundleUrl] ?? null
    : null;

  let previous = reviewPrevious;
  let nextArtifact = expected;
  let nextLabel = "Expected Next State";
  let nextTerminated = review?.expectedTerminated ?? null;
  let previousStatePayload = focusedWitnessBundleMeta?.previousStatePayload
    ?? bundleMeta?.previousStatePayload
    ?? null;
  let nextStatePayload = focusedWitnessBundleMeta?.expectedStatePayload
    ?? bundleMeta?.expectedStatePayload
    ?? null;
  if (selection.activeVariant === "success" && successArtifact?.url) {
    nextArtifact = successArtifact;
    nextLabel = "Predicted Explained";
    nextTerminated = review?.explained?.terminated ?? review?.expectedTerminated ?? null;
    nextStatePayload = focusedWitnessBundleMeta?.predictedStatePayload
      ?? bundleMeta?.predictedStatePayload
      ?? bundleMeta?.expectedStatePayload
      ?? nextStatePayload;
  } else if (selection.activeVariant === "success" && successBundle?.url) {
    nextLabel = "Predicted Explained";
    nextTerminated = review?.explained?.terminated ?? review?.expectedTerminated ?? null;
    nextStatePayload = focusedWitnessBundleMeta?.predictedStatePayload
      ?? bundleMeta?.predictedStatePayload
      ?? bundleMeta?.expectedStatePayload
      ?? nextStatePayload;
  } else if (selection.activeVariant === "fail") {
    nextArtifact = selectedFail?.image || null;
    const failLabel = formatFailureLabel(selectedFail);
    const failPhase = String(bundleMeta?.predictionErrorPhase || "").trim().toLowerCase();
    const emptyFailLabel = failPhase.includes("planning")
      ? "Planning Objective Failed"
      : "Prediction Failed";
    nextLabel = selectedFail?.image?.url
      ? `Predicted Unexplained · ${failLabel}`
      : `${emptyFailLabel} · ${failLabel}`;
    nextTerminated = selectedFail?.terminated ?? review?.expectedTerminated ?? null;
    nextStatePayload = selectedFail?.image?.url
      ? (
        focusedWitnessBundleMeta?.predictedStatePayload
        ?? bundleMeta?.predictedStatePayload
        ?? null
      )
      : null;
  }

  if (focusedWitness) {
    const focusedWitnessLabel = formatTransitionWitnessFocusLabel(focusedWitness, runPayload);
    const focusedWitnessLocationText = formatTransitionLocationText(focusedWitness);
    const focusedWitnessTitle = focusedWitnessLocationText
      ? `${focusedWitnessLabel} · ${focusedWitnessLocationText}`
      : focusedWitnessLabel;
    previous = focusedWitness.previous || previous;
    if (focusedWitness.predicted?.url) {
      nextArtifact = focusedWitness.predicted;
      nextLabel = `Witness Predicted · ${focusedWitnessTitle}`;
    } else if (focusedWitness.expected?.url) {
      nextArtifact = focusedWitness.expected;
      nextLabel = `Witness Expected · ${focusedWitnessTitle}`;
    } else {
      nextArtifact = focusedWitness.previous || nextArtifact;
      nextLabel = `Witness Preview · ${focusedWitnessTitle}`;
    }
    previousStatePayload = focusedWitnessBundleMeta?.previousStatePayload
      ?? previousStatePayload;
    nextStatePayload = focusedWitnessBundleMeta?.predictedStatePayload
      ?? focusedWitnessBundleMeta?.expectedStatePayload
      ?? focusedWitnessBundleMeta?.previousStatePayload
      ?? nextStatePayload;
    nextTerminated = focusedWitnessBundleMeta?.predictedTerminated
      ?? focusedWitnessBundleMeta?.expectedTerminated
      ?? focusedWitnessBundleMeta?.previousTerminated
      ?? nextTerminated;
  }

  const resolvedPreviousAction = focusedWitnessBundleMeta?.action
    ?? review?.action
    ?? bundleMeta?.action
    ?? null;
  const resolvedPreviousTerminated = focusedWitnessBundleMeta?.previousTerminated
    ?? review?.previousTerminated
    ?? bundleMeta?.previousTerminated
    ?? null;
  const resolvedNextTerminated = focusedWitness
    ? (
      nextTerminated
      ?? focusedWitnessBundleMeta?.predictedTerminated
      ?? focusedWitnessBundleMeta?.expectedTerminated
      ?? focusedWitnessBundleMeta?.previousTerminated
      ?? bundleMeta?.predictedTerminated
      ?? bundleMeta?.expectedTerminated
      ?? null
    )
    : (
      nextTerminated
      ?? bundleMeta?.predictedTerminated
      ?? bundleMeta?.expectedTerminated
      ?? null
    );

  return {
    gallery,
    selection,
    version: selectedVersion,
    review,
    failOptions,
    selectedFail,
    successArtifact,
    successBundle,
    previous,
    expected,
    bundleMeta,
    focusedWitness,
    focusedWitnessIndex: focusedWitness ? requestedWitnessIndex : null,
    focusedWitnessBundle,
    focusedWitnessBundleMeta,
    previousStatePayload,
    nextStatePayload,
    previousAction: resolvedPreviousAction,
    previousTerminated: resolvedPreviousTerminated,
    nextTerminated: resolvedNextTerminated,
    nextTone: transitionVariantTone(selection.activeVariant),
    nextArtifact,
    nextLabel,
    metadataBundle,
  };
}

function selectTransitionLive(runPayload) {
  const runKey = transitionSelectionRunKey(runPayload);
  state.transitionSelectionByRun[runKey] = {
    mode: "live",
    railSelection: "live",
    versionKey: null,
    reviewId: null,
    failureId: null,
    activeVariant: "expected",
    focusMode: "target",
    witnessIndex: null,
  };
  invalidateTransitionPanels();
  renderActive();
}

function selectTransitionLivePatch(runPayload) {
  const gallery = normalizeTransitionGallery(runPayload);
  const currentPatchVisible = isCurrentPatchVisible(runPayload, gallery);
  if (!currentPatchVisible) return;
  const currentPatchAvailable = isCurrentPatchAvailable(runPayload, gallery);
  const currentPatchActive = currentPatchAvailable && Boolean(runPayload?.isLive);
  const currentVersion = resolveCurrentLiveVersion(
    runPayload,
    gallery,
    normalizeProgramVersions(runPayload),
  );
  const liveReviewId = String(gallery.live?.defaultReview?.reviewId || "").trim();
  if (!liveReviewId) return;
  const review = gallery.reviewsById.get(liveReviewId) || null;
  if (!review) return;
  const runKey = transitionSelectionRunKey(runPayload);
  state.transitionSelectionByRun[runKey] = {
    mode: "version",
    railSelection: "current",
    versionKey: currentVersion?.versionKey || null,
    reviewId: liveReviewId,
    failureId: currentPatchActive ? null : String(gallery.live?.defaultReview?.failureId || "").trim() || null,
    activeVariant: currentPatchActive ? "expected" : "fail",
    focusMode: "target",
    witnessIndex: null,
  };
  invalidateTransitionPanels();
  renderActive();
}

function selectTransitionVersion(runPayload, versionKey) {
  const gallery = normalizeTransitionGallery(runPayload);
  const programVersions = normalizeProgramVersions(runPayload);
  const normalizedVersionKey = normalizeVersionKey(versionKey);
  const version = programVersions.versionsByKey.get(normalizedVersionKey)
    || gallery.versionsByKey.get(normalizedVersionKey)
    || null;
  if (!version) return;
  const runKey = transitionSelectionRunKey(runPayload);
  state.transitionSelectionByRun[runKey] = {
    mode: "version",
    railSelection: "version",
    versionKey: version.versionKey,
    reviewId: resolvePrimaryReviewId(version, gallery),
    failureId: null,
    activeVariant: "expected",
    focusMode: "target",
    witnessIndex: null,
  };
  invalidateTransitionPanels();
  renderActive();
}

function selectTransitionTarget(runPayload, variant, failureId = null) {
  const runKey = transitionSelectionRunKey(runPayload);
  const current = ensureTransitionSelection(runPayload);
  if (current.mode !== "version") return;
  const inspectorRunKey = programInspectorRunKey(runPayload);
  const inspectorState = state.programInspectorByRun[inspectorRunKey] || null;
  const resolved = resolveTransitionSelection(runPayload);
  if (variant === "fail") {
    const fail = resolved.failOptions.find((item) => item.failureId === String(failureId || "").trim()) || null;
    if (!fail) return;
    state.transitionSelectionByRun[runKey] = {
      ...current,
      failureId: fail.failureId,
      activeVariant: "fail",
      focusMode: "target",
      witnessIndex: null,
    };
    if (inspectorState?.open && inspectorState.mode !== "attempt") {
      closeProgramInspector(runPayload);
    }
  } else if (variant === "success") {
    if (!resolved.successArtifact?.url) return;
    state.transitionSelectionByRun[runKey] = {
      ...current,
      failureId: null,
      activeVariant: "success",
      focusMode: "target",
      witnessIndex: null,
    };
    if (inspectorState?.open && inspectorState.mode === "attempt") {
      closeProgramInspector(runPayload);
    }
  } else {
    state.transitionSelectionByRun[runKey] = {
      ...current,
      failureId: null,
      activeVariant: "expected",
      focusMode: "target",
      witnessIndex: null,
    };
    if (inspectorState?.open && inspectorState.mode === "attempt") {
      closeProgramInspector(runPayload);
    }
  }
  invalidateTransitionPanels();
  renderActive();
}

function selectTransitionWitness(runPayload, witnessIndex) {
  const runKey = transitionSelectionRunKey(runPayload);
  const current = ensureTransitionSelection(runPayload);
  if (current.mode !== "version" || current.activeVariant !== "fail") return;
  const resolved = resolveTransitionSelection(runPayload);
  const selectedWitnesses = Array.isArray(resolved.bundleMeta?.selectedWitnesses)
    ? resolved.bundleMeta.selectedWitnesses
    : [];
  const normalizedWitnessIndex = coerceTransitionWitnessIndex(witnessIndex);
  if (
    normalizedWitnessIndex === null
    || normalizedWitnessIndex >= selectedWitnesses.length
  ) {
    return;
  }
  const currentWitnessIndex = coerceTransitionWitnessIndex(current.witnessIndex);
  const isSameWitnessFocused = (
    current.focusMode === "witness"
    && currentWitnessIndex === normalizedWitnessIndex
  );
  state.transitionSelectionByRun[runKey] = {
    ...current,
    focusMode: isSameWitnessFocused ? "target" : "witness",
    witnessIndex: isSameWitnessFocused ? null : normalizedWitnessIndex,
  };
  invalidateTransitionPanels();
  renderActive();
}

function resetTransitionWitnessFocus(runPayload) {
  const runKey = transitionSelectionRunKey(runPayload);
  const current = ensureTransitionSelection(runPayload);
  if (current.mode !== "version" || current.activeVariant !== "fail") return;
  state.transitionSelectionByRun[runKey] = {
    ...current,
    focusMode: "target",
    witnessIndex: null,
  };
  invalidateTransitionPanels();
  renderActive();
}

function persistPollInterval(value) {
  try {
    window.localStorage.setItem(POLL_INTERVAL_STORAGE_KEY, String(value));
  } catch (_error) {
    // Ignore storage failures and keep the in-memory value for this session.
  }
}

function syncPollIntervalInputs() {
  if (elements.livePollInput) {
    if (document.activeElement === elements.livePollInput) return;
    elements.livePollInput.value = String(state.pollIntervalMs);
  }
}

function logScrollRunKey(runPayload) {
  return String(runPayload?.runId || "none");
}

function ensureLogScrollState(runKey) {
  const normalizedRunKey = String(runKey || "none");
  if (!Object.prototype.hasOwnProperty.call(state.logScrollStateByRun, normalizedRunKey)) {
    state.logScrollStateByRun[normalizedRunKey] = {
      autoFollow: true,
      distanceFromBottom: 0,
    };
  }
  return state.logScrollStateByRun[normalizedRunKey];
}

function logDistanceFromBottom(element) {
  const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
  return Math.max(0, maxScrollTop - element.scrollTop);
}

function rememberLogScrollState(runPayload = activeRun()) {
  if (!elements.log) return;
  const scrollState = ensureLogScrollState(logScrollRunKey(runPayload));
  const distanceFromBottom = logDistanceFromBottom(elements.log);
  scrollState.distanceFromBottom = distanceFromBottom;
  scrollState.autoFollow = distanceFromBottom <= LOG_AUTO_FOLLOW_THRESHOLD_PX;
}

function restoreLogScrollState(runPayload = activeRun()) {
  if (!elements.log) return;
  const scrollState = ensureLogScrollState(logScrollRunKey(runPayload));
  const maxScrollTop = Math.max(0, elements.log.scrollHeight - elements.log.clientHeight);
  if (scrollState.autoFollow) {
    elements.log.scrollTop = maxScrollTop;
    scrollState.distanceFromBottom = 0;
    return;
  }
  const nextScrollTop = clamp(maxScrollTop - scrollState.distanceFromBottom, 0, maxScrollTop);
  elements.log.scrollTop = nextScrollTop;
  scrollState.distanceFromBottom = Math.max(0, maxScrollTop - elements.log.scrollTop);
  scrollState.autoFollow = scrollState.distanceFromBottom <= LOG_AUTO_FOLLOW_THRESHOLD_PX;
}

function hostSizeSignature(element) {
  if (!element) return "0x0";
  const rect = element.getBoundingClientRect();
  return `${Math.round(rect.width)}x${Math.round(rect.height)}`;
}

function formatMetricValue(value) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "True" : "False";
  const parsed = optionalNumber(value);
  if (parsed === null) return String(value);
  if (Number.isInteger(parsed)) return String(parsed);
  const magnitude = Math.abs(parsed);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) {
    return parsed.toExponential(2);
  }
  if (magnitude >= 100) return parsed.toFixed(1);
  if (magnitude >= 1) return parsed.toFixed(3);
  return parsed.toFixed(4);
}

function formatMetricValueForKey(key, value) {
  const normalizedKey = String(key || "").trim().toLowerCase();
  if (normalizedKey.endsWith("_rate") || normalizedKey.endsWith("_accuracy")) {
    return formatPercent(value);
  }
  const formatted = formatMetricValue(value);
  return formatted;
}

function trimTrailingZeros(text) {
  return String(text)
    .replace(/(\.\d*?[1-9])0+$/u, "$1")
    .replace(/\.0+$/u, "");
}

function formatAxisTick(value) {
  const parsed = optionalNumber(value);
  if (parsed === null) return "-";
  const magnitude = Math.abs(parsed);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) return parsed.toExponential(1);
  if (magnitude >= 100) return trimTrailingZeros(parsed.toFixed(0));
  if (magnitude >= 10) return trimTrailingZeros(parsed.toFixed(1));
  if (magnitude >= 1) return trimTrailingZeros(parsed.toFixed(2));
  return trimTrailingZeros(parsed.toFixed(3));
}

function formatPercent(value) {
  const parsed = optionalNumber(value);
  if (parsed === null) return "-";
  return `${(clamp(parsed, 0, 1) * 100).toFixed(1)}%`;
}

function normalizeTerminalLog(text) {
  const source = String(text || "").replaceAll("\r\n", "\n");
  const outputLines = [];
  let currentLine = [];
  let cursor = 0;
  let index = 0;

  const ensureLength = (length) => {
    while (currentLine.length < length) {
      currentLine.push(" ");
    }
  };

  const writeChar = (char) => {
    ensureLength(cursor + 1);
    currentLine[cursor] = char;
    cursor += 1;
  };

  const clearToEndOfLine = () => {
    currentLine = currentLine.slice(0, cursor);
  };

  const clearFromStartOfLine = () => {
    ensureLength(cursor);
    for (let position = 0; position < cursor; position += 1) {
      currentLine[position] = " ";
    }
  };

  const clearEntireLine = () => {
    currentLine = [];
    cursor = 0;
  };

  const flushLine = () => {
    outputLines.push(currentLine.join("").replace(/\s+$/u, ""));
    currentLine = [];
    cursor = 0;
  };

  const consumeAnsiSequence = () => {
    if (source[index] !== "\x1b" || source[index + 1] !== "[") {
      return false;
    }
    let end = index + 2;
    while (end < source.length) {
      const code = source.charCodeAt(end);
      if (code >= 0x40 && code <= 0x7e) {
        break;
      }
      end += 1;
    }
    if (end >= source.length) {
      index = source.length;
      return true;
    }
    const finalChar = source[end];
    const parameterText = source.slice(index + 2, end);
    const params = parameterText.split(";").map((value) => {
      const parsed = Number.parseInt(value, 10);
      return Number.isFinite(parsed) ? parsed : null;
    });
    const first = params[0] ?? 0;

    if (finalChar === "K") {
      if (first === 1) {
        clearFromStartOfLine();
      } else if (first === 2) {
        clearEntireLine();
      } else {
        clearToEndOfLine();
      }
    } else if (finalChar === "D") {
      cursor = Math.max(0, cursor - Math.max(1, first || 1));
    } else if (finalChar === "C") {
      cursor = Math.max(0, cursor + Math.max(1, first || 1));
    } else if (finalChar === "G") {
      cursor = Math.max(0, Math.max(1, first || 1) - 1);
    }

    index = end + 1;
    return true;
  };

  while (index < source.length) {
    if (consumeAnsiSequence()) {
      continue;
    }
    const char = source[index];
    if (char === "\r") {
      cursor = 0;
    } else if (char === "\n") {
      flushLine();
    } else if (char === "\b") {
      cursor = Math.max(0, cursor - 1);
    } else if (char === "\t") {
      const nextTabStop = cursor + (4 - (cursor % 4 || 0));
      while (cursor < nextTabStop) {
        writeChar(" ");
      }
    } else if (char >= " ") {
      writeChar(char);
    }
    index += 1;
  }

  if (currentLine.length || source.endsWith("\r")) {
    outputLines.push(currentLine.join("").replace(/\s+$/u, ""));
  }
  return outputLines.join("\n");
}

function formatFieldLabel(key) {
  const lookup = String(key || "").trim().toLowerCase();
  return FIELD_LABELS[lookup] || String(key || "").trim().toUpperCase().replaceAll(" ", "_");
}

function formatAgentHeading(rawName) {
  const normalized = String(rawName || "").trim();
  if (!normalized) return "HUD Metrics";
  return normalized.replaceAll("_", " ").replaceAll("-", " ").toUpperCase();
}

function formatSeriesLabel(key) {
  const lookup = String(key || "").trim().toLowerCase();
  return SERIES_LABELS[lookup] || formatFieldLabel(key);
}

function normalizeRunStatus(run) {
  if (!run) return "idle";
  return run.effectiveStatus || run.status || "idle";
}

function isTerminalRunStatus(run) {
  const status = normalizeRunStatus(run);
  return status === "completed" || status === "failed" || status === "interrupted";
}

function statusClassName(status) {
  return `status-${String(status || "idle").toLowerCase().replace(/[^a-z0-9_-]+/g, "-")}`;
}

function formatHeartbeatAge(seconds) {
  if (!finiteNumber(seconds)) return "-";
  if (seconds < 1) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function classColor(classIndex) {
  const normalized = Number(classIndex || 0);
  const palette = ["#f59e0b", "#60a5fa", "#34d399", "#f472b6", "#a78bfa", "#fb7185", "#22d3ee", "#c084fc", "#f97316"];
  if (normalized <= 0) return "#94a3b8";
  return palette[(normalized - 1) % palette.length];
}

function hexColorToRgb(color) {
  const hex = String(color || "").trim().replace(/^#/, "");
  return {
    r: Number.parseInt(hex.slice(0, 2), 16),
    g: Number.parseInt(hex.slice(2, 4), 16),
    b: Number.parseInt(hex.slice(4, 6), 16),
  };
}

function blendRgb(left, right, mix) {
  const ratio = clamp(optionalNumber(mix) ?? 0, 0, 1);
  return {
    r: Math.round((left.r * (1 - ratio)) + (right.r * ratio)),
    g: Math.round((left.g * (1 - ratio)) + (right.g * ratio)),
    b: Math.round((left.b * (1 - ratio)) + (right.b * ratio)),
  };
}

function srgbChannelToLinear(channel) {
  const normalized = clamp(Number(channel || 0) / 255, 0, 1);
  if (normalized <= 0.04045) {
    return normalized / 12.92;
  }
  return ((normalized + 0.055) / 1.055) ** 2.4;
}

function rgbRelativeLuminance(rgb) {
  if (!rgb) return 0;
  const r = srgbChannelToLinear(rgb.r);
  const g = srgbChannelToLinear(rgb.g);
  const b = srgbChannelToLinear(rgb.b);
  return (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
}

function projectionShadowColor(color, baseOpacity) {
  const rgb = hexColorToRgb(color);
  const opacityBase = clamp(optionalNumber(baseOpacity) ?? 0.16, 0, 1);
  const softened = blendRgb(rgb, { r: 214, g: 220, b: 228 }, 0.18);
  const luminance = rgbRelativeLuminance(softened);
  const opacity = clamp(
    opacityBase * (1 - (luminance * 0.42)),
    opacityBase * 0.58,
    opacityBase,
  );
  return `rgba(${softened.r}, ${softened.g}, ${softened.b}, ${opacity.toFixed(3)})`;
}

function colorWithOpacity(color, opacity) {
  const rgb = hexColorToRgb(color);
  const resolvedOpacity = clamp(optionalNumber(opacity) ?? 1, 0, 1);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${resolvedOpacity.toFixed(3)})`;
}

function projectionFocusedTransitionFillOpacity(focusCount) {
  const normalizedCount = Math.max(1, optionalInteger(focusCount) ?? 1);
  const penalty = Math.min(0.1, Math.log2(normalizedCount) * 0.022);
  return clamp(0.2 - penalty, 0.11, 0.2);
}

function projectionFocusedTransitionBorderOpacity(focusCount) {
  const normalizedCount = Math.max(1, optionalInteger(focusCount) ?? 1);
  const penalty = Math.min(0.18, Math.log2(normalizedCount) * 0.038);
  return clamp(0.9 - penalty, 0.68, 0.9);
}

function projectionFocusedTransitionSymbolSize(focusCount) {
  const normalizedCount = Math.max(1, optionalInteger(focusCount) ?? 1);
  return clamp(11.4 - (Math.log2(normalizedCount) * 0.7), 8.6, 11.4);
}

function setQueryRunId(runId) {
  const url = new URL(window.location.href);
  if (runId) {
    url.searchParams.set("run_id", runId);
  } else {
    url.searchParams.delete("run_id");
  }
  window.history.replaceState(null, "", url);
}

function chooseActiveRun(payload) {
  const runs = payload.runs || [];
  if (state.activeRunId && runs.some((run) => run.runId === state.activeRunId)) {
    return state.activeRunId;
  }
  if (queryRunId && runs.some((run) => run.runId === queryRunId)) {
    return queryRunId;
  }
  const running = runs.find((run) => normalizeRunStatus(run) === "running");
  if (running) return running.runId;
  return payload.activeRunId || runs[0]?.runId || null;
}

function formatRunTabLabel(run) {
  const label = run.runOutputDir ? run.runOutputDir.split(/[\\/]/).pop() : String(run.runId || "").slice(0, 8);
  const status = normalizeRunStatus(run);
  return `${label} · ${status}`;
}

function formatRunTabTitle(run) {
  const status = normalizeRunStatus(run);
  return [
    `runId=${run.runId}`,
    `effective=${status}`,
    `raw=${run.rawStatus || run.status || "-"}`,
    `live=${run.isLive ? "yes" : "no"}`,
    `heartbeat=${formatHeartbeatAge(run.heartbeatAgeSec)}`,
    `reason=${run.statusReason || "-"}`,
  ].join("\n");
}

function resolveRunDeleteDisabledReason(run) {
  if (!run?.runId) {
    return "Unknown discovery run.";
  }
  if (run.isLive) {
    return "Live discovery runs cannot be deleted.";
  }
  return "";
}

function buildRunTabsSnapshot() {
  return state.runs.map((run) => ({
    runId: run.runId,
    text: formatRunTabLabel(run),
    title: formatRunTabTitle(run),
    active: run.runId === state.activeRunId,
    className: `button run-tab ${statusClassName(normalizeRunStatus(run))}`,
  }));
}

function updateTabs() {
  const snapshot = buildRunTabsSnapshot();
  const signature = stableSignature(snapshot);
  if (state.tabsSignature === signature) {
    return;
  }
  state.tabsSignature = signature;
  const existingEntries = new Map(
    Array.from(elements.tabs.querySelectorAll(".run-tab")).map((entry) => [entry.dataset.runId || "", entry]),
  );
  const desiredEntries = [];

  for (const run of state.runs) {
    let button = existingEntries.get(run.runId) || null;
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "button run-tab";
      button.addEventListener("click", async () => {
        try {
          const targetRunId = button?.dataset.runId || "";
          if (!targetRunId) return;
          await selectRunTab(targetRunId);
        } catch (error) {
          elements.meta.textContent = formatError(error);
        }
      });
    }
    const tabState = snapshot.find((entry) => entry.runId === run.runId);
    if (!tabState) continue;
    button.dataset.runId = run.runId;
    button.className = tabState.className;
    button.classList.toggle("active", tabState.active);
    if (button.textContent !== tabState.text) {
      button.textContent = tabState.text;
    }
    if (button.title !== tabState.title) {
      button.title = tabState.title;
    }
    desiredEntries.push(button);
    existingEntries.delete(run.runId);
  }

  for (const orphan of existingEntries.values()) {
    orphan.remove();
  }

  let cursor = elements.tabs.firstChild;
  for (const button of desiredEntries) {
    if (button === cursor) {
      cursor = cursor?.nextSibling || null;
      continue;
    }
    elements.tabs.insertBefore(button, cursor);
  }
}

async function startCompareSession() {
  if (state.offlineEval.modalBusy || !state.offlineEval.modalSource) return;
  const runId = String(state.offlineEval.modalRunId || state.activeRunPayload?.runId || "").trim();
  if (!runId) {
    elements.evalModalStatus.textContent = "No active discovery run is selected for compare launch.";
    return;
  }

  let compareWindow = null;
  state.offlineEval.modalBusy = true;
  try {
    compareWindow = window.open("", "_blank");
    if (compareWindow && compareWindow.document) {
      compareWindow.document.title = "Launching Compare";
      compareWindow.document.body.textContent = "Launching compare web UI...";
    }
  } catch (_error) {
    compareWindow = null;
  }

  elements.evalModalStatus.textContent = "Launching compare web UI...";
  try {
    const payload = await fetchJson(`/api/runs/${encodeURIComponent(runId)}/compare`, {
      method: "POST",
      body: JSON.stringify({
        versionKey: state.offlineEval.modalSource.versionKey,
        seed: 42,
      }),
    });
    const compareUrl = String(payload?.url || "/compare").trim() || "/compare";
    if (compareWindow) {
      compareWindow.location.replace(compareUrl);
    } else {
      window.location.assign(compareUrl);
    }
    closeOfflineEvalModal();
  } catch (error) {
    if (compareWindow && !compareWindow.closed) {
      compareWindow.close();
    }
    elements.evalModalStatus.textContent = formatError(error);
  } finally {
    state.offlineEval.modalBusy = false;
  }
}

function updateViewModeButton() {
  if (!elements.viewModeButton) return;
  if (isPausedView()) {
    elements.viewModeButton.textContent = "Resume Live";
    elements.viewModeButton.title = "Resume live dashboard updates";
  } else {
    const remainingSeconds = Math.ceil(liveViewRemainingMs() / 1000);
    elements.viewModeButton.textContent = `Pause View (${remainingSeconds}s)`;
    elements.viewModeButton.title = `Auto-pauses live dashboard updates in ${remainingSeconds} seconds`;
  }
  elements.viewModeButton.classList.toggle("button-primary", isPausedView());
}

function scheduleLiveViewAutoPause() {
  if (state.liveViewCountdownHandle) {
    window.clearTimeout(state.liveViewCountdownHandle);
    state.liveViewCountdownHandle = null;
  }
  if (isPausedView()) {
    updateViewModeButton();
    return;
  }
  const remainingMs = liveViewRemainingMs();
  updateViewModeButton();
  if (remainingMs <= 0) {
    pauseLiveView().catch((error) => {
      elements.meta.textContent = formatError(error);
    });
    return;
  }
  state.liveViewCountdownHandle = window.setTimeout(() => {
    state.liveViewCountdownHandle = null;
    scheduleLiveViewAutoPause();
  }, Math.min(LIVE_VIEW_COUNTDOWN_REFRESH_MS, remainingMs));
}

function updateProjectionModeButton() {
  if (!elements.projectionModeButton) return;
  const paused = isProjectionPaused();
  elements.projectionModeButton.textContent = paused ? "Resume T-SNE" : "Pause T-SNE";
  elements.projectionModeButton.classList.toggle("button-primary", paused);
  elements.projectionModeButton.title = paused
    ? "Resume live T-SNE updates for the projection panel"
    : "Pause live T-SNE updates while keeping the rest of the dashboard live";
  elements.projectionModeButton.setAttribute("aria-pressed", paused ? "true" : "false");
}

function updateProjectionSparseFilterInput() {
  if (!elements.projectionSparseFilterInput) return;
  elements.projectionSparseFilterInput.checked = isProjectionSparseFilterEnabled();
  elements.projectionSparseFilterInput.setAttribute(
    "aria-checked",
    isProjectionSparseFilterEnabled() ? "true" : "false",
  );
}

function scheduleShutdownReload(delayMs = SERVER_SHUTDOWN_RELOAD_DELAY_MS) {
  const normalizedDelayMs = Math.max(0, Number(delayMs) || 0);
  window.setTimeout(() => {
    window.location.reload();
  }, normalizedDelayMs);
}

function updateShutdownButton() {
  if (!elements.shutdownButton) return;
  const shutdownReady = Boolean(state.canShutdownServer);
  const shutdownBusy = Boolean(state.shutdownInFlight);
  const shutdownDisabled = shutdownBusy || !shutdownReady;
  elements.shutdownButton.disabled = false;
  elements.shutdownButton.classList.toggle("is-disabled", shutdownDisabled);
  elements.shutdownButton.textContent = shutdownBusy ? "Shutting Down..." : "Shutdown Server";
  elements.shutdownButton.classList.toggle("button-primary", shutdownReady && !shutdownBusy);
  elements.shutdownButton.setAttribute("aria-disabled", shutdownDisabled ? "true" : "false");
  if (shutdownDisabled) {
    elements.shutdownButton.setAttribute("tabindex", "-1");
  } else {
    elements.shutdownButton.removeAttribute("tabindex");
  }
  if (shutdownBusy) {
    elements.shutdownButton.title = "Shutdown has been requested for the detached dashboard server.";
    return;
  }
  if (shutdownReady) {
    elements.shutdownButton.title = "Terminate the detached dashboard server now that no discovery runs are live.";
    return;
  }
  elements.shutdownButton.title = "Live discovery runs still exist. Click to re-check, then shut down once all runs finish.";
}

async function sendViewerState(runId, viewMode, force = false) {
  const safeRunId = String(runId || "").trim();
  if (!safeRunId) return;
  const syncKey = `${safeRunId}:${viewMode}:${state.projectionMode}:${Number(isProjectionSparseFilterEnabled())}`;
  const now = Date.now();
  if (!force && state.lastViewerSyncKey === syncKey && (now - state.lastViewerSyncAt) < 1000) {
    return;
  }
  await fetchJson(`/api/runs/${encodeURIComponent(safeRunId)}/viewer-state`, {
    method: "POST",
    body: JSON.stringify({
      clientId: state.viewerId,
      viewMode,
      projectionMode: state.projectionMode,
      projectionExcludeSparseClasses: isProjectionSparseFilterEnabled(),
    }),
  });
  state.lastViewerSyncAt = now;
  state.lastViewerSyncKey = syncKey;
}

function setPausedViewMode() {
  state.viewMode = "paused";
  if (state.pollHandle) {
    window.clearTimeout(state.pollHandle);
    state.pollHandle = null;
  }
  updateViewModeButton();
  scheduleLiveViewAutoPause();
}

async function selectRunTab(targetRunId) {
  const safeTargetRunId = String(targetRunId || "").trim();
  if (!safeTargetRunId) return;
  const previousRunId = state.activeRunId;
  setPausedViewMode();
  if (previousRunId && previousRunId !== safeTargetRunId) {
    await sendViewerState(previousRunId, "paused", true);
  }
  await sendViewerState(safeTargetRunId, "paused", true);
  state.activeRunId = safeTargetRunId;
  setQueryRunId(safeTargetRunId);
  state.activeRunPayload = await fetchJson(`/api/runs/${encodeURIComponent(safeTargetRunId)}`);
  updateTabs();
  renderActive();
  scheduleNextPoll(livePollIntervalMs());
}

async function toggleViewMode() {
  if (isPausedView()) {
    state.viewMode = "live";
    state.liveViewStartedAtMs = Date.now();
    updateViewModeButton();
    scheduleLiveViewAutoPause();
    await sendViewerState(state.activeRunId, state.viewMode, true);
    await pollRuns();
    return;
  }
  await pauseLiveView();
}

async function pauseLiveView() {
  if (isPausedView()) return;
  setPausedViewMode();
  await sendViewerState(state.activeRunId, state.viewMode, true);
  renderActive();
  scheduleNextPoll(livePollIntervalMs());
}

async function toggleProjectionMode() {
  state.projectionMode = isProjectionPaused() ? "live" : "paused";
  updateProjectionModeButton();
  await sendViewerState(state.activeRunId, state.viewMode, true);
  renderActive();
  scheduleNextPoll(livePollIntervalMs());
}

async function toggleProjectionSparseFilter() {
  state.projectionExcludeSparseClasses = Boolean(elements.projectionSparseFilterInput?.checked);
  updateProjectionSparseFilterInput();
  await sendViewerState(state.activeRunId, state.viewMode, true);
  renderActive();
  scheduleNextPoll(livePollIntervalMs());
}

async function waitForServerShutdownAndReload() {
  elements.meta.textContent = "Shutdown requested. Waiting for the dashboard server to stop responding...";
  await waitMs(SERVER_SHUTDOWN_HEALTHCHECK_INITIAL_DELAY_MS);
  const deadlineMs = Date.now() + SERVER_SHUTDOWN_HEALTHCHECK_TIMEOUT_MS;
  while (Date.now() < deadlineMs) {
    try {
      await fetchJson(`/api/health?shutdownCheck=${Date.now()}`);
    } catch (_error) {
      const reloadDelaySec = Math.max(1, Math.ceil(SERVER_SHUTDOWN_RELOAD_DELAY_MS / 1000));
      elements.meta.textContent = `Dashboard server is no longer responding. Reloading in ${reloadDelaySec} seconds...`;
      scheduleShutdownReload();
      return;
    }
    await waitMs(SERVER_SHUTDOWN_HEALTHCHECK_INTERVAL_MS);
  }
  elements.meta.textContent = "Shutdown requested. Reloading to confirm whether the dashboard server has stopped...";
  scheduleShutdownReload();
}

async function requestServerShutdown() {
  if (state.shutdownInFlight) return;
  if (!state.canShutdownServer) return;
  await loadRuns();
  if (!state.canShutdownServer) {
    updateShutdownButton();
    elements.meta.textContent = "Discovery runs are still live. Wait for them to finish before shutting down the dashboard server.";
    return;
  }
  const confirmed = window.confirm(
    "Shut down the detached discovery dashboard server now?\n\nYou will need to launch discovery again to bring the dashboard back.",
  );
  if (!confirmed) return;
  state.shutdownInFlight = true;
  setPausedViewMode();
  updateShutdownButton();
  await fetchJson("/api/server/shutdown", {
    method: "POST",
  });
  await waitForServerShutdownAndReload();
}

function resolveRunDeleteTarget(runPayload = null) {
  const targetRunId = String(state.runDelete.runId || "").trim();
  if (!targetRunId) {
    return null;
  }
  if (runPayload && String(runPayload?.runId || "").trim() === targetRunId) {
    return runPayload;
  }
  if (
    state.activeRunPayload
    && String(state.activeRunPayload?.runId || "").trim() === targetRunId
  ) {
    return state.activeRunPayload;
  }
  return state.runs.find((candidate) => String(candidate?.runId || "").trim() === targetRunId) || null;
}

function runDeleteReady() {
  return state.runDelete.inputText === RUN_DELETE_CONFIRMATION_TEXT;
}

function closeRunDeleteModal() {
  state.runDelete.open = false;
  state.runDelete.busy = false;
  state.runDelete.runId = null;
  state.runDelete.inputText = "";
  state.runDelete.notice = "";
  if (elements.runDeleteInput && elements.runDeleteInput.value !== "") {
    elements.runDeleteInput.value = "";
  }
  if (elements.runDeleteModal) {
    elements.runDeleteModal.hidden = true;
  }
}

function renderRunDeleteModal(runPayload = null) {
  if (
    !elements.runDeleteModal
    || !elements.runDeleteSubtitle
    || !elements.runDeletePhrase
    || !elements.runDeleteInput
    || !elements.runDeleteStatus
    || !elements.runDeleteCancel
    || !elements.runDeleteConfirm
  ) {
    return;
  }
  const targetRun = resolveRunDeleteTarget(runPayload);
  if (!state.runDelete.open || !targetRun) {
    if (state.runDelete.open) {
      closeRunDeleteModal();
      return;
    }
    elements.runDeleteModal.hidden = true;
    return;
  }
  const runIdLabel = String(targetRun?.runId || "").trim();
  const runOutputLabel = String(targetRun?.runOutputDir || "").trim().split(/[\\/]/).pop() || "";
  const subtitle = runIdLabel && runOutputLabel && runIdLabel !== runOutputLabel
    ? `${runIdLabel} · ${runOutputLabel}`
    : (runIdLabel || runOutputLabel || "discovery run");
  const disabledReason = resolveRunDeleteDisabledReason(targetRun);
  const statusText = state.runDelete.notice
    || disabledReason
    || `Type exactly: ${RUN_DELETE_CONFIRMATION_TEXT}`;
  elements.runDeleteModal.hidden = false;
  elements.runDeleteSubtitle.textContent = subtitle;
  elements.runDeletePhrase.textContent = RUN_DELETE_CONFIRMATION_TEXT;
  if (elements.runDeleteInput.value !== state.runDelete.inputText) {
    elements.runDeleteInput.value = state.runDelete.inputText;
  }
  elements.runDeleteStatus.textContent = statusText;
  elements.runDeleteInput.disabled = state.runDelete.busy;
  elements.runDeleteCancel.disabled = state.runDelete.busy;
  elements.runDeleteConfirm.disabled = (
    state.runDelete.busy
    || Boolean(disabledReason)
    || !runDeleteReady()
  );
  elements.runDeleteConfirm.textContent = state.runDelete.busy ? "Deleting..." : "Delete Run";
}

function openRunDeleteModal(run) {
  const runId = String(run?.runId || "").trim();
  if (!runId) {
    elements.meta.textContent = "Unknown discovery run.";
    return;
  }
  state.runDelete.open = true;
  state.runDelete.busy = false;
  state.runDelete.runId = runId;
  state.runDelete.inputText = "";
  state.runDelete.notice = "";
  renderRunDeleteModal(run);
  window.requestAnimationFrame(() => {
    elements.runDeleteInput?.focus();
  });
}

async function submitRunDelete() {
  if (state.runDelete.busy) return;
  const targetRun = resolveRunDeleteTarget();
  if (!targetRun) {
    closeRunDeleteModal();
    return;
  }
  const disabledReason = resolveRunDeleteDisabledReason(targetRun);
  if (disabledReason) {
    state.runDelete.notice = disabledReason;
    renderRunDeleteModal(targetRun);
    return;
  }
  if (!runDeleteReady()) {
    state.runDelete.notice = `Type exactly: ${RUN_DELETE_CONFIRMATION_TEXT}`;
    renderRunDeleteModal(targetRun);
    return;
  }
  state.runDelete.busy = true;
  state.runDelete.notice = `${targetRun.runId} deleting...`;
  renderRunDeleteModal(targetRun);
  try {
    const payload = await fetchJson(
      `/api/runs/${encodeURIComponent(targetRun.runId)}/delete`,
      {
        method: "POST",
        body: JSON.stringify({
          confirmText: state.runDelete.inputText,
        }),
      },
    );
    const deletedRunId = String(payload?.deleted?.runId || targetRun.runId || "").trim() || "discovery run";
    const deletedDirectories = Array.isArray(payload?.deleted?.deletedDirectories)
      ? payload.deleted.deletedDirectories.length
      : 0;
    await loadRuns();
    closeRunDeleteModal();
    invalidateTransitionPanels();
    renderActive();
    elements.meta.textContent = `${deletedRunId} deleted (${deletedDirectories} dirs).`;
  } catch (error) {
    state.runDelete.busy = false;
    state.runDelete.notice = formatError(error);
    renderRunDeleteModal(targetRun);
  }
}

function livePollIntervalMs() {
  return state.pollIntervalMs;
}

function scheduleNextPoll(delayMs = livePollIntervalMs()) {
  if (state.pollInFlight) return;
  if (state.pollHandle) {
    window.clearTimeout(state.pollHandle);
  }
  if (isPausedView() || state.pollIntervalEditing) {
    state.pollHandle = null;
    return;
  }
  state.pollHandle = window.setTimeout(() => {
    state.pollHandle = null;
    pollRuns().catch((error) => {
      elements.meta.textContent = formatError(error);
    });
  }, delayMs);
}

function updatePollInterval(rawValue) {
  const nextValue = sanitizePollIntervalMs(rawValue, state.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
  state.pollIntervalMs = nextValue;
  persistPollInterval(nextValue);
  syncPollIntervalInputs();
  scheduleNextPoll(livePollIntervalMs());
}

function deriveIntrinsicCurrentPanelSpec(panelSpec) {
  const rightKeys = Array.isArray(panelSpec?.right_keys) ? panelSpec.right_keys : [];
  return {
    enabled: panelSpec?.enabled !== false,
    title: String(panelSpec?.current_title || "Intrinsic Current Reward"),
    left_keys: rightKeys,
    left_label: String(panelSpec?.current_label || "current transition reward"),
    left_ylabel: String(panelSpec?.right_ylabel || panelSpec?.right_label || panelSpec?.current_label || "current transition reward"),
    left_palette: ["#f97316", "#22c55e", "#a855f7"],
    left_smoothing: panelSpec?.right_smoothing,
    left_axis_name_gap: 48,
    host_height: 392,
  };
}

function deriveIntrinsicMeanPanelSpec(panelSpec) {
  const leftKeys = Array.isArray(panelSpec?.left_keys) ? panelSpec.left_keys : [];
  const shadeKeys = Array.isArray(panelSpec?.shade_keys) ? panelSpec.shade_keys : [];
  return {
    enabled: panelSpec?.enabled !== false,
    title: String(panelSpec?.mean_title || "Intrinsic Mean Reward"),
    left_keys: leftKeys,
    left_label: String(panelSpec?.mean_label || panelSpec?.left_label || "train batch mean reward"),
    left_ylabel: String(panelSpec?.left_ylabel || panelSpec?.mean_label || panelSpec?.left_label || "train batch mean reward"),
    left_palette: ["#f97316", "#22c55e", "#a855f7"],
    left_smoothing: panelSpec?.left_smoothing,
    shade_keys: shadeKeys,
    shade_color: panelSpec?.shade_color || "#fb923c",
    shade_smoothing: panelSpec?.shade_smoothing,
    left_axis_name_gap: 48,
    host_height: 392,
  };
}

function clearDashboardPanels(emptyMessage = "dashboard unavailable") {
  resetDashboardPanelVisibility();
  renderBoard(null, elements.board, { emptyMessage });
  if (elements.targetWorldsHead) {
    elements.targetWorldsHead.classList.remove("is-open");
    elements.targetWorldsHead.setAttribute("aria-expanded", "false");
  }
  setElementHidden(elements.targetWorldsTitle, true);
  if (elements.targetWorldsTitle) {
    elements.targetWorldsTitle.textContent = "Target Maps";
  }
  setElementHidden(elements.targetWorlds, true);
  setElementHidden(elements.targetWorldBody, true);
  if (elements.targetWorldList) {
    elements.targetWorldList.innerHTML = "";
  }
  if (elements.targetWorldMeta) {
    elements.targetWorldMeta.textContent = "";
  }
  if (elements.targetWorldBoard) {
    renderBoard(null, elements.targetWorldBoard, { emptyMessage });
  }
  elements.hud.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  if (elements.projectionSummary) {
    elements.projectionSummary.textContent = "";
  }
  if (elements.prototypeLegend) {
    elements.prototypeLegend.innerHTML = "";
  }
  if (elements.projectionTargetWorldList) {
    elements.projectionTargetWorldList.innerHTML = "";
  }
  if (elements.projectionTargetWorldsHead) {
    elements.projectionTargetWorldsHead.classList.remove("is-open");
    elements.projectionTargetWorldsHead.setAttribute("aria-expanded", "false");
  }
  if (elements.projectionTargetWorldsTitle) {
    elements.projectionTargetWorldsTitle.textContent = "Target Worlds";
  }
  setElementHidden(elements.projectionTargetWorldBody, true);
  setElementHidden(elements.projectionTargetWorlds, true);
  if (elements.probabilitiesSummary) {
    elements.probabilitiesSummary.textContent = "";
  }
  elements.topLeftChart.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  elements.middleLeftChart.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  elements.bottomLeftChart.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  elements.projection.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  elements.probabilities.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
  elements.classRows.innerHTML = `<div class="chart-empty">${escapeHtml(emptyMessage)}</div>`;
}

function applyDashboardLayout(dashboardPayload, panelEnabled = resolveDashboardPanelEnabled(dashboardPayload)) {
  if (!elements.dashboardComposite || !elements.dashboardLeftStack || !elements.dashboardRightStack) {
    return;
  }

  const compositeWidth = Math.max(
    0,
    Number(elements.dashboardComposite.clientWidth || elements.dashboardComposite.getBoundingClientRect().width || 0),
  );
  const columnMinWidth = 640;
  const columnGap = 22;
  const compactViewport = compositeWidth > 0
    ? compositeWidth < (columnMinWidth * 2 + columnGap)
    : window.innerWidth <= 1500;

  elements.dashboardLeftStack.style.gridTemplateRows = "";
  elements.dashboardRightStack.style.gridTemplateRows = "";

  if (compactViewport || !panelEnabled.left_column || !panelEnabled.right_column) {
    elements.dashboardComposite.style.gridTemplateColumns = "minmax(0, 1fr)";
    return;
  }

  elements.dashboardComposite.style.gridTemplateColumns = "minmax(0, 1fr) minmax(0, 1fr)";
}

function resolveHudChip(chip, metrics) {
  const candidates = Array.isArray(chip) ? chip : [chip];
  for (const candidate of candidates) {
    const key = String(candidate || "").trim();
    if (!key || key.toLowerCase() === "rtotal_std") continue;
    const value = metrics?.[key];
    if (value !== null && value !== undefined && value !== "") {
      return { key, value };
    }
  }
  const fallbackKey = String(candidates[0] || "").trim();
  if (!fallbackKey || fallbackKey.toLowerCase() === "rtotal_std") return null;
  return fallbackKey ? { key: fallbackKey, value: metrics?.[fallbackKey] } : null;
}

function isStandaloneHudHeader(line) {
  const normalized = String(line || "").trim();
  if (!normalized) return false;
  if (normalized.includes(":")) return false;
  if (/[0-9]/.test(normalized)) return false;
  return /^[A-Z][A-Z\s/_&-]+$/u.test(normalized);
}

function parseHudInlineItem(token) {
  const normalized = String(token || "").trim();
  if (!normalized) return null;
  const colonIndex = normalized.indexOf(":");
  if (colonIndex > 0) {
    const label = normalized.slice(0, colonIndex).trim();
    const value = normalized.slice(colonIndex + 1).trim();
    if (label && value) {
      return { label, value };
    }
  }
  const firstSpaceIndex = normalized.search(/\s/u);
  if (firstSpaceIndex > 0) {
    const label = normalized.slice(0, firstSpaceIndex).trim();
    const value = normalized.slice(firstSpaceIndex + 1).trim();
    if (label && value) {
      return { label, value };
    }
  }
  return { label: null, value: normalized };
}

function parseHudRow(line) {
  const normalized = String(line || "").trim();
  if (!normalized) {
    return { label: null, items: [], rawText: "" };
  }

  let rowLabel = null;
  let content = normalized;
  const colonIndex = normalized.indexOf(":");
  if (colonIndex > 0) {
    rowLabel = normalized.slice(0, colonIndex).trim();
    content = normalized.slice(colonIndex + 1).trim();
  }

  const tokens = content.split(/\s{3,}/u).map((token) => token.trim()).filter(Boolean);
  if (!tokens.length) {
    return { label: rowLabel, items: [], rawText: normalized };
  }
  const items = tokens.map(parseHudInlineItem).filter(Boolean);
  if (!items.length) {
    return { label: rowLabel, items: [], rawText: normalized };
  }
  return {
    label: rowLabel,
    items,
    rawText: items.length ? "" : normalized,
  };
}

function parseStructuredHudText(text) {
  const lines = String(text || "").split(/\r?\n/u).map((line) => line.trimRight());
  const summaryRows = [];
  const sections = [];
  const rawLines = [];
  let currentSection = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    if (isStandaloneHudHeader(line)) {
      currentSection = { title: line, rows: [] };
      sections.push(currentSection);
      continue;
    }

    const row = parseHudRow(line);
    if (currentSection) {
      currentSection.rows.push(row);
      continue;
    }

    if (row.items.length && !row.label) {
      summaryRows.push(row);
      continue;
    }

    if (row.items.length && row.label) {
      sections.push({
        title: row.label,
        rows: [{ label: null, items: row.items, rawText: "" }],
      });
      continue;
    }

    rawLines.push(line);
  }

  return {
    summaryRows,
    sections: sections.filter((section) => Array.isArray(section.rows) && section.rows.length),
    rawLines,
  };
}

function normalizeSummaryRows(summaryRows) {
  if (!Array.isArray(summaryRows) || !summaryRows.length) {
    return [];
  }

  const normalizedRows = [];
  for (const row of summaryRows) {
    const items = Array.isArray(row?.items) ? row.items.filter(Boolean) : [];
    if (!items.length) {
      continue;
    }

    const contextItems = [];
    const detailItems = [];
    for (const item of items) {
      const label = String(item?.label || "").trim().toLowerCase();
      if (label === "world" || label === "map") {
        contextItems.push(item);
      } else {
        detailItems.push(item);
      }
    }

    if (contextItems.length && detailItems.length) {
      normalizedRows.push({ label: row?.label || null, items: contextItems, rawText: "" });
      normalizedRows.push({ label: null, items: detailItems, rawText: "" });
      continue;
    }

    normalizedRows.push({ label: row?.label || null, items, rawText: row?.rawText || "" });
  }

  return normalizedRows;
}

function createHudItemChip(item) {
  const chip = document.createElement("div");
  chip.className = "hud-inline-chip";

  if (item.label) {
    const label = document.createElement("span");
    label.className = "hud-inline-chip-label";
    label.textContent = item.label;
    chip.append(label);
  }

  const value = document.createElement("span");
  value.className = "hud-inline-chip-value";
  value.textContent = item.value;
  chip.append(value);
  return chip;
}

function renderStructuredHudText(hudText) {
  const parsed = parseStructuredHudText(hudText);
  const summaryRows = normalizeSummaryRows(parsed.summaryRows);
  const hasStructuredRows = summaryRows.length || parsed.sections.length;
  if (!hasStructuredRows) {
    return false;
  }

  if (summaryRows.length) {
    const summary = document.createElement("section");
    summary.className = "hud-metric-section hud-metric-summary-section";
    const stack = document.createElement("div");
    stack.className = "hud-summary-stack";
    for (const row of summaryRows) {
      const strip = document.createElement("div");
      strip.className = "hud-summary-strip";
      for (const item of row.items) {
        strip.append(createHudItemChip(item));
      }
      stack.append(strip);
    }
    summary.append(stack);
    elements.hud.append(summary);
  }

  for (const section of parsed.sections) {
    const host = document.createElement("section");
    host.className = "hud-metric-section";

    if (section.title) {
      const header = document.createElement("div");
      header.className = "hud-metric-section-title";
      header.textContent = section.title;
      host.append(header);
    }

    const rows = document.createElement("div");
    rows.className = "hud-detail-stack";
    for (const row of section.rows) {
      const entry = document.createElement("div");
      entry.className = "hud-detail-row";

      if (row.label) {
        const label = document.createElement("div");
        label.className = "hud-detail-label";
        label.textContent = row.label;
        entry.append(label);
      }

      const content = document.createElement("div");
      content.className = row.items.length ? "hud-detail-chip-wrap" : "hud-detail-raw";
      if (row.items.length) {
        for (const item of row.items) {
          content.append(createHudItemChip(item));
        }
      } else {
        content.textContent = row.rawText || "";
      }
      entry.append(content);
      rows.append(entry);
    }
    host.append(rows);
    elements.hud.append(host);
  }

  if (parsed.rawLines.length) {
    const fallback = document.createElement("section");
    fallback.className = "hud-metric-section hud-metric-section-text";
    const body = document.createElement("pre");
    body.className = "hud-metric-text";
    body.textContent = parsed.rawLines.join("\n");
    fallback.append(body);
    elements.hud.append(fallback);
  }

  return true;
}

function renderHudSections(dashboardPayload) {
  elements.hud.innerHTML = "";
  const hudSnapshot = dashboardPayload?.hud || {};
  const hudText = String(hudSnapshot.text || "").trim();
  if (hudText) {
    if (renderStructuredHudText(hudText)) {
      return;
    }
    const host = document.createElement("section");
    host.className = "hud-metric-section hud-metric-section-text";

    const body = document.createElement("pre");
    body.className = "hud-metric-text";
    body.textContent = hudText;
    host.append(body);

    elements.hud.append(host);
    return;
  }

  const display = dashboardPayload?.display || {};
  const metrics = dashboardPayload?.metrics || {};
  const sections = Array.isArray(display.hud_sections) ? display.hud_sections : [];
  if (!sections.length) {
    elements.hud.innerHTML = `<div class="chart-empty">no HUD sections</div>`;
    return;
  }

  for (const section of sections) {
    const chips = Array.isArray(section?.chips) ? section.chips : [];
    const resolved = chips.map((chip) => resolveHudChip(chip, metrics)).filter(Boolean);
    if (!resolved.length) continue;

    const host = document.createElement("section");
    host.className = "hud-metric-section";
    const title = String(section?.title || "").trim();
    if (title) {
      const header = document.createElement("div");
      header.className = "hud-metric-section-title";
      header.textContent = title;
      host.append(header);
    }

    const grid = document.createElement("div");
    grid.className = "hud-metric-grid";
    for (const chip of resolved) {
      const row = document.createElement("div");
      row.className = "hud-metric-item";
      row.innerHTML = `
        <span class="hud-metric-label">${escapeHtml(formatFieldLabel(chip.key))}</span>
        <span class="hud-metric-value">${escapeHtml(formatMetricValueForKey(chip.key, chip.value))}</span>
      `;
      grid.append(row);
    }
    host.append(grid);
    elements.hud.append(host);
  }

  if (!elements.hud.childElementCount) {
    elements.hud.innerHTML = `<div class="chart-empty">no HUD sections</div>`;
  }
}

function computeRange(values, padding = 0.12) {
  const bounds = computeFiniteBounds(values);
  if (!bounds) return { min: 0, max: 1, span: 1 };
  let { min, max } = bounds;
  if (Math.abs(max - min) < 1e-8) {
    const delta = Math.max(1, Math.abs(max || min || 1)) * 0.5;
    min -= delta;
    max += delta;
  } else {
    const pad = (max - min) * padding;
    min -= pad;
    max += pad;
  }
  return { min, max, span: Math.max(1e-8, max - min) };
}

function computeFiniteBounds(values) {
  let min = null;
  let max = null;
  for (const value of values) {
    if (!finiteNumber(value)) continue;
    if (min === null || value < min) min = value;
    if (max === null || value > max) max = value;
  }
  if (min === null || max === null) return null;
  return { min, max };
}

function emaSmooth(values, alpha = 0.2) {
  const output = [];
  let last = null;
  for (const raw of values) {
    const current = optionalNumber(raw);
    if (current === null) {
      output.push(last);
      continue;
    }
    last = last === null ? current : ((alpha * current) + ((1 - alpha) * last));
    output.push(last);
  }
  return output;
}

function shouldSmoothSeries(panelSpec, key, fallback = true) {
  return typeof panelSpec?.[key] === "boolean" ? panelSpec[key] : fallback;
}

function buildPanelSeries(panelSpec, history) {
  const steps = Array.isArray(history?.steps) ? history.steps.map((value, index) => optionalNumber(value) ?? index) : [];
  const metrics = history?.metrics || {};
  const length = steps.length;
  const leftKeys = Array.isArray(panelSpec?.left_keys) ? panelSpec.left_keys : [];
  const rightKeys = Array.isArray(panelSpec?.right_keys) ? panelSpec.right_keys : [];
  const shadeKeys = Array.isArray(panelSpec?.shade_keys) ? panelSpec.shade_keys : [];
  const leftPalette = Array.isArray(panelSpec?.left_palette) && panelSpec.left_palette.length
    ? panelSpec.left_palette.map((value) => String(value))
    : (leftKeys.length > 1 ? ["#b45309", "#d97706", "#f59e0b"] : [panelSpec?.left_color || "#0f766e"]);
  const rightPalette = Array.isArray(panelSpec?.right_palette) && panelSpec.right_palette.length
    ? panelSpec.right_palette.map((value) => String(value))
    : (rightKeys.length > 1 ? ["#1d4ed8", "#2563eb", "#60a5fa"] : [panelSpec?.right_color || "#2563eb"]);

  const buildGroup = (keys, palette, group, smoothKey) => keys.map((key, index) => {
    const rawValues = [];
    const seriesSource = Array.isArray(metrics?.[key]) ? metrics[key] : [];
    for (let i = 0; i < length; i += 1) {
      rawValues.push(optionalNumber(seriesSource[i]));
    }
    const smooth = shouldSmoothSeries(panelSpec, smoothKey, true);
    const values = smooth ? emaSmooth(rawValues, 0.2) : rawValues.slice();
    const singleLabel = group === "left" ? panelSpec?.left_label : panelSpec?.right_label;
    return {
      key: String(key),
      label: keys.length === 1 && singleLabel ? String(singleLabel) : formatSeriesLabel(key),
      group,
      color: palette[Math.min(index, palette.length - 1)],
      rawValues,
      values,
      showRaw: smooth,
      strokeWidth: group === "left" ? (index === 0 ? 2.4 : 1.9) : (index === 0 ? 2.1 : 1.7),
    };
  });

  const left = buildGroup(leftKeys, leftPalette, "left", "left_smoothing");
  const right = buildGroup(rightKeys, rightPalette, "right", "right_smoothing");
  const shade = shadeKeys.map((key) => {
    const rawValues = [];
    const seriesSource = Array.isArray(metrics?.[key]) ? metrics[key] : [];
    for (let i = 0; i < length; i += 1) {
      rawValues.push(optionalNumber(seriesSource[i]));
    }
    return {
      key: String(key),
      label: formatFieldLabel(key),
      group: "shade",
      color: panelSpec?.shade_color || "#f59e0b",
      values: shouldSmoothSeries(panelSpec, "shade_smoothing", true) ? emaSmooth(rawValues, 0.2) : rawValues.slice(),
    };
  });

  return { steps, left, right, shade };
}

function axisTicks(range) {
  return [0, 0.25, 0.5, 0.75, 1].map((ratio) => ({
    ratio,
    value: range.max - (range.span * ratio),
  }));
}

function xTicks(steps) {
  const length = steps.length;
  return [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const index = clamp(Math.round((length - 1) * ratio), 0, Math.max(0, length - 1));
    return { ratio, value: steps[index] ?? index };
  });
}

function axisNameLayout(name, side = "left") {
  const normalized = String(name || "").trim();
  const useSidePlacement = normalized.length >= 18;
  return {
    useSidePlacement,
    nameLocation: useSidePlacement ? "middle" : "end",
    nameRotate: useSidePlacement ? (side === "right" ? -90 : 90) : 0,
    nameGap: useSidePlacement ? 40 : 10,
    padding: useSidePlacement ? [0, 0, 0, 0] : [0, 0, 4, 0],
  };
}

function svgLinePath(steps, values, xFor, yFor) {
  const size = Math.min(steps.length, values.length);
  let path = "";
  let open = false;
  for (let index = 0; index < size; index += 1) {
    const step = optionalNumber(steps[index]);
    const value = optionalNumber(values[index]);
    if (step === null || value === null) {
      open = false;
      continue;
    }
    const x = xFor(step, index);
    const y = yFor(value);
    path += `${open ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)} `;
    open = true;
  }
  return path.trim();
}

function svgBandPath(steps, centerValues, bandValues, xFor, yFor) {
  const upper = [];
  const lower = [];
  const size = Math.min(steps.length, centerValues.length, bandValues.length);
  for (let index = 0; index < size; index += 1) {
    const step = optionalNumber(steps[index]);
    const center = optionalNumber(centerValues[index]);
    const band = optionalNumber(bandValues[index]);
    if (step === null || center === null || band === null) continue;
    const x = xFor(step, index);
    upper.push(`${x.toFixed(2)} ${yFor(center + Math.abs(band)).toFixed(2)}`);
    lower.push(`${x.toFixed(2)} ${yFor(center - Math.abs(band)).toFixed(2)}`);
  }
  if (!upper.length || !lower.length) return "";
  return `M${upper.join(" L")} L${lower.reverse().join(" L")} Z`;
}

function renderLineChart(host, panelSpec, history) {
  let chartKey = "bottomLeft";
  if (host === elements.topLeftChart) {
    chartKey = "topLeft";
  } else if (host === elements.middleLeftChart) {
    chartKey = "middleLeft";
  }
  const built = buildPanelSeries(panelSpec, history);
  const allSeries = [...built.left, ...built.right];
  if (!built.steps.length || !allSeries.length) {
    clearChartHost(host, chartKey, "no data");
    return;
  }
  const chart = ensureChart(host, chartKey);
  if (!chart) {
    clearChartHost(host, chartKey, "chart unavailable");
    return;
  }

  const leftValues = built.left.flatMap((series) => series.rawValues.concat(series.values)).filter((value) => value !== null);
  const rightValues = built.right.flatMap((series) => series.rawValues.concat(series.values)).filter((value) => value !== null);
  const bandValues = built.shade[0]?.values || [];
  const leftAxisAccent = built.left[0]?.color || "#34d399";
  const rightAxisAccent = built.right[0]?.color || "#60a5fa";
  const rightAxisVisible = built.right.length > 0;
  const leftBandValues = (built.left[0]?.values || []).flatMap((value, index) => {
    const band = optionalNumber(bandValues[index]);
    if (!finiteNumber(value) || band === null) return [value];
    return [value - Math.abs(band), value + Math.abs(band)];
  }).filter((value) => value !== null);
  const leftRange = computeRange(leftValues.concat(leftBandValues));
  const rightRange = built.right.length ? computeRange(rightValues) : leftRange;
  const leftAxisName = String(panelSpec?.left_ylabel || panelSpec?.left_label || "value");
  const rightAxisName = String(panelSpec?.right_ylabel || panelSpec?.right_label || "value");
  const leftAxisNameLayout = axisNameLayout(leftAxisName, "left");
  const rightAxisNameLayout = axisNameLayout(rightAxisName, "right");
  const leftAxisNameGap = optionalInteger(panelSpec?.left_axis_name_gap);
  const legendItems = [];
  const seriesOptions = [];
  const legendType = allSeries.length > 4 ? "scroll" : "plain";
  const legendTop = legendType === "scroll" ? 68 : 56;

  host.style.height = `${Math.max(340, optionalInteger(panelSpec?.host_height) ?? 440)}px`;

  if (built.left.length && built.shade.length) {
    const baseSeries = built.left[0];
    const shadeSeries = built.shade[0];
    seriesOptions.push({
      name: "__band_lower__",
      type: "line",
      data: built.steps.map((step, index) => {
        const center = optionalNumber(baseSeries.values[index]);
        const band = optionalNumber(shadeSeries.values[index]);
        if (center === null || band === null) return [step, null];
        return [step, center - Math.abs(band)];
      }),
      lineStyle: { opacity: 0 },
      itemStyle: { color: "rgba(0, 0, 0, 0)" },
      symbol: "none",
      xAxisIndex: 0,
      yAxisIndex: 0,
      silent: true,
      tooltip: { show: false },
      emphasis: { disabled: true },
    });
    seriesOptions.push({
      name: formatFieldLabel(shadeSeries.key),
      type: "line",
      data: built.steps.map((step, index) => {
        const center = optionalNumber(baseSeries.values[index]);
        const band = optionalNumber(shadeSeries.values[index]);
        if (center === null || band === null) return [step, null];
        return [step, center + Math.abs(band)];
      }),
      lineStyle: { opacity: 0 },
      itemStyle: { color: "rgba(245, 158, 11, 0.14)" },
      symbol: "none",
      xAxisIndex: 0,
      yAxisIndex: 0,
      areaStyle: { color: "rgba(245, 158, 11, 0.14)" },
      stack: "std-band",
      tooltip: { show: false },
      emphasis: { disabled: true },
    });
  }

  for (const series of built.left) {
    if (series.showRaw) {
      seriesOptions.push({
        name: `${series.label} raw`,
        type: "line",
        data: built.steps.map((step, index) => [step, series.rawValues[index]]),
        xAxisIndex: 0,
        yAxisIndex: 0,
        symbol: "none",
        itemStyle: { color: series.color },
        lineStyle: { width: 1, color: series.color, opacity: 0.22 },
        silent: true,
        tooltip: { show: false },
        emphasis: { disabled: true },
      });
    }
    seriesOptions.push({
      name: series.label,
      type: "line",
      data: built.steps.map((step, index) => [step, series.values[index]]),
      xAxisIndex: 0,
      yAxisIndex: 0,
      symbol: "none",
      smooth: 0.15,
      itemStyle: { color: series.color },
      lineStyle: { width: series.strokeWidth, color: series.color },
    });
    legendItems.push(series.label);
  }

  for (const series of built.right) {
    if (series.showRaw) {
      seriesOptions.push({
        name: `${series.label} raw`,
        type: "line",
        data: built.steps.map((step, index) => [step, series.rawValues[index]]),
        xAxisIndex: 0,
        yAxisIndex: 1,
        symbol: "none",
        itemStyle: { color: series.color },
        lineStyle: { width: 1, color: series.color, opacity: 0.22 },
        silent: true,
        tooltip: { show: false },
        emphasis: { disabled: true },
      });
    }
    seriesOptions.push({
      name: series.label,
      type: "line",
      data: built.steps.map((step, index) => [step, series.values[index]]),
      xAxisIndex: 0,
      yAxisIndex: 1,
      symbol: "none",
      smooth: 0.15,
      itemStyle: { color: series.color },
      lineStyle: { width: series.strokeWidth, color: series.color },
    });
    legendItems.push(series.label);
  }

  chart.setOption({
    animation: false,
    backgroundColor: "transparent",
    color: seriesOptions.map((seriesOption) => (
      seriesOption?.itemStyle?.color
      || seriesOption?.lineStyle?.color
      || "#94a3b8"
    )),
    grid: {
      left: leftAxisNameLayout.useSidePlacement ? 58 : 34,
      right: rightAxisVisible ? (rightAxisNameLayout.useSidePlacement ? 58 : 36) : 10,
      top: legendTop,
      bottom: 34,
      containLabel: true,
    },
    legend: {
      type: legendType,
      top: 4,
      left: 4,
      right: 4,
      data: legendItems,
      textStyle: {
        color: "#d9d1c4",
        fontSize: 14,
        fontFamily: "Cascadia Code, Consolas, monospace",
      },
      itemWidth: 16,
      itemHeight: 10,
      itemGap: 10,
      pageTextStyle: { color: "#b8b0a4", fontSize: 12.5 },
      pageIconColor: "#e6ddcf",
      pageIconInactiveColor: "rgba(230, 221, 207, 0.38)",
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(9, 12, 17, 0.96)",
      borderColor: "rgba(223, 211, 186, 0.18)",
      textStyle: { color: "#f2ede2", fontFamily: "Cascadia Code, Consolas, monospace", fontSize: 13.5 },
      valueFormatter: (value) => formatMetricValue(value),
    },
    xAxis: {
      type: "value",
      name: "global step",
      nameLocation: "middle",
      nameGap: 24,
      nameTextStyle: {
        color: "#bdb4a6",
        fontSize: 14,
        fontWeight: "600",
        fontFamily: "Cascadia Code, Consolas, monospace",
      },
      min: built.steps[0],
      max: built.steps[built.steps.length - 1],
      axisLabel: {
        color: "#d7cfc1",
        fontSize: 13,
        fontFamily: "Cascadia Code, Consolas, monospace",
        formatter: (value) => formatAxisTick(value),
      },
      axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.34)" } },
      splitLine: { lineStyle: { color: "rgba(223, 211, 186, 0.12)", type: "dashed" } },
    },
    yAxis: [
      {
        type: "value",
        name: leftAxisName,
        min: leftRange.min,
        max: leftRange.max,
        nameLocation: leftAxisNameLayout.nameLocation,
        nameRotate: leftAxisNameLayout.nameRotate,
        nameGap: leftAxisNameGap ?? leftAxisNameLayout.nameGap,
        nameTextStyle: {
          color: leftAxisAccent,
          fontSize: 14,
          fontWeight: "700",
          padding: leftAxisNameLayout.padding,
          fontFamily: "Cascadia Code, Consolas, monospace",
        },
        axisLabel: {
          color: "#efd596",
          fontSize: 13,
          fontFamily: "Cascadia Code, Consolas, monospace",
          formatter: (value) => formatAxisTick(value),
        },
        axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.34)" } },
        splitLine: { lineStyle: { color: "rgba(223, 211, 186, 0.12)", type: "dashed" } },
      },
      {
        type: "value",
        name: rightAxisName,
        min: rightRange.min,
        max: rightRange.max,
        position: "right",
        show: rightAxisVisible,
        nameLocation: rightAxisNameLayout.nameLocation,
        nameRotate: rightAxisNameLayout.nameRotate,
        nameGap: rightAxisNameLayout.nameGap,
        nameTextStyle: {
          color: rightAxisAccent,
          fontSize: 14,
          fontWeight: "700",
          padding: rightAxisNameLayout.padding,
          fontFamily: "Cascadia Code, Consolas, monospace",
        },
        axisLabel: {
          color: "#bad3ff",
          fontSize: 13,
          fontFamily: "Cascadia Code, Consolas, monospace",
          formatter: (value) => formatAxisTick(value),
        },
        axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.28)" } },
        splitLine: { show: false },
      },
    ],
    series: seriesOptions,
  }, true);
  chart.resize();
}

function formatCurrentClassText(projection, metrics) {
  const currentClassIndex = optionalInteger(projection?.current_class_index)
    ?? optionalInteger(projection?.current_transition?.class_index)
    ?? optionalInteger(metrics?.current_dynamics_class);
  if (!currentClassIndex || currentClassIndex <= 0) {
    if (projection?.current_transition_explained_by_current_program === false || projection?.current_is_new_dynamics_class) {
      return "Current: ?";
    }
    return "Current: none (0)";
  }
  const groupLabel = String(projection?.current_group_label || projection?.current_group_id || projection?.current_class_label || "").trim();
  const groupClassIndex = optionalInteger(projection?.current_group_class_index);
  const resolvedClassIndex = groupClassIndex && groupClassIndex > 0 ? groupClassIndex : currentClassIndex;
  const classCount = optionalInteger(projection?.current_class_count);
  const duplicateCount = optionalInteger(metrics?.current_state_action_duplicate_count);
  const suffixParts = [];
  if (groupLabel) suffixParts.push(groupLabel);
  if (classCount !== null && classCount >= 0) suffixParts.push(`${classCount} canon`);
  if (duplicateCount !== null && duplicateCount >= 0) suffixParts.push(`duplicate samples: ${duplicateCount}`);
  return `Current: C${resolvedClassIndex}${suffixParts.length ? ` [${suffixParts.join(" | ")}]` : ""}`;
}

function formatCurrentPredictedSummary(currentText, predictedText) {
  return formatLines([currentText, predictedText]);
}

function resolveLlmCallCount(summary) {
  const usage = summary?.llmUsageSummary || {};
  const overall = usage?.overall;
  if (overall && typeof overall === "object") {
    const count = optionalInteger(overall.call_count);
    if (count !== null) return count;
    const requestCount = optionalInteger(overall.requests);
    if (requestCount !== null) return requestCount;
  }
  const directCount = optionalInteger(usage?.total_calls);
  if (directCount !== null) return directCount;
  return null;
}

function resolvePrototypeCount(payload) {
  const anchors = payload?.dashboard?.visitationHeatmap?.anchors;
  if (Array.isArray(anchors)) {
    return anchors.filter((anchor) => (
      optionalNumber(anchor?.x) !== null
      && optionalNumber(anchor?.y) !== null
    )).length;
  }
  const classRows = payload?.dashboard?.classRows;
  if (Array.isArray(classRows)) {
    return classRows.length;
  }
  const agent = payload?.dashboard?.agent || {};
  const activeCount = optionalInteger(agent.active_class_count);
  if (activeCount !== null && activeCount >= 0) return activeCount;
  const knownCount = optionalInteger(agent.known_class_count);
  if (knownCount !== null && knownCount >= 0) return knownCount;
  return null;
}

function basename(path) {
  const normalized = String(path || "").trim();
  if (!normalized) return "";
  const parts = normalized.split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : normalized;
}

function formatLlmRuntimeLine(summary, payload) {
  const llmSummary = summary?.llmRuntimeSummary || summary?.llmConfigSummary || {};
  const provider = String(llmSummary.provider || "").trim();
  const model = String(llmSummary.model || "").trim();
  const thinking = String(llmSummary.thinkingLevel || "").trim();
  const reasoning = String(llmSummary.reasoningEffort || "").trim();
  const parts = [];
  if (provider) parts.push(`llm=${provider}`);
  if (model) parts.push(`model=${model}`);
  if (thinking) parts.push(`thinking=${thinking}`);
  if (reasoning) parts.push(`reasoning=${reasoning}`);
  return parts.join(" | ");
}

function resolveAgentTypeLabel(payload) {
  const agentType = String(
    payload?.dashboard?.agent?.name
    || payload?.dashboard?.diagnostics?.name
    || "",
  ).trim();
  return agentType || "-";
}

function formatAgentTypeLine(payload) {
  return `agent type: ${resolveAgentTypeLabel(payload)}`;
}

function projectionMethodLabel(rawMethod) {
  const normalized = String(rawMethod || "projection").trim().toLowerCase();
  if (normalized === "pca" || normalized === "tsne") return "T-SNE";
  if (normalized === "degenerate") return "T-SNE warmup";
  return String(rawMethod || "projection").trim().toUpperCase();
}

function projectionSelectionRunKey(runPayload) {
  return String(runPayload?.runId || activeRun()?.runId || "none");
}

function projectionPinnedClass(runPayload) {
  const runKey = projectionSelectionRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.projectionPinnedClassByRun, runKey)) {
    return null;
  }
  return optionalInteger(state.projectionPinnedClassByRun[runKey]);
}

function projectionPinnedWorld(runPayload) {
  const runKey = projectionSelectionRunKey(runPayload);
  if (!Object.prototype.hasOwnProperty.call(state.projectionPinnedWorldByRun, runKey)) {
    return null;
  }
  return normalizeTransitionWorldIndex(state.projectionPinnedWorldByRun[runKey]);
}

function projectionActiveFocusClass(runPayload) {
  return projectionPinnedClass(runPayload);
}

function projectionActiveFocusWorld(runPayload) {
  return projectionPinnedWorld(runPayload);
}

function projectionClassSummaryLabel(classIndex) {
  const normalized = optionalInteger(classIndex);
  return normalized !== null && normalized > 0 ? `C${normalized}` : "NEW";
}

function toggleProjectionPinnedClass(runPayload, classIndex) {
  const runKey = projectionSelectionRunKey(runPayload);
  const normalized = optionalInteger(classIndex);
  if (normalized === null || normalized < 0) return;
  const currentPinned = projectionPinnedClass(runPayload);
  if (currentPinned !== null && currentPinned !== normalized) {
    return;
  }
  let changed = false;

  if (currentPinned === normalized) {
    if (Object.prototype.hasOwnProperty.call(state.projectionPinnedClassByRun, runKey)) {
      delete state.projectionPinnedClassByRun[runKey];
      changed = true;
    }
  } else {
    state.projectionPinnedClassByRun[runKey] = normalized;
    changed = true;
  }

  if (changed) {
    syncProjectionSelectionState(runPayload);
  }
}

function toggleProjectionPinnedWorld(runPayload, worldIndex) {
  const runKey = projectionSelectionRunKey(runPayload);
  const normalized = normalizeTransitionWorldIndex(worldIndex);
  if (normalized === null) return;
  const currentPinned = projectionPinnedWorld(runPayload);
  if (currentPinned !== null && currentPinned !== normalized) {
    return;
  }
  let changed = false;

  if (currentPinned === normalized) {
    if (Object.prototype.hasOwnProperty.call(state.projectionPinnedWorldByRun, runKey)) {
      delete state.projectionPinnedWorldByRun[runKey];
      changed = true;
    }
  } else {
    state.projectionPinnedWorldByRun[runKey] = normalized;
    changed = true;
  }

  if (changed) {
    syncProjectionSelectionState(runPayload);
  }
}

function formatTransitionClassLabel(classIndex) {
  const resolved = optionalInteger(classIndex);
  return resolved && resolved > 0 ? `C${resolved}` : "none";
}

function setProjectionSummaryText(model) {
  if (!elements.projectionSummary || !model) return;
  const lines = [model.currentText, model.predictedText, model.sparseFilterText];
  elements.projectionSummary.textContent = formatLines(lines);
}

function syncPrototypeLegendSelectionState(
  runPayload,
  activeClass = projectionActiveFocusClass(runPayload),
  pinnedClass = projectionPinnedClass(runPayload),
) {
  if (!elements.prototypeLegend) return;
  const hasFocus = activeClass !== null;
  const items = elements.prototypeLegend.querySelectorAll(".prototype-legend-item[data-class-index]");
  for (const item of items) {
    const itemClassIndex = optionalInteger(item.dataset.classIndex);
    const isActive = hasFocus && itemClassIndex === activeClass;
    const isPinned = pinnedClass !== null && itemClassIndex === pinnedClass;
    item.classList.toggle("is-active", isActive);
    item.classList.toggle("is-dimmed", hasFocus && !isActive);
    item.classList.toggle("is-locked", isPinned);
  }
}

function syncProjectionTargetWorldSelectionState(
  runPayload,
  activeWorldIndex = projectionActiveFocusWorld(runPayload),
  pinnedWorldIndex = projectionPinnedWorld(runPayload),
) {
  if (!elements.projectionTargetWorldList) return;
  const hasFocus = activeWorldIndex !== null;
  const items = elements.projectionTargetWorldList.querySelectorAll(".projection-target-world-chip[data-world-index]");
  for (const item of items) {
    const itemWorldIndex = normalizeTransitionWorldIndex(item.dataset.worldIndex);
    const isActive = hasFocus && itemWorldIndex === activeWorldIndex;
    const isPinned = pinnedWorldIndex !== null && itemWorldIndex === pinnedWorldIndex;
    item.classList.toggle("is-selected", isActive);
    item.classList.toggle("is-dimmed", hasFocus && !isActive);
    item.classList.toggle("is-locked", isPinned);
    item.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
}

function attachPrototypeLegendSelectionInteractions(item, runPayload, classIndex) {
  if (!item) return;
  item.addEventListener("click", (event) => {
    event.preventDefault();
    toggleProjectionPinnedClass(runPayload, classIndex);
  });
}

function buildProjectionRenderModel(runPayload) {
  const dashboard = runPayload?.dashboard || {};
  const projection = dashboard.visitationHeatmap;
  const metrics = dashboard.metrics || {};
  if (!projection || !Array.isArray(projection.transitions)) {
    return null;
  }

  const anchors = Array.isArray(projection.anchors) ? projection.anchors : [];
  const prototypeAnchors = Array.isArray(projection.prototype_anchors) ? projection.prototype_anchors : [];
  const transitions = Array.isArray(projection.transitions) ? projection.transitions : [];
  const current = projection.current_transition || null;
  const allPoints = [...anchors, ...prototypeAnchors, ...transitions]
    .filter((point) => optionalNumber(point?.x) !== null && optionalNumber(point?.y) !== null);
  if (!allPoints.length) {
    return null;
  }

  const xs = allPoints.map((point) => Number(point.x));
  const ys = allPoints.map((point) => Number(point.y));
  const xBounds = computeFiniteBounds(xs);
  const yBounds = computeFiniteBounds(ys);
  if (!xBounds || !yBounds) {
    return null;
  }
  const minRawX = xBounds.min;
  const maxRawX = xBounds.max;
  const minRawY = yBounds.min;
  const maxRawY = yBounds.max;
  const maxRange = Math.max(maxRawX - minRawX, maxRawY - minRawY, 1e-6);
  const centerX = (minRawX + maxRawX) / 2;
  const centerY = (minRawY + maxRawY) / 2;
  const minX = centerX - (maxRange / 2);
  const maxX = centerX + (maxRange / 2);
  const minY = centerY - (maxRange / 2);
  const maxY = centerY + (maxRange / 2);

  const predictedClass = optionalInteger(metrics.predicted_dynamics_class) ?? optionalInteger(projection.current_predicted_class_index);
  const predictedConfidence = optionalNumber(metrics.predicted_dynamics_confidence) ?? optionalNumber(projection.current_predicted_class_probability);
  const currentText = formatCurrentClassText(projection, metrics);
  const predictedText = predictedClass !== null && predictedClass > 0
    ? `Pred: C${predictedClass}${predictedConfidence !== null ? ` (${formatPercent(predictedConfidence)})` : ""}`
    : "Pred: none";
  const method = projectionMethodLabel(projection.method);
  const sampledSize = optionalInteger(projection.sampled_sample_store_size) ?? transitions.length;
  const totalSize = optionalInteger(projection.total_sample_store_size) ?? sampledSize;
  const sparseFilterEnabled = Boolean(projection.projection_exclude_sparse_classes);
  const sparseFilterThreshold = optionalInteger(projection.projection_sparse_class_count_threshold);
  const sparseFilteredSize = optionalInteger(projection.projection_sparse_filtered_sample_store_size) ?? sampledSize;
  const sparseFilteredOutCount = optionalInteger(projection.projection_sparse_filtered_out_count) ?? 0;
  const sparseFilterText = sparseFilterEnabled
    ? `Filter < ${sparseFilterThreshold ?? "min"}: kept ${sparseFilteredSize}/${totalSize}, dropped ${sparseFilteredOutCount}`
    : "";
  const currentClassIndex = optionalInteger(current?.class_index) ?? 0;
  const currentClassColor = classColor(currentClassIndex);

  const anchorData = anchors
    .filter((anchor) => optionalNumber(anchor?.x) !== null && optionalNumber(anchor?.y) !== null)
    .map((anchor) => {
      const anchorClassIndex = optionalInteger(anchor.class_index) ?? 0;
      return {
        kind: "prototype",
        value: [Number(anchor.x), Number(anchor.y)],
        classIndex: anchorClassIndex,
        groupId: String(anchor.group_id || "").trim() || null,
        groupLabel: String(anchor.group_label || "").trim() || null,
        classCount: optionalInteger(anchor.class_count),
      };
    });

  const prototypeAnchorData = prototypeAnchors
    .filter((anchor) => optionalNumber(anchor?.x) !== null && optionalNumber(anchor?.y) !== null)
    .map((anchor) => {
      const anchorClassIndex = optionalInteger(anchor.class_index) ?? 0;
      return {
        kind: "prototype-member",
        value: [Number(anchor.x), Number(anchor.y)],
        classIndex: anchorClassIndex,
        prototypeSlot: optionalInteger(anchor.prototype_slot) ?? 1,
        prototypeCount: optionalInteger(anchor.prototype_count),
        groupId: String(anchor.group_id || "").trim() || null,
        groupLabel: String(anchor.group_label || "").trim() || null,
        classCount: optionalInteger(anchor.class_count),
      };
    });

  const transitionData = transitions
    .filter((transition) => optionalNumber(transition?.x) !== null && optionalNumber(transition?.y) !== null)
    .map((transition, index) => {
      const transitionClassIndex = optionalInteger(transition.class_index) ?? 0;
      return {
        id: `transition-${index}`,
        kind: "transition",
        value: [Number(transition.x), Number(transition.y)],
        classIndex: transitionClassIndex,
        envName: String(transition.env_name || "").trim() || null,
        worldIndex: normalizeTransitionWorldIndex(transition.world_index ?? transition.worldIndex),
        worldSeed: optionalInteger(transition.world_seed),
      };
    });

  const currentHaloData = current && optionalNumber(current.x) !== null && optionalNumber(current.y) !== null
    ? [{ kind: "current-halo", value: [Number(current.x), Number(current.y)] }]
    : [];
  const currentData = current && optionalNumber(current.x) !== null && optionalNumber(current.y) !== null
    ? [{ kind: "current", value: [Number(current.x), Number(current.y)], classIndex: currentClassIndex }]
    : [];
  const currentBadgeData = current && optionalNumber(current.x) !== null && optionalNumber(current.y) !== null
    ? [{ kind: "current-badge", value: [Number(current.x), Number(current.y)], classIndex: currentClassIndex }]
    : [];

  return {
    method,
    sampledSize,
    totalSize,
    minX,
    maxX,
    minY,
    maxY,
    centerX,
    centerY,
    currentText,
    predictedText,
    sparseFilterText,
    currentClassIndex,
    currentWorldIndex: normalizeTransitionWorldIndex(current?.world_index ?? current?.worldIndex),
    currentClassColor,
    transitionData,
    anchorData,
    prototypeAnchorData,
    currentHaloData,
    currentData,
    currentBadgeData,
  };
}

function transitionMatchesProjectionFocus(transition, activeClass, activeWorldIndex) {
  if (activeClass !== null && transition.classIndex !== activeClass) {
    return false;
  }
  if (activeWorldIndex !== null && transition.worldIndex !== activeWorldIndex) {
    return false;
  }
  return true;
}

function buildProjectionOption(model, activeClass = null, activeWorldIndex = null) {
  const hasClassFocus = activeClass !== null;
  const hasWorldFocus = activeWorldIndex !== null;
  const hasFocus = hasClassFocus || hasWorldFocus;
  const mutedTransitionFill = "rgba(96, 107, 122, 0.24)";
  const mutedTransitionStroke = "rgba(150, 162, 178, 0.42)";
  const mutedTransitionShadow = "rgba(154, 166, 182, 0.16)";
  const focusTransitionCount = hasFocus
    ? model.transitionData.reduce(
      (
        count,
        transition,
      ) => count + (transitionMatchesProjectionFocus(transition, activeClass, activeWorldIndex) ? 1 : 0),
      0,
    )
    : 0;
  const focusedTransitionFillOpacity = projectionFocusedTransitionFillOpacity(focusTransitionCount);
  const focusedTransitionBorderOpacity = projectionFocusedTransitionBorderOpacity(focusTransitionCount);
  const focusedTransitionSymbolSize = projectionFocusedTransitionSymbolSize(focusTransitionCount);
  const backdropTransitionSymbolSize = hasFocus
    ? clamp(focusedTransitionSymbolSize / 4, 2.2, 3.0)
    : 10;
  const backdropTransitionData = [];
  const focusTransitionData = [];
  for (const transition of model.transitionData) {
    const transitionColor = classColor(transition.classIndex);
    if (hasFocus && transitionMatchesProjectionFocus(transition, activeClass, activeWorldIndex)) {
      focusTransitionData.push({
        ...transition,
        itemStyle: {
          color: colorWithOpacity(transitionColor, focusedTransitionFillOpacity),
          borderColor: colorWithOpacity(transitionColor, focusedTransitionBorderOpacity),
          borderWidth: 1.35,
        },
      });
      continue;
    }
    backdropTransitionData.push({
      ...transition,
      itemStyle: hasFocus
        ? {
          color: mutedTransitionFill,
          borderColor: mutedTransitionStroke,
          borderWidth: 0.9,
          opacity: 0.2,
          shadowBlur: 3,
          shadowColor: mutedTransitionShadow,
        }
        : {
          color: transitionColor,
          opacity: 0.58,
        },
    });
  }

  const backdropAnchorData = [];
  const focusAnchorData = [];
  for (const anchor of model.anchorData) {
    const anchorColor = classColor(anchor.classIndex);
    const anchorEntry = {
      ...anchor,
      label: {
        show: true,
        formatter: () => String(anchor.classIndex > 0 ? anchor.classIndex : "?"),
        color: "#f4efe6",
        fontFamily: "Cascadia Code, Consolas, monospace",
        fontWeight: "700",
        fontSize: 14,
        textShadowBlur: 8,
        textShadowColor: "rgba(0, 0, 0, 0.48)",
      },
    };
    if (hasClassFocus && anchor.classIndex === activeClass) {
      focusAnchorData.push({
        ...anchorEntry,
        symbolSize: 38,
        itemStyle: {
          color: "rgba(10, 14, 20, 0.94)",
          borderColor: anchorColor,
          borderWidth: 2.5,
          shadowBlur: 12,
          shadowColor: projectionShadowColor(anchorColor, 0.21),
        },
      });
      continue;
    }
    backdropAnchorData.push({
      ...anchorEntry,
      symbolSize: hasFocus ? 31 : 34,
      itemStyle: hasFocus
        ? {
          color: "rgba(10, 14, 20, 0.62)",
          borderColor: `${anchorColor}78`,
          borderWidth: 1.8,
          opacity: 0.2,
          shadowBlur: 2,
          shadowColor: projectionShadowColor(anchorColor, 0.08),
        }
        : {
          color: "rgba(10, 14, 20, 0.92)",
          borderColor: anchorColor,
          borderWidth: 2.2,
          shadowBlur: 9,
          shadowColor: projectionShadowColor(anchorColor, 0.14),
        },
      label: hasFocus
        ? {
          ...anchorEntry.label,
          color: "rgba(244, 239, 230, 0.36)",
          textShadowBlur: 0,
        }
        : anchorEntry.label,
    });
  }
  const focusPrototypeAnchorData = hasClassFocus
    ? model.prototypeAnchorData
      .filter((anchor) => anchor.classIndex === activeClass)
      .map((anchor) => {
        const anchorColor = classColor(anchor.classIndex);
        const prototypeSlot = optionalInteger(anchor.prototypeSlot) ?? 1;
        return {
          ...anchor,
          symbol: "diamond",
          symbolSize: 22,
          itemStyle: {
            color: colorWithOpacity(anchorColor, 0.24),
            borderColor: anchorColor,
            borderWidth: 2.1,
            shadowBlur: 10,
            shadowColor: projectionShadowColor(anchorColor, 0.18),
          },
          label: {
            show: true,
            formatter: () => `P${prototypeSlot}`,
            color: "#f4efe6",
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontWeight: "700",
            fontSize: 11.5,
            textShadowBlur: 6,
            textShadowColor: "rgba(0, 0, 0, 0.38)",
          },
        };
      })
    : [];

  const currentMuted = (
    (hasClassFocus && model.currentClassIndex !== activeClass)
    || (hasWorldFocus && model.currentWorldIndex !== activeWorldIndex)
  );
  const currentBadgePosition = model.currentData.length && Number(model.currentData[0].value[0]) > model.centerX ? "left" : "right";

  return {
    animation: true,
    animationDuration: 0,
    animationDurationUpdate: 180,
    animationEasingUpdate: "cubicInOut",
    backgroundColor: "transparent",
    grid: { left: 14, right: 14, top: 18, bottom: 14, containLabel: false },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(9, 12, 17, 0.96)",
      borderColor: "rgba(223, 211, 186, 0.18)",
      textStyle: { color: "#f2ede2", fontFamily: "Cascadia Code, Consolas, monospace", fontSize: 12.5 },
      formatter: (params) => {
        if (params.data?.kind === "current") {
          return `CURRENT<br/>${escapeHtml(model.currentText)}<br/>${escapeHtml(model.predictedText)}`;
        }
        if (params.data?.kind === "prototype") {
          const prototypeIdentity = formatPrototypeIdentity(params.data);
          const canonCount = optionalInteger(params.data?.classCount);
          const detailLines = [
            `Prototype ${escapeHtml(projectionClassSummaryLabel(params.data.classIndex))}`,
            `ID: ${escapeHtml(prototypeIdentity)}`,
          ];
          if (canonCount !== null && canonCount >= 0) {
            detailLines.push(`Canon: ${escapeHtml(canonCount)}`);
          }
          return detailLines.join("<br/>");
        }
        if (params.data?.kind === "prototype-member") {
          const prototypeIdentity = formatPrototypeIdentity(params.data);
          const canonCount = optionalInteger(params.data?.classCount);
          const prototypeSlot = optionalInteger(params.data?.prototypeSlot);
          const detailLines = [
            `Prototype ${escapeHtml(projectionClassSummaryLabel(params.data.classIndex))}${prototypeSlot !== null ? ` · P${prototypeSlot}` : ""}`,
            `ID: ${escapeHtml(prototypeIdentity)}`,
          ];
          if (canonCount !== null && canonCount >= 0) {
            detailLines.push(`Canon: ${escapeHtml(canonCount)}`);
          }
          return detailLines.join("<br/>");
        }
        return [
          "Transition",
          `Class: ${escapeHtml(formatTransitionClassLabel(params.data?.classIndex))}`,
          `World: ${escapeHtml(formatTransitionWorldLabel({
            env_name: params.data?.envName,
            world_index: params.data?.worldIndex,
            world_seed: params.data?.worldSeed,
          }))}`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      min: model.minX,
      max: model.maxX,
      scale: true,
      axisLabel: { show: false },
      axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.22)" } },
      splitLine: { lineStyle: { color: "rgba(223, 211, 186, 0.1)", type: "dashed" } },
    },
    yAxis: {
      type: "value",
      min: model.minY,
      max: model.maxY,
      scale: true,
      axisLabel: { show: false },
      axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.22)" } },
      splitLine: { lineStyle: { color: "rgba(223, 211, 186, 0.1)", type: "dashed" } },
    },
    series: [
      {
        id: "projection-transitions-backdrop",
        name: "TransitionsBackdrop",
        type: "scatter",
        data: backdropTransitionData,
        symbolSize: backdropTransitionSymbolSize,
        z: 1,
        universalTransition: {
          enabled: true,
          seriesKey: "projection-transitions",
        },
        emphasis: { scale: hasFocus ? 1.02 : 1.12 },
      },
      {
        id: "projection-transitions-focus",
        name: "TransitionsFocus",
        type: "scatter",
        data: focusTransitionData,
        symbolSize: focusedTransitionSymbolSize,
        z: 8,
        universalTransition: {
          enabled: true,
          seriesKey: "projection-transitions",
        },
        emphasis: { scale: 1.04 },
      },
      {
        name: "PrototypesBackdrop",
        type: "scatter",
        data: backdropAnchorData,
        cursor: "pointer",
        z: 3,
        emphasis: { scale: 1.0 },
      },
      {
        name: "PrototypesFocus",
        type: "scatter",
        data: focusAnchorData,
        symbolSize: 38,
        cursor: "pointer",
        z: 9,
        emphasis: { scale: 1.01 },
      },
      {
        name: "PrototypeMembersFocus",
        type: "scatter",
        data: focusPrototypeAnchorData,
        cursor: "pointer",
        z: 10,
        emphasis: { scale: 1.03 },
      },
      {
        name: "CurrentHalo",
        type: "scatter",
        data: model.currentHaloData.map((item) => ({
          ...item,
          itemStyle: {
            color: currentMuted ? mutedTransitionFill : `${model.currentClassColor}22`,
            borderColor: currentMuted ? mutedTransitionStroke : `${model.currentClassColor}88`,
            borderWidth: 1.2,
            opacity: currentMuted ? 0.55 : 1,
            shadowBlur: currentMuted ? 4 : 14,
            shadowColor: currentMuted
              ? mutedTransitionShadow
              : projectionShadowColor(model.currentClassColor, 0.26),
          },
        })),
        symbolSize: currentMuted ? 38 : 48,
        silent: true,
        tooltip: { show: false },
        z: 5,
      },
      {
        name: "Current",
        type: "scatter",
        data: model.currentData.map((item) => ({
          ...item,
          itemStyle: {
            color: currentMuted ? "rgba(208, 215, 224, 0.76)" : "rgba(247, 242, 233, 0.98)",
            borderColor: currentMuted ? mutedTransitionStroke : model.currentClassColor,
            borderWidth: currentMuted ? 3.2 : 4.2,
            shadowBlur: currentMuted ? 5 : 10,
            shadowColor: currentMuted
              ? mutedTransitionShadow
              : projectionShadowColor(model.currentClassColor, 0.27),
          },
          label: {
            show: true,
            formatter: () => String((optionalInteger(item.classIndex) ?? 0) > 0 ? optionalInteger(item.classIndex) : "?"),
            color: currentMuted ? "rgba(15, 23, 32, 0.82)" : "#0f1720",
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontWeight: "700",
            fontSize: currentMuted ? 15 : 17,
            textShadowBlur: 0,
            textShadowColor: "rgba(0, 0, 0, 0)",
          },
        })),
        symbolSize: currentMuted ? 29 : 33,
        z: 6,
      },
      {
        name: "CurrentBadge",
        type: "scatter",
        data: model.currentBadgeData.map((item) => ({
          ...item,
          label: {
            show: true,
            formatter: () => "CURRENT",
            position: currentBadgePosition,
            distance: 22,
            color: currentMuted ? "rgba(248, 250, 252, 0.8)" : "#f8fafc",
            backgroundColor: currentMuted ? "rgba(15, 23, 32, 0.72)" : "rgba(15, 23, 32, 0.94)",
            borderColor: currentMuted ? mutedTransitionStroke : model.currentClassColor,
            borderWidth: 1.4,
            borderRadius: 10,
            padding: [6, 12, 6, 12],
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontWeight: "700",
            fontSize: 13.5,
          },
        })),
        symbolSize: 1,
        silent: true,
        tooltip: { show: false },
        itemStyle: { color: "rgba(0, 0, 0, 0)" },
        z: 7,
      },
    ],
  };
}

function bindProjectionSelectionInteractions(chart, runPayload) {
  if (!chart) return;
  chart.off("mouseover");
  chart.off("mouseout");
  chart.off("globalout");
  chart.off("click");
  chart.on("click", (params) => {
    if (params?.data?.kind !== "prototype" && params?.data?.kind !== "prototype-member") return;
    toggleProjectionPinnedClass(runPayload, params.data.classIndex);
  });
}

function syncProjectionSelectionState(runPayload) {
  const chart = chartState.projection;
  const model = chart?.__babaProjectionModel || null;
  const activeClass = projectionActiveFocusClass(runPayload);
  const activeWorldIndex = projectionActiveFocusWorld(runPayload);
  const pinnedClass = projectionPinnedClass(runPayload);
  const pinnedWorldIndex = projectionPinnedWorld(runPayload);
  if (chart && model) {
    chart.setOption(buildProjectionOption(model, activeClass, activeWorldIndex), true);
  }
  renderProjectionTargetWorldLegend(runPayload);
  renderPrototypeLegend(runPayload);
  setProjectionSummaryText(model);
  syncPrototypeLegendSelectionState(runPayload, activeClass, pinnedClass);
  syncProjectionTargetWorldSelectionState(runPayload, activeWorldIndex, pinnedWorldIndex);
}

function formatTransitionWorldLabel(transition) {
  if (!transition || typeof transition !== "object") return "-";
  const envName = String(transition.env_name || "").trim();
  const worldIndex = optionalInteger(transition.world_index);
  const seed = optionalInteger(transition.world_seed);
  const parts = [];
  if (envName) {
    parts.push(envName);
  }
  if (worldIndex && worldIndex > 0) {
    parts.push(`world ${worldIndex}`);
  }
  if (seed !== null) {
    parts.push(`seed ${seed}`);
  }
  return parts.length ? parts.join(" | ") : "-";
}

function formatCountLabel(value) {
  const parsed = optionalInteger(value);
  if (parsed === null) return "-";
  return parsed.toLocaleString("en-US");
}

function classRowsScrollRunKey(runPayload) {
  return String(runPayload?.runId || state.activeRunId || "none");
}

function ensureClassRowsScrollTop(runKey) {
  const normalizedRunKey = String(runKey || "none");
  if (!Object.prototype.hasOwnProperty.call(state.classRowsScrollTopByRun, normalizedRunKey)) {
    state.classRowsScrollTopByRun[normalizedRunKey] = 0;
  }
  return normalizedRunKey;
}

function titleCaseWords(value) {
  return String(value || "")
    .split(/\s+/u)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function activeConfigSnapshotVisibility(runId) {
  const runKey = String(runId || "none");
  if (!state.configSnapshotVisibilityByRun[runKey]) {
    state.configSnapshotVisibilityByRun[runKey] = {};
  }
  return state.configSnapshotVisibilityByRun[runKey];
}

function normalizeConfigSnapshotFiles(summary) {
  const rawFiles = summary?.configSnapshotFiles;
  if (!Array.isArray(rawFiles)) return [];
  return rawFiles
    .filter((entry) => entry && typeof entry === "object")
    .map((entry) => {
      const name = String(entry.name || "").trim();
      const labelSource = String(entry.label || name || "").trim();
      return {
        name,
        label: titleCaseWords(labelSource),
        content: String(entry.content || ""),
      };
    })
    .filter((entry) => entry.name && entry.content);
}

function renderConfigSnapshots(payload) {
  const host = elements.configSnapshots;
  if (!host) return;
  const safeRunId = String(payload?.runId || "").trim();
  const files = normalizeConfigSnapshotFiles(payload?.summary || {});
  if (!safeRunId) {
    host.innerHTML = "";
    return;
  }
  const visibility = activeConfigSnapshotVisibility(payload?.runId);
  const activeNames = new Set(files.map((entry) => entry.name));
  for (const key of Object.keys(visibility)) {
    if (!activeNames.has(key)) delete visibility[key];
  }

  host.innerHTML = "";

  const controls = document.createElement("div");
  controls.className = "config-snapshot-toggle-row";
  const controlsActions = document.createElement("div");
  controlsActions.className = "config-snapshot-actions";
  const panels = document.createElement("div");
  panels.className = "config-snapshot-panels";

  for (const file of files) {
    const isOpen = Boolean(visibility[file.name]);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `button config-snapshot-toggle${isOpen ? " active" : ""}`;
    button.textContent = file.label;
    button.addEventListener("click", () => {
      visibility[file.name] = !Boolean(visibility[file.name]);
      invalidatePanelCache("configSnapshots");
      renderActive();
    });
    controls.append(button);
  }

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "button config-snapshot-delete-button";
  deleteButton.setAttribute("aria-label", `${safeRunId} delete`);
  deleteButton.innerHTML = `
    <svg viewBox="0 0 16 16" focusable="false" aria-hidden="true">
      <path d="M3.5 4.5h9"></path>
      <path d="M6 4.5v-1a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1"></path>
      <path d="M5.2 6.5v5.2"></path>
      <path d="M8 6.5v5.2"></path>
      <path d="M10.8 6.5v5.2"></path>
      <path d="M4.5 4.5l.6 8.1a1.1 1.1 0 0 0 1.1 1h3.6a1.1 1.1 0 0 0 1.1-1l.6-8.1"></path>
    </svg>
    <span>Delete Run</span>
  `;
  const deleteDisabledReason = resolveRunDeleteDisabledReason(payload);
  deleteButton.disabled = Boolean(deleteDisabledReason);
  deleteButton.title = deleteDisabledReason || `${safeRunId} delete`;
  deleteButton.addEventListener("click", () => {
    openRunDeleteModal(payload);
  });
  controlsActions.append(deleteButton);
  controls.append(controlsActions);

  for (const file of files) {
    if (!Boolean(visibility[file.name])) continue;
    const panel = document.createElement("section");
    panel.className = "config-snapshot-panel";

    const title = document.createElement("div");
    title.className = "config-snapshot-panel-title";
    title.textContent = file.label;

    const body = document.createElement("pre");
    body.className = "config-snapshot-content";
    body.textContent = file.content.trimEnd();

    panel.append(title, body);
    panels.append(panel);
  }

  host.append(controls);
  if (files.length) {
    host.append(panels);
  }
}

function formatPrototypeIdentity(anchor) {
  if (!anchor || typeof anchor !== "object") return "-";
  const label = String(anchor.groupLabel || anchor.groupId || "").trim();
  return label || "-";
}

function buildClassRowLookup(dashboard) {
  const rows = dashboard?.classRows;
  const lookup = new Map();
  if (!Array.isArray(rows)) {
    return lookup;
  }
  for (const row of rows) {
    const classIndex = optionalInteger(row?.version_index);
    if (classIndex === null || classIndex <= 0) continue;
    lookup.set(classIndex, row);
  }
  return lookup;
}

function normalizeTransitionWitnessGroupKey(value, fallbackCommitVersion = null) {
  const text = String(value || "").trim();
  if (!text) return null;
  return normalizeGroupVersionKey(text, fallbackCommitVersion) || text.toLowerCase();
}

function buildClassRowGroupLookup(dashboard) {
  if (dashboard && typeof dashboard === "object") {
    const cached = classRowGroupLookupCache.get(dashboard);
    if (cached) {
      return cached;
    }
  }
  const rows = dashboard?.classRows;
  const lookup = new Map();
  if (!Array.isArray(rows)) {
    return lookup;
  }
  for (const row of rows) {
    const groupKey = normalizeTransitionWitnessGroupKey(
      row?.group_id || row?.version_id || row?.display_id,
      row?.commit_version,
    );
    if (!groupKey) continue;
    lookup.set(groupKey, row);
  }
  if (dashboard && typeof dashboard === "object") {
    classRowGroupLookupCache.set(dashboard, lookup);
  }
  return lookup;
}

function resolveTransitionWitnessGroupContext(runPayload, groupId, fallbackCommitVersion = null) {
  const rawGroupId = String(groupId || "").trim();
  if (!rawGroupId) return null;
  const groupKey = normalizeTransitionWitnessGroupKey(rawGroupId, fallbackCommitVersion);
  const row = groupKey
    ? buildClassRowGroupLookup(runPayload?.dashboard || null).get(groupKey) || null
    : null;
  const classIndex = optionalInteger(row?.version_index);
  const groupLabelSource = String(
    row?.display_id
    || row?.version_id
    || row?.group_id
    || groupKey
    || rawGroupId
  ).trim();
  return {
    classIndex: classIndex !== null && classIndex > 0 ? classIndex : null,
    groupId: rawGroupId,
    groupKey: groupKey || rawGroupId.toLowerCase(),
    groupLabel: compactPrefixIdentityLabel(groupLabelSource),
  };
}

function formatTransitionWitnessGroupIdentity(runPayload, groupId, fallbackCommitVersion = null) {
  const context = resolveTransitionWitnessGroupContext(runPayload, groupId, fallbackCommitVersion);
  if (!context) return null;
  const parts = [];
  if (context.classIndex !== null) {
    parts.push(`C${context.classIndex}`);
  }
  if (context.groupLabel) {
    parts.push(context.groupLabel);
  }
  if (!parts.length) {
    return context.groupId || null;
  }
  return parts.join(" · ");
}

function formatTransitionWitnessGroupDisplayLabel(runPayload, groupId, fallbackCommitVersion = null) {
  const formattedIdentity = formatTransitionWitnessGroupIdentity(
    runPayload,
    groupId,
    fallbackCommitVersion,
  );
  if (formattedIdentity) {
    return formattedIdentity;
  }
  const rawGroupId = String(groupId || "").trim();
  if (rawGroupId) {
    return compactPrefixIdentityLabel(rawGroupId);
  }
  const rawCommitVersion = normalizeCommitVersion(fallbackCommitVersion);
  return rawCommitVersion ? versionLabel(rawCommitVersion) : null;
}

function buildTransitionWitnessGroupPresentationSignature(runPayload, bundleMeta) {
  const labels = [];
  const selectedRegressionGroupIds = Array.isArray(bundleMeta?.selectedRegressionGroupIds)
    ? bundleMeta.selectedRegressionGroupIds
    : [];
  for (const groupId of selectedRegressionGroupIds) {
    const label = formatTransitionWitnessGroupDisplayLabel(runPayload, groupId);
    if (label) {
      labels.push(label);
    }
  }
  const selectedWitnesses = Array.isArray(bundleMeta?.selectedWitnesses)
    ? bundleMeta.selectedWitnesses
    : [];
  for (const witness of selectedWitnesses) {
    const label = formatTransitionWitnessGroupDisplayLabel(
      runPayload,
      witness?.groupId,
      witness?.commitVersion,
    );
    if (label) {
      labels.push(label);
    }
  }
  return Array.from(new Set(labels))
    .sort((left, right) => left.localeCompare(right));
}

function formatProbabilityAxisLabel(classIndex, classRow) {
  if (!classIndex || classIndex <= 0) {
    return "Others";
  }
  const compactClass = `C${classIndex}`;
  const groupId = String(
    classRow?.group_id
    || classRow?.version_id
    || classRow?.display_id
    || ""
  ).trim();
  if (!groupId) {
    return compactClass;
  }
  if (groupId.includes(":g")) {
    const [versionToken, groupToken] = groupId.split(":g", 2);
    const versionLabel = String(versionToken || "").trim() || "-";
    const normalizedGroupToken = String(groupToken || "").trim();
    const groupLabel = normalizedGroupToken
      ? (normalizedGroupToken.startsWith("g") ? normalizedGroupToken : `g${normalizedGroupToken}`)
      : "g-";
    return `${compactClass}\n${versionLabel}\n${groupLabel}`;
  }
  return `${compactClass}\n${groupId}`;
}

function renderProjection(runPayload) {
  const host = elements.projection;
  const projectionModel = buildProjectionRenderModel(runPayload);
  if (!projectionModel) {
    clearChartHost(host, "projection", "projection unavailable");
    elements.projectionTitle.textContent = "Transition Projection";
    if (elements.projectionSummary) {
      elements.projectionSummary.textContent = "";
    }
    return;
  }

  elements.projectionTitle.textContent = `Sample ${projectionModel.method} (${projectionModel.sampledSize}/${projectionModel.totalSize})`;
  host.style.height = "628px";
  const chart = ensureChart(host, "projection");
  if (!chart) {
    clearChartHost(host, "projection", "projection unavailable");
    return;
  }
  const activeClass = projectionActiveFocusClass(runPayload);
  const activeWorldIndex = projectionActiveFocusWorld(runPayload);
  chart.__babaProjectionModel = projectionModel;
  chart.setOption(buildProjectionOption(projectionModel, activeClass, activeWorldIndex), true);
  bindProjectionSelectionInteractions(chart, runPayload);
  setProjectionSummaryText(projectionModel);
  chart.resize();
}

function renderProbabilities(runPayload) {
  const host = elements.probabilities;
  const dashboard = runPayload?.dashboard || {};
  const heatmap = dashboard.visitationHeatmap || {};
  const metrics = dashboard.metrics || {};
  const rows = Array.isArray(heatmap.current_class_probabilities) ? heatmap.current_class_probabilities : [];
  let title = String(heatmap.class_probability_title || "Predicted Class Probs");
  const softmaxTemperature = optionalNumber(metrics.softmax_temperature) ?? optionalNumber(heatmap.softmax_temperature);
  if (softmaxTemperature !== null) {
    title = `${title} (T=${softmaxTemperature.toFixed(2)})`;
  }
  elements.probabilitiesTitle.textContent = title;

  if (!rows.length) {
    clearChartHost(host, "probabilities", "No class probability available yet");
    if (elements.probabilitiesSummary) {
      elements.probabilitiesSummary.textContent = "";
    }
    return;
  }

  const predictedClass = optionalInteger(metrics.predicted_dynamics_class) ?? optionalInteger(heatmap.current_predicted_class_index);
  const predictedConfidence = optionalNumber(metrics.predicted_dynamics_confidence) ?? optionalNumber(heatmap.current_predicted_class_probability);
  const currentClass = optionalInteger(heatmap.current_class_index)
    ?? optionalInteger(heatmap.current_transition?.class_index)
    ?? optionalInteger(metrics.current_dynamics_class);
  const sortedRows = rows
    .map((row) => ({
      classIndex: optionalInteger(row?.class_index),
      probability: clamp(optionalNumber(row?.probability) ?? 0, 0, 1),
    }))
    .filter((row) => row.classIndex !== null && row.classIndex > 0)
    .sort((left, right) => right.probability - left.probability);
  const maxProbabilityBars = 5;
  const needsOthers = sortedRows.length > maxProbabilityBars;
  const visibleRows = sortedRows.slice(0, needsOthers ? maxProbabilityBars - 1 : maxProbabilityBars);
  if (needsOthers) {
    const remainingProbability = sortedRows
      .slice(maxProbabilityBars - 1)
      .reduce((sum, row) => sum + row.probability, 0);
    if (remainingProbability > 1e-6) {
      visibleRows.push({ classIndex: 0, probability: remainingProbability });
    }
  }
  if (elements.probabilitiesSummary) {
    elements.probabilitiesSummary.textContent = formatCurrentPredictedSummary(
      formatCurrentClassText(heatmap, metrics),
      predictedClass && predictedClass > 0
        ? `Pred: C${predictedClass}${predictedConfidence !== null ? ` (${formatPercent(predictedConfidence)})` : ""}`
        : "Pred: none",
    );
  }
  host.style.height = `${Math.max(620, Math.round((136 + (visibleRows.length * 58)) * 1.55))}px`;
  const chart = ensureChart(host, "probabilities");
  if (!chart) {
    clearChartHost(host, "probabilities", "probability chart unavailable");
    return;
  }
  const mismatchPrediction = currentClass && predictedClass && predictedClass > 0 && predictedClass !== currentClass;
  const classRowLookup = buildClassRowLookup(dashboard);
  const yLabels = [];
  const values = [];
  const colors = [];
  const borders = [];
  for (const row of visibleRows) {
    let fill = "#fb923c";
    let edge = "#c2410c";
    if (row.classIndex > 0 && currentClass && row.classIndex === currentClass) {
      fill = "#4ade80";
      edge = "#15803d";
    } else if (row.classIndex > 0 && mismatchPrediction && row.classIndex === predictedClass) {
      fill = "#f87171";
      edge = "#b91c1c";
    }
    yLabels.push(formatProbabilityAxisLabel(row.classIndex, classRowLookup.get(row.classIndex)));
    values.push(row.probability);
    colors.push(fill);
    borders.push(edge);
  }
  chart.setOption({
    animation: false,
    backgroundColor: "transparent",
    grid: { left: 0, right: 28, top: 18, bottom: 34, containLabel: true },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "rgba(9, 12, 17, 0.96)",
      borderColor: "rgba(223, 211, 186, 0.18)",
      textStyle: { color: "#f2ede2", fontFamily: "Cascadia Code, Consolas, monospace", fontSize: 12.5 },
      formatter: (items) => {
        const item = items[0];
        return `${escapeHtml(item.axisValueLabel)}<br/>${escapeHtml(formatPercent(item.value))}`;
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: {
        color: "#c5bcae",
        fontSize: 13,
        fontFamily: "Cascadia Code, Consolas, monospace",
        formatter: (value) => `${Math.round(value * 100)}%`,
      },
      axisLine: { lineStyle: { color: "rgba(223, 211, 186, 0.34)" } },
      splitLine: { lineStyle: { color: "rgba(223, 211, 186, 0.1)", type: "dashed" } },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: yLabels,
      axisLabel: {
        color: "#ece6db",
        fontSize: 13.5,
        inside: false,
        margin: 4,
        align: "right",
        verticalAlign: "middle",
        overflow: "truncate",
        fontWeight: "700",
        lineHeight: 17,
        fontFamily: "Cascadia Code, Consolas, monospace",
        formatter: (value) => {
          const [classLabel, versionLabel, groupLabel] = String(value || "").split("\n", 3);
          if (!versionLabel && !groupLabel) {
            return `{classLabel|${classLabel || "-"}}`;
          }
          if (!groupLabel) {
            return `{classLabel|${classLabel}}\n{groupLabel|${versionLabel}}`;
          }
          return `{classLabel|${classLabel}}\n{versionLabel|${versionLabel}}\n{groupLabel|${groupLabel}}`;
        },
        rich: {
          classLabel: {
            color: "#f5efe4",
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontSize: 13.5,
            fontWeight: "700",
            lineHeight: 17,
            align: "right",
          },
          versionLabel: {
            color: "#d8d0c3",
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontSize: 11.5,
            fontWeight: "600",
            lineHeight: 14,
            align: "right",
          },
          groupLabel: {
            color: "#b8b0a4",
            fontFamily: "Cascadia Code, Consolas, monospace",
            fontSize: 11.5,
            fontWeight: "600",
            lineHeight: 14,
            align: "right",
          },
        },
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [{
      type: "bar",
      data: values.map((value, index) => ({
        value,
        itemStyle: {
          color: colors[index],
          borderColor: borders[index],
          borderWidth: 1.1,
          borderRadius: [0, 10, 10, 0],
        },
        label: {
          show: true,
          position: value > 0.12 ? "insideRight" : "right",
          formatter: () => formatPercent(value),
          color: value > 0.12 ? "#ffffff" : "#ece6db",
          fontWeight: "bold",
          fontFamily: "Cascadia Code, Consolas, monospace",
          fontSize: 13.5,
        },
      })),
      barWidth: "68%",
      barCategoryGap: "24%",
    }],
  }, true);
  chart.resize();
}

function lineChartSignature(panelSpec, history, revisionSource = null) {
  void history;
  return stableSignature({
    panelSpec: panelSpec || null,
    revision: dashboardRevisionToken(revisionSource || history),
  });
}

function hudSignature(dashboard) {
  return stableSignature({
    revision: dashboardRevisionToken(dashboard),
  });
}

function projectionSignature(runPayload) {
  return stableSignature({
    revision: dashboardRevisionToken(runPayload?.dashboard, runPayload?.runId),
  });
}

function probabilitiesSignature(runPayload) {
  return stableSignature({
    revision: dashboardRevisionToken(runPayload?.dashboard, runPayload?.runId),
  });
}

function buildConfigSnapshotRenderSnapshot(summary, runId) {
  return {
    files: normalizeConfigSnapshotFiles(summary).map((entry) => ({
      name: entry.name,
      label: entry.label,
      content: entry.content,
    })),
    open: activeConfigSnapshotVisibility(runId),
  };
}

function buildClassRowsRenderSnapshot(rows) {
  if (!Array.isArray(rows)) return null;
  const cached = classRowsRenderSnapshotCache.get(rows);
  if (cached) {
    return cached;
  }
  const snapshot = rows.map((row) => ({
    displayClass: row?.is_new_dynamics_class ? "NEW" : formatClassRowFocusLabel(row),
    displayId: formatClassRowDisplayLabel(row),
    state: resolvePrefixStateLabel(row),
    flags: buildPrefixFlagText(row),
    transitions: formatCountLabel(row?.transition_count ?? 0),
    isCurrent: Boolean(row?.is_current || row?.is_latest_active),
    isExplainer: Boolean(row?.is_explainer),
    isActive: Boolean(row?.is_active),
  }));
  classRowsRenderSnapshotCache.set(rows, snapshot);
  return snapshot;
}

function buildProjectionLegendSummary(runPayload) {
  const heatmap = runPayload?.dashboard?.visitationHeatmap;
  if (!heatmap || typeof heatmap !== "object") {
    return {
      sampledPointCountByWorldIndex: new Map(),
      displayedPointCountByClassIndex: new Map(),
      displayedClassCountByWorldIndex: new Map(),
    };
  }
  const cached = projectionLegendSummaryCache.get(heatmap);
  if (cached) {
    return cached;
  }
  const sampledPointCountByWorldIndex = new Map();
  const displayedPointCountByClassIndex = new Map();
  const displayedClassCountByWorldIndex = new Map();
  const transitions = Array.isArray(heatmap.transitions) ? heatmap.transitions : [];
  for (const transition of transitions) {
    const classIndex = optionalInteger(transition?.class_index) ?? 0;
    const worldIndex = normalizeTransitionWorldIndex(transition?.world_index ?? transition?.worldIndex);
    if (worldIndex === null) continue;

    sampledPointCountByWorldIndex.set(
      worldIndex,
      (sampledPointCountByWorldIndex.get(worldIndex) || 0) + 1,
    );

    let worldCountMap = displayedPointCountByClassIndex.get(classIndex);
    if (!worldCountMap) {
      worldCountMap = new Map();
      displayedPointCountByClassIndex.set(classIndex, worldCountMap);
    }
    worldCountMap.set(
      worldIndex,
      (worldCountMap.get(worldIndex) || 0) + 1,
    );

    let classCountMap = displayedClassCountByWorldIndex.get(worldIndex);
    if (!classCountMap) {
      classCountMap = new Map();
      displayedClassCountByWorldIndex.set(worldIndex, classCountMap);
    }
    classCountMap.set(
      classIndex,
      (classCountMap.get(classIndex) || 0) + 1,
    );
  }
  const summary = {
    sampledPointCountByWorldIndex,
    displayedPointCountByClassIndex,
    displayedClassCountByWorldIndex,
  };
  projectionLegendSummaryCache.set(heatmap, summary);
  return summary;
}

function buildProjectionWorldCountMap(
  runPayload,
  activeClass = projectionActiveFocusClass(runPayload),
) {
  const summary = buildProjectionLegendSummary(runPayload);
  if (activeClass === null) {
    return summary.sampledPointCountByWorldIndex;
  }
  return summary.displayedPointCountByClassIndex.get(activeClass) || new Map();
}

function buildProjectionClassCountMap(
  runPayload,
  activeWorldIndex = projectionActiveFocusWorld(runPayload),
) {
  const summary = buildProjectionLegendSummary(runPayload);
  if (activeWorldIndex === null) {
    return new Map();
  }
  return summary.displayedClassCountByWorldIndex.get(activeWorldIndex) || new Map();
}

function buildPrototypeLegendEntries(runPayload) {
  const dashboard = runPayload?.dashboard || {};
  const rows = Array.isArray(dashboard?.classRows) ? dashboard.classRows : [];
  const projection = dashboard?.visitationHeatmap || {};
  const activeWorldIndex = projectionActiveFocusWorld(runPayload);
  const visibleClassCountByIndex = buildProjectionClassCountMap(runPayload, activeWorldIndex);
  const newDynamicsRow = rows.find((row) => row?.is_new_dynamics_class) || null;
  const baseQuestionMarkCount = Math.max(
    0,
    optionalInteger(projection?.current_collect_question_mark_count)
      ?? optionalInteger(newDynamicsRow?.transition_count)
      ?? 0,
  );
  const questionMarkCount = activeWorldIndex !== null
    ? visibleClassCountByIndex.get(0) || 0
    : baseQuestionMarkCount;
  return {
    rows: rows
      .map((row) => {
        const classIndex = optionalInteger(row?.version_index);
        const displayCount = Math.max(0, optionalInteger(row?.transition_count) ?? 0);
        return {
          ...row,
          classIndex,
          displayCount: activeWorldIndex !== null
            ? visibleClassCountByIndex.get(classIndex) || 0
            : displayCount,
        };
      })
      .filter((row) => row.classIndex !== null && row.classIndex > 0)
      .sort(comparePrototypeLegendRows),
    questionMarkCount,
    hasQuestionMarkRow: baseQuestionMarkCount > 0,
  };
}

function buildPrototypeLegendSnapshot(runPayload) {
  const { rows, questionMarkCount, hasQuestionMarkRow } = buildPrototypeLegendEntries(runPayload);
  const snapshot = {
    rows: rows
      .map((row) => ({
        classIndex: row.classIndex,
        label: compactPrefixIdentityLabel(String(row?.display_id || row?.version_id || "-")),
        count: row.displayCount,
        isCurrent: Boolean(row?.is_current || row?.is_latest_active),
      }))
      .sort((left, right) => Number(left.classIndex) - Number(right.classIndex)),
    questionMarkCount,
    hasQuestionMarkRow,
  };
  return snapshot;
}

function buildProjectionTargetWorldLegendEntries(runPayload) {
  const targetWorlds = normalizeTargetWorlds(runPayload);
  if (!targetWorlds.length) {
    return [];
  }
  const sampledPointCountByWorldIndex = buildProjectionWorldCountMap(runPayload, null);
  const displayedPointCountByWorldIndex = buildProjectionWorldCountMap(
    runPayload,
    projectionActiveFocusClass(runPayload),
  );
  return targetWorlds.map((entry) => ({
    ...entry,
    sampledPointCount: sampledPointCountByWorldIndex.get(entry.worldIndex) || 0,
    displayedCount: displayedPointCountByWorldIndex.get(entry.worldIndex) || 0,
  }));
}

function buildProjectionTargetWorldLegendSnapshot(runPayload) {
  const targetWorlds = buildProjectionTargetWorldLegendEntries(runPayload);
  return {
    worlds: targetWorlds.map((entry) => ({
      worldIndex: entry.worldIndex,
      mapName: entry.mapName,
      worldLabel: entry.worldLabel,
      sampledPointCount: entry.sampledPointCount,
      displayedCount: entry.displayedCount,
      transitionCount: entry.transitionCount,
      isCurrent: entry.isCurrent,
      isActive: entry.isActive,
    })),
    selectedWorldIndex: projectionPinnedWorld(runPayload),
    panelExpanded: isProjectionTargetWorldPanelExpanded(runPayload),
  };
}

function buildTargetWorldsRenderSnapshot(runPayload) {
  const targetWorlds = normalizeTargetWorlds(runPayload);
  const selected = resolveSelectedTargetWorld(runPayload, targetWorlds);
  const panelExpanded = isTargetWorldPanelExpanded(runPayload);
  return {
    worlds: targetWorlds.map((entry) => ({
      worldIndex: entry.worldIndex,
      worldSeed: entry.worldSeed,
      mapName: entry.mapName,
      worldLabel: entry.worldLabel,
      scenarioType: entry.scenarioType,
      transitionCount: entry.transitionCount,
      isActive: entry.isActive,
      isCurrent: entry.isCurrent,
      resumePending: entry.resumePending,
      previewStateJson: entry.previewStateJson,
    })),
    selectedWorldIndex: selected?.worldIndex ?? null,
    panelExpanded,
    host: panelExpanded ? hostSizeSignature(elements.targetWorldBoard) : "collapsed",
  };
}

function formatTargetWorldButtonLabel(targetWorld, transitionCountLabel = null) {
  return (
    `W${targetWorld.worldIndex} · ${targetWorld.mapName || targetWorld.worldLabel}`
    + ` · ${transitionCountLabel ?? formatCountLabel(targetWorld.transitionCount)}`
  );
}

function syncTargetWorldButtons(runPayload, targetWorlds, selectedWorld, buttonLabelsByWorldIndex = null) {
  const existingButtons = new Map(
    Array.from(
      elements.targetWorldList.querySelectorAll("[data-discovery-action='select-target-world']"),
    ).map((button) => [String(button.dataset.worldIndex || ""), button]),
  );
  const activeWorldIndices = new Set();
  const liveCurrentWorldIndex = resolveLiveCurrentTargetWorldIndex(runPayload, targetWorlds);

  targetWorlds.forEach((targetWorld, index) => {
    const worldIndexText = String(targetWorld.worldIndex);
    activeWorldIndices.add(worldIndexText);
    let button = existingButtons.get(worldIndexText) || null;
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.dataset.discoveryAction = "select-target-world";
      button.dataset.worldIndex = worldIndexText;
    }

    button.className = "discovery-target-world-button";
    const isCurrent = (
      liveCurrentWorldIndex !== null
        ? targetWorld.worldIndex === liveCurrentWorldIndex
        : Boolean(targetWorld.isCurrent)
    );
    button.classList.toggle(
      "is-selected",
      Boolean(selectedWorld && targetWorld.worldIndex === selectedWorld.worldIndex),
    );
    button.classList.toggle("is-current", isCurrent);
    button.classList.toggle("is-inactive", !targetWorld.isActive && !isCurrent);

    const nextLabel = buttonLabelsByWorldIndex?.get(targetWorld.worldIndex)
      ?? formatTargetWorldButtonLabel(targetWorld);
    if (button.textContent !== nextLabel) {
      button.textContent = nextLabel;
    }

    const currentButtonAtIndex = elements.targetWorldList.children[index] || null;
    if (currentButtonAtIndex !== button) {
      elements.targetWorldList.insertBefore(button, currentButtonAtIndex);
    }
  });

  Array.from(elements.targetWorldList.querySelectorAll("[data-discovery-action='select-target-world']"))
    .forEach((button) => {
      if (!activeWorldIndices.has(String(button.dataset.worldIndex || ""))) {
        button.remove();
      }
    });
}

function resolvePrefixStateLabel(row) {
  if (row?.is_new_dynamics_class) return "NEW";
  const predictionStatus = String(row?.prediction_status || "").trim().toLowerCase();
  const marker = String(row?.marker || "").trim();
  if (predictionStatus === "missing") return "MISS";
  if (predictionStatus === "error") return "ERR";
  if (marker === "O") return "OK";
  if (marker === "X") return "FAIL";
  return "WAIT";
}

function buildPrefixFlagText(row) {
  const currentFlag = row?.is_current || row?.is_latest_active ? "CUR" : "-";
  const explainerFlag = row?.is_explainer ? "EXP" : "-";
  return `${currentFlag} ${explainerFlag}`;
}

function compactPrefixIdentityLabel(label) {
  const text = String(label || "").trim();
  if (!text) return "-";
  if (!text.includes(":g")) return text;
  const [prefix, groupSuffix] = text.split(":g", 2);
  const pieces = prefix.split(/[:/_-]+/).filter(Boolean);
  const compact = (pieces[pieces.length - 1] || prefix).slice(-10);
  return `${compact}:g${groupSuffix}`;
}

function renderTargetWorldPanel(runPayload) {
  if (
    !elements.targetWorlds
    || !elements.targetWorldsHead
    || !elements.targetWorldsTitle
    || !elements.targetWorldBody
    || !elements.targetWorldList
    || !elements.targetWorldBoard
    || !elements.targetWorldMeta
  ) {
    return;
  }

  const targetWorlds = normalizeTargetWorlds(runPayload);
  if (!targetWorlds.length) {
    elements.targetWorldsHead.classList.remove("is-open");
    elements.targetWorldsHead.setAttribute("aria-expanded", "false");
    setElementHidden(elements.targetWorldsTitle, true);
    elements.targetWorldsTitle.textContent = "Target Maps";
    setElementHidden(elements.targetWorlds, true);
    elements.targetWorldList.innerHTML = "";
    if (elements.targetWorldMeta) {
      elements.targetWorldMeta.textContent = "";
      setElementHidden(elements.targetWorldMeta, true);
    }
    setElementHidden(elements.targetWorldBody, true);
    renderBoard(null, elements.targetWorldBoard, { emptyMessage: "target map preview unavailable" });
    return;
  }

  const panelExpanded = isTargetWorldPanelExpanded(runPayload);
  setElementHidden(elements.targetWorlds, false);
  setElementHidden(elements.targetWorldsTitle, false);
  elements.targetWorldsTitle.textContent = `Target Maps (${targetWorlds.length})`;
  elements.targetWorldsHead.classList.toggle("is-open", panelExpanded);
  elements.targetWorldsHead.setAttribute("aria-expanded", panelExpanded ? "true" : "false");
  setElementHidden(elements.targetWorldBody, !panelExpanded);
  if (!panelExpanded) {
    return;
  }

  const selectedWorld = resolveSelectedTargetWorld(runPayload, targetWorlds);
  const targetWorldButtonLabels = new Map(
    targetWorlds.map((targetWorld) => [
      targetWorld.worldIndex,
      formatTargetWorldButtonLabel(
        targetWorld,
        formatCountLabel(targetWorld.transitionCount),
      ),
    ]),
  );
  syncTargetWorldButtons(runPayload, targetWorlds, selectedWorld, targetWorldButtonLabels);
  if (elements.targetWorldMeta) {
    setElementHidden(elements.targetWorldMeta, true);
    elements.targetWorldMeta.textContent = "";
  }

  const previewScene = buildStateScene(
    parseSerializedStatePayload(selectedWorld?.previewStateJson || null),
    runPayload?.dashboard?.visualConfig || null,
  );
  renderBoard(previewScene, elements.targetWorldBoard, {
    tileSize: 40,
    fitToHost: true,
    minTileSize: 20,
    maxTileSize: 48,
    hostPadding: 16,
    fallbackHostHeight: Math.min(window.innerHeight * 0.5, 520),
    emptyMessage: "target map preview unavailable",
  });
}

function formatClassRowFocusLabel(row) {
  const classIndex = optionalInteger(row?.version_index);
  if (classIndex !== null && classIndex > 0) {
    return `C${classIndex}`;
  }
  return compactPrefixIdentityLabel(String(row?.display_id || row?.version_id || "-")).slice(0, 12) || "-";
}

function formatClassRowDisplayLabel(row) {
  return compactPrefixIdentityLabel(String(row?.display_id || row?.version_id || "-")) || "-";
}

function appendClassRowCell(container, text, className = "") {
  const cell = document.createElement("span");
  cell.className = `class-row-cell${className ? ` ${className}` : ""}`;
  cell.textContent = text;
  container.append(cell);
}

function buildClassRowSummaryEntries(rows) {
  const currentRow = rows.find((row) => row?.is_current || row?.is_latest_active) || null;
  const activeCount = rows.reduce((total, row) => total + (row?.is_active ? 1 : 0), 0);
  const newCount = rows.reduce((total, row) => total + (row?.is_new_dynamics_class ? 1 : 0), 0);
  const entries = [
    { label: "TOTAL", value: formatCountLabel(rows.length) },
  ];
  if (currentRow) {
    entries.push({ label: "CURRENT", value: formatClassRowFocusLabel(currentRow) });
  }
  if (activeCount > 0) {
    entries.push({ label: "ACTIVE", value: formatCountLabel(activeCount) });
  }
  if (newCount > 0) {
    entries.push({ label: "NEW", value: formatCountLabel(newCount) });
  }
  return entries;
}

function createClassRowItem(row) {
  const stateLabel = resolvePrefixStateLabel(row);
  const item = document.createElement("div");
  item.className = `class-row class-row-data class-row-virtual-item ${PREFIX_STATE_STYLE[stateLabel] || "state-wait"}`;
  if (row?.is_current || row?.is_latest_active) item.classList.add("is-current");
  if (row?.is_explainer) item.classList.add("is-explainer");
  if (row?.is_active) item.classList.add("is-active");
  item.title = formatLines([
    `class: ${formatClassRowFocusLabel(row)}`,
    `identity: ${String(row?.display_id || row?.version_id || "-")}`,
    `state: ${stateLabel}`,
    `flags: ${buildPrefixFlagText(row)}`,
    `transitions: ${formatCountLabel(row?.transition_count ?? 0)}`,
  ]);
  const grid = document.createElement("div");
  grid.className = "class-row-grid";
  appendClassRowCell(
    grid,
    row?.is_new_dynamics_class ? "NEW" : formatClassRowFocusLabel(row),
    "class-row-cell-index",
  );
  appendClassRowCell(grid, formatClassRowDisplayLabel(row), "class-row-cell-label");
  appendClassRowCell(grid, stateLabel, "class-row-cell-state");
  appendClassRowCell(grid, buildPrefixFlagText(row), "class-row-cell-flags");
  appendClassRowCell(grid, formatCountLabel(row?.transition_count ?? 0), "class-row-cell-count");
  item.append(grid);
  return item;
}

function mountVirtualClassRows(scroller, surface, rows, runKey) {
  let frameHandle = null;
  const totalHeight = Math.max(0, (rows.length * CLASS_ROW_STRIDE) - CLASS_ROW_GAP);
  surface.style.height = `${totalHeight}px`;
  const viewport = scroller?.parentElement || null;

  const syncHeaderScrollbarGutter = () => {
    if (!viewport) return;
    const gutterWidth = Math.max(
      0,
      Number(scroller.offsetWidth || 0) - Number(scroller.clientWidth || 0),
    );
    viewport.style.setProperty("--class-row-scrollbar-gutter", `${gutterWidth}px`);
  };

  const renderVisibleRows = () => {
    frameHandle = null;
    syncHeaderScrollbarGutter();
    const viewportHeight = Math.max(scroller.clientHeight, CLASS_ROW_HEIGHT);
    const startIndex = Math.max(0, Math.floor(scroller.scrollTop / CLASS_ROW_STRIDE) - CLASS_ROW_OVERSCAN);
    const visibleCount = Math.ceil(viewportHeight / CLASS_ROW_STRIDE) + (CLASS_ROW_OVERSCAN * 2);
    const endIndex = Math.min(rows.length, startIndex + visibleCount);
    const fragment = document.createDocumentFragment();
    surface.innerHTML = "";
    for (let index = startIndex; index < endIndex; index += 1) {
      const item = createClassRowItem(rows[index]);
      item.style.top = `${index * CLASS_ROW_STRIDE}px`;
      fragment.append(item);
    }
    surface.append(fragment);
  };

  const scheduleVisibleRowsRender = () => {
    if (frameHandle !== null) return;
    frameHandle = window.requestAnimationFrame(renderVisibleRows);
  };

  scroller.addEventListener("scroll", () => {
    state.classRowsScrollTopByRun[runKey] = scroller.scrollTop;
    scheduleVisibleRowsRender();
  }, { passive: true });

  const maxScrollTop = Math.max(0, totalHeight - scroller.clientHeight);
  const savedScrollTop = clamp(state.classRowsScrollTopByRun[runKey] ?? 0, 0, maxScrollTop);
  state.classRowsScrollTopByRun[runKey] = savedScrollTop;
  scroller.scrollTop = savedScrollTop;
  syncHeaderScrollbarGutter();
  renderVisibleRows();
}

function renderPrototypeLegend(runPayload) {
  if (!elements.prototypeLegend) return;
  elements.prototypeLegend.innerHTML = "";
  const { rows, questionMarkCount, hasQuestionMarkRow } = buildPrototypeLegendEntries(runPayload);
  if (!rows.length && !hasQuestionMarkRow) {
    return;
  }

  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "prototype-legend-item";
    if (row.is_current || row.is_latest_active) {
      item.classList.add("is-current");
    }
    item.title = String(row.display_id || row.version_id || `C${row.classIndex}`);
    item.dataset.classIndex = String(row.classIndex);
    attachPrototypeLegendSelectionInteractions(item, runPayload, row.classIndex);

    const swatch = document.createElement("span");
    swatch.className = "prototype-legend-swatch";
    const color = classColor(row.classIndex);
    swatch.style.backgroundColor = color;
    swatch.style.boxShadow = `inset 0 0 0 1px ${color}`;

    const text = document.createElement("span");
    text.className = "prototype-legend-text";
    text.textContent = `C${row.classIndex} · ${compactPrefixIdentityLabel(String(row.display_id || row.version_id || `C${row.classIndex}`))}`;

    const count = document.createElement("span");
    count.className = "prototype-legend-count";
    count.textContent = String(row.displayCount);

    item.append(swatch, text, count);
    elements.prototypeLegend.append(item);
  }

  if (hasQuestionMarkRow) {
    const item = document.createElement("div");
    item.className = "prototype-legend-item";
    item.title = "new dynamics class! (? collected in current collect loop)";
    item.dataset.classIndex = "0";
    attachPrototypeLegendSelectionInteractions(item, runPayload, 0);

    const swatch = document.createElement("span");
    swatch.className = "prototype-legend-swatch";
    const color = classColor(0);
    swatch.style.backgroundColor = color;
    swatch.style.boxShadow = `inset 0 0 0 1px ${color}`;

    const text = document.createElement("span");
    text.className = "prototype-legend-text";
    text.textContent = "NEW · ? in collect";

    const count = document.createElement("span");
    count.className = "prototype-legend-count";
    count.textContent = String(questionMarkCount);

    item.append(swatch, text, count);
    elements.prototypeLegend.append(item);
  }
  syncPrototypeLegendSelectionState(runPayload);
}

function clearProjectionTargetWorldLegend() {
  if (elements.projectionTargetWorldsHead) {
    elements.projectionTargetWorldsHead.classList.remove("is-open");
    elements.projectionTargetWorldsHead.setAttribute("aria-expanded", "false");
  }
  if (elements.projectionTargetWorldsTitle) {
    elements.projectionTargetWorldsTitle.textContent = "Target Worlds";
  }
  setElementHidden(elements.projectionTargetWorldBody, true);
  if (elements.projectionTargetWorldList) {
    elements.projectionTargetWorldList.innerHTML = "";
  }
  setElementHidden(elements.projectionTargetWorlds, true);
}

function projectionTargetWorldButtonLabel(targetWorld) {
  return (
    `W${targetWorld.worldIndex} · ${targetWorld.mapName || targetWorld.worldLabel}`
    + ` · ${formatCountLabel(targetWorld.displayedCount)}`
  );
}

function projectionTargetWorldButtonTitle(targetWorld) {
  const details = [];
  if (targetWorld.displayedCount !== targetWorld.sampledPointCount) {
    details.push(`${formatCountLabel(targetWorld.displayedCount)} matching points`);
  }
  details.push(`${formatCountLabel(targetWorld.transitionCount)} transitions total`);
  return (
    `${targetWorld.mapName || targetWorld.worldLabel || `world ${targetWorld.worldIndex}`}`
    + ` (${details.join(", ")})`
  );
}

function syncProjectionTargetWorldButtons(targetWorlds) {
  if (!elements.projectionTargetWorldList) return;
  const host = elements.projectionTargetWorldList;
  const existingButtons = new Map(
    Array.from(host.querySelectorAll(".projection-target-world-chip[data-world-index]"))
      .map((button) => [String(button.dataset.worldIndex || ""), button]),
  );

  for (let index = 0; index < targetWorlds.length; index += 1) {
    const targetWorld = targetWorlds[index];
    const worldKey = String(targetWorld.worldIndex);
    let button = existingButtons.get(worldKey) || null;
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.dataset.discoveryAction = "toggle-projection-target-world";
      button.dataset.worldIndex = worldKey;
      button.setAttribute("aria-pressed", "false");
    }
    existingButtons.delete(worldKey);

    button.className = "discovery-target-world-button projection-target-world-chip";
    button.dataset.worldIndex = worldKey;
    const nextText = projectionTargetWorldButtonLabel(targetWorld);
    if (button.textContent !== nextText) {
      button.textContent = nextText;
    }
    const nextTitle = projectionTargetWorldButtonTitle(targetWorld);
    if (button.title !== nextTitle) {
      button.title = nextTitle;
    }
    if (targetWorld.isCurrent) {
      button.classList.add("is-current");
    }
    const shouldDisable = targetWorld.displayedCount <= 0;
    if (!targetWorld.isActive || shouldDisable) {
      button.classList.add("is-inactive");
    }
    if (button.disabled !== shouldDisable) {
      button.disabled = shouldDisable;
    }

    const currentChild = host.children[index] || null;
    if (currentChild !== button) {
      host.insertBefore(button, currentChild);
    }
  }

  for (const staleButton of existingButtons.values()) {
    staleButton.remove();
  }
}

function renderProjectionTargetWorldLegend(runPayload) {
  if (
    !elements.projectionTargetWorlds
    || !elements.projectionTargetWorldsHead
    || !elements.projectionTargetWorldsTitle
    || !elements.projectionTargetWorldBody
    || !elements.projectionTargetWorldList
  ) {
    return;
  }
  const targetWorlds = buildProjectionTargetWorldLegendEntries(runPayload);
  if (!targetWorlds.length) {
    clearProjectionTargetWorldLegend();
    return;
  }

  const selectableWorldIndices = new Set(
    targetWorlds
      .filter((entry) => entry.displayedCount > 0)
      .map((entry) => entry.worldIndex),
  );
  const pinnedWorldIndex = projectionPinnedWorld(runPayload);
  if (pinnedWorldIndex !== null && !selectableWorldIndices.has(pinnedWorldIndex)) {
    delete state.projectionPinnedWorldByRun[projectionSelectionRunKey(runPayload)];
  }

  const panelExpanded = isProjectionTargetWorldPanelExpanded(runPayload);
  setElementHidden(elements.projectionTargetWorlds, false);
  elements.projectionTargetWorldsTitle.textContent = `Target Worlds (${targetWorlds.length})`;
  elements.projectionTargetWorldsHead.classList.toggle("is-open", panelExpanded);
  elements.projectionTargetWorldsHead.setAttribute("aria-expanded", panelExpanded ? "true" : "false");
  setElementHidden(elements.projectionTargetWorldBody, !panelExpanded);
  if (!panelExpanded) {
    return;
  }
  syncProjectionTargetWorldButtons(targetWorlds);
  syncProjectionTargetWorldSelectionState(runPayload);
}

function renderClassRows(runPayload) {
  elements.classRows.innerHTML = "";
  const rows = runPayload?.dashboard?.classRows || [];
  if (!Array.isArray(rows) || !rows.length) {
    elements.classRows.innerHTML = `<div class="chart-empty">No dynamics class yet</div>`;
    return;
  }

  const runKey = ensureClassRowsScrollTop(classRowsScrollRunKey(runPayload));
  const summary = document.createElement("div");
  summary.className = "class-row-summary";
  for (const entry of buildClassRowSummaryEntries(rows)) {
    const chip = document.createElement("div");
    chip.className = "class-row-summary-chip";

    const label = document.createElement("span");
    label.className = "class-row-summary-label";
    label.textContent = entry.label;

    const value = document.createElement("span");
    value.className = "class-row-summary-value";
    value.textContent = entry.value;

    chip.append(label, value);
    summary.append(chip);
  }

  const viewport = document.createElement("div");
  viewport.className = "class-row-viewport";

  const header = document.createElement("div");
  header.className = "class-row class-row-header class-row-grid";
  appendClassRowCell(header, "CLASS", "class-row-cell-index");
  appendClassRowCell(header, "DYNAMICS", "class-row-cell-label");
  appendClassRowCell(header, "STATE", "class-row-cell-state");
  appendClassRowCell(header, "FLAGS", "class-row-cell-flags");
  appendClassRowCell(header, "TRANSITIONS", "class-row-cell-count");

  const scroller = document.createElement("div");
  scroller.className = "class-row-scroller";

  const surface = document.createElement("div");
  surface.className = "class-row-virtual-surface";
  scroller.append(surface);

  viewport.append(header, scroller);
  elements.classRows.append(summary, viewport);
  mountVirtualClassRows(scroller, surface, rows, runKey);
}

function clearTransitionNavigator() {
  if (elements.versionSummary) {
    elements.versionSummary.textContent = "";
  }
  if (elements.versionList) {
    elements.versionList.innerHTML = "";
  }
}

function setViewerStageMode(mode) {
  const isTransitionMode = mode === "transition";
  elements.boardWrap?.classList.toggle("is-transition-mode", isTransitionMode);
  if (elements.transitionStage) {
    elements.transitionStage.hidden = !isTransitionMode;
  }
}

function clearTransitionStage() {
  setViewerStageMode("live");
  if (elements.transitionToolbar) {
    elements.transitionToolbar.innerHTML = "";
  }
  if (elements.transitionPreviousMeta) {
    elements.transitionPreviousMeta.innerHTML = "";
  }
  if (elements.transitionNextMeta) {
    elements.transitionNextMeta.innerHTML = "";
  }
  if (elements.transitionWitnessStrip) {
    elements.transitionWitnessStrip.innerHTML = "";
    elements.transitionWitnessStrip.hidden = true;
  }
  if (elements.transitionNextLabel) {
    elements.transitionNextLabel.textContent = "Expected Next State";
  }
  elements.transitionNextPanel?.classList.remove(
    "is-variant-expected",
    "is-variant-success",
    "is-variant-fail",
  );
  renderTransitionImageFrame(
    elements.transitionPreviousFrame,
    null,
    "Select a version block to inspect a stored transition",
  );
  renderTransitionImageFrame(
    elements.transitionNextFrame,
    null,
    "Choose Expected, Success, or Fail n after selecting a version",
  );
}

function sourceLineCount(source) {
  if (typeof source !== "string" || !source.length) return 0;
  const parts = source.split("\n");
  if (parts[parts.length - 1] === "") {
    parts.pop();
  }
  return parts.length;
}

function resetProgramCopyButton() {
  if (state.programCopyResetHandle) {
    window.clearTimeout(state.programCopyResetHandle);
    state.programCopyResetHandle = null;
  }
  if (!elements.programCopy) return;
  elements.programCopy.classList.remove("is-copied", "is-copy-failed");
  const copyLabel = state.programInspectorText ? "Copy source" : "No source to copy";
  elements.programCopy.title = copyLabel;
  elements.programCopy.setAttribute("aria-label", copyLabel);
}

function setProgramCopyButton(sourceText) {
  state.programInspectorText = typeof sourceText === "string" ? sourceText : null;
  resetProgramCopyButton();
  if (!elements.programCopy) return;
  const hasSourceText = Boolean(state.programInspectorText);
  elements.programCopy.disabled = !hasSourceText;
  const copyLabel = hasSourceText ? "Copy source" : "No source to copy";
  elements.programCopy.title = copyLabel;
  elements.programCopy.setAttribute("aria-label", copyLabel);
}

function highlightPythonCodeSegment(segment) {
  if (!segment) return "";
  let html = "";
  let lastIndex = 0;
  for (const match of segment.matchAll(PYTHON_TOKEN_RE)) {
    const token = match[0];
    const startIndex = match.index ?? 0;
    html += escapeHtml(segment.slice(lastIndex, startIndex));
    let className = "py-builtin";
    if (token.startsWith("@")) {
      className = "py-decorator";
    } else if (PYTHON_KEYWORDS.has(token)) {
      className = "py-keyword";
    } else if (!Number.isNaN(Number(token.replaceAll("_", ""))) || /^0[xX]/.test(token)) {
      className = "py-number";
    }
    html += `<span class="${className}">${escapeHtml(token)}</span>`;
    lastIndex = startIndex + token.length;
  }
  html += escapeHtml(segment.slice(lastIndex));
  return html;
}

function highlightPythonSource(source) {
  const normalizedSource = normalizeSourceText(source) || "";
  const lines = [[]];
  let codeBuffer = "";
  let specialBuffer = "";
  let mode = "code";
  let quoteChar = null;
  let tripleQuoted = false;
  let escapeNext = false;

  function currentLine() {
    return lines[lines.length - 1];
  }

  function flushCodeBuffer() {
    if (!codeBuffer) return;
    currentLine().push(highlightPythonCodeSegment(codeBuffer));
    codeBuffer = "";
  }

  function flushSpecialBuffer(className) {
    if (!specialBuffer) return;
    currentLine().push(`<span class="${className}">${escapeHtml(specialBuffer)}</span>`);
    specialBuffer = "";
  }

  for (let index = 0; index < normalizedSource.length; index += 1) {
    const char = normalizedSource[index];
    if (mode === "code") {
      const tripleCandidate = normalizedSource.slice(index, index + 3);
      if (char === "#") {
        flushCodeBuffer();
        mode = "comment";
        specialBuffer = "#";
        continue;
      }
      if (tripleCandidate === "'''" || tripleCandidate === "\"\"\"") {
        flushCodeBuffer();
        mode = "string";
        quoteChar = char;
        tripleQuoted = true;
        specialBuffer = tripleCandidate;
        index += 2;
        escapeNext = false;
        continue;
      }
      if (char === "'" || char === "\"") {
        flushCodeBuffer();
        mode = "string";
        quoteChar = char;
        tripleQuoted = false;
        specialBuffer = char;
        escapeNext = false;
        continue;
      }
      if (char === "\n") {
        flushCodeBuffer();
        lines.push([]);
        continue;
      }
      codeBuffer += char;
      continue;
    }

    if (mode === "comment") {
      if (char === "\n") {
        flushSpecialBuffer("py-comment");
        mode = "code";
        lines.push([]);
        continue;
      }
      specialBuffer += char;
      continue;
    }

    if (char === "\n") {
      flushSpecialBuffer("py-string");
      if (!tripleQuoted) {
        mode = "code";
        quoteChar = null;
      }
      lines.push([]);
      escapeNext = false;
      continue;
    }

    if (!tripleQuoted && escapeNext) {
      specialBuffer += char;
      escapeNext = false;
      continue;
    }

    if (!tripleQuoted && char === "\\") {
      specialBuffer += char;
      escapeNext = true;
      continue;
    }

    if (tripleQuoted) {
      const tripleCandidate = normalizedSource.slice(index, index + 3);
      if (quoteChar && tripleCandidate === quoteChar.repeat(3)) {
        specialBuffer += tripleCandidate;
        index += 2;
        flushSpecialBuffer("py-string");
        mode = "code";
        quoteChar = null;
        tripleQuoted = false;
        continue;
      }
      specialBuffer += char;
      continue;
    }

    specialBuffer += char;
    if (char === quoteChar) {
      flushSpecialBuffer("py-string");
      mode = "code";
      quoteChar = null;
    }
  }

  if (mode === "code") {
    flushCodeBuffer();
  } else if (mode === "comment") {
    flushSpecialBuffer("py-comment");
  } else {
    flushSpecialBuffer("py-string");
  }

  if (normalizedSource.endsWith("\n") && lines.length > 1 && lines[lines.length - 1].length === 0) {
    lines.pop();
  }

  return lines.map((parts, lineIndex) => ({
    lineNumber: lineIndex + 1,
    html: parts.join("") || " ",
  }));
}

function renderPlainTextSource(source) {
  const normalizedSource = typeof source === "string" ? source.replace(/\r\n?/g, "\n") : "";
  const lines = normalizedSource.split("\n");
  if (normalizedSource.endsWith("\n") && lines.length > 1 && lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines.map((line, lineIndex) => ({
    lineNumber: lineIndex + 1,
    html: escapeHtml(line) || " ",
  }));
}

function clampSelectionIndex(value, totalCount) {
  if (!Number.isFinite(totalCount) || totalCount <= 0) return 0;
  const parsed = optionalInteger(value);
  if (parsed === null || parsed < 0) return 0;
  return Math.min(totalCount - 1, parsed);
}

function buildAttemptArtifactEntries(selectedAttempt, selectedGeneratorAttempt) {
  return [
    {
      key: "output",
      label: "LLM Output",
      artifact: selectedGeneratorAttempt?.output || null,
      contentMode: "text",
    },
    {
      key: "prompt",
      label: "LLM Input",
      artifact: selectedGeneratorAttempt?.prompt || null,
      contentMode: "text",
    },
    {
      key: "reasoning",
      label: "LLM Reasoning",
      artifact: selectedGeneratorAttempt?.reasoning || null,
      contentMode: "text",
    },
    {
      key: "error",
      label: "LLM Error",
      artifact: selectedGeneratorAttempt?.error || null,
      contentMode: "text",
    },
    {
      key: "appliedProgram",
      label: "Patched .py",
      artifact: selectedAttempt?.appliedProgram || null,
      contentMode: "python",
    },
    {
      key: "summary",
      label: "Attempt Summary",
      artifact: selectedAttempt?.summary || null,
      contentMode: "text",
    },
  ].filter((entry) => entry.artifact?.url);
}

function chooseDefaultAttemptArtifactKey(selectedAttempt, selectedGeneratorAttempt) {
  const artifactEntries = buildAttemptArtifactEntries(selectedAttempt, selectedGeneratorAttempt);
  return (
    artifactEntries.find((entry) => entry.key === "output")
    || artifactEntries.find((entry) => entry.key === "prompt")
    || artifactEntries[0]
    || null
  )?.key || null;
}

function readCachedProgramArtifactText(artifact, revisionToken = null) {
  const url = artifactKey(artifact);
  if (!url) {
    return {
      hasValue: false,
      text: null,
    };
  }
  const normalizedRevisionToken = String(revisionToken || "").trim() || null;
  if (
    Object.prototype.hasOwnProperty.call(state.programArtifactTextByUrl, url)
    && (state.programArtifactRevisionByUrl[url] || null) === normalizedRevisionToken
  ) {
    return {
      hasValue: true,
      text: state.programArtifactTextByUrl[url],
    };
  }
  return {
    hasValue: false,
    text: null,
  };
}

async function ensureProgramArtifactText(artifact, revisionToken = null) {
  const url = artifactKey(artifact);
  if (!url) return null;
  const normalizedRevisionToken = String(revisionToken || "").trim() || null;
  const cached = readCachedProgramArtifactText(artifact, normalizedRevisionToken);
  if (cached.hasValue) {
    return cached.text;
  }
  if (state.programArtifactRequestByUrl[url] === normalizedRevisionToken) {
    return null;
  }
  state.programArtifactRequestByUrl[url] = normalizedRevisionToken;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`artifact fetch failed: ${response.status}`);
    }
    const rawText = await response.text();
    if (state.programArtifactRequestByUrl[url] === normalizedRevisionToken) {
      state.programArtifactTextByUrl[url] = rawText.replace(/\r\n?/g, "\n");
      state.programArtifactRevisionByUrl[url] = normalizedRevisionToken;
    }
  } catch (_error) {
    if (state.programArtifactRequestByUrl[url] === normalizedRevisionToken) {
      state.programArtifactTextByUrl[url] = null;
      state.programArtifactRevisionByUrl[url] = normalizedRevisionToken;
    }
  } finally {
    if (state.programArtifactRequestByUrl[url] === normalizedRevisionToken) {
      delete state.programArtifactRequestByUrl[url];
      invalidatePanelCache("programSourcePanel");
      renderActive();
    }
  }
  return state.programArtifactTextByUrl[url] ?? null;
}

function formatProgramArtifactDisplayText(artifactEntry, rawText) {
  if (typeof rawText !== "string") return null;
  const normalizedText = rawText.replace(/\r\n?/g, "\n");
  if (artifactEntry?.key === "summary") {
    const trimmed = normalizedText.trim();
    if (trimmed && (trimmed.startsWith("{") || trimmed.startsWith("["))) {
      try {
        return JSON.stringify(JSON.parse(trimmed), null, 2);
      } catch (_error) {
        return normalizedText;
      }
    }
  }
  return normalizedText;
}

function classifyProgramDetailBadge(badgeText) {
  const text = String(badgeText || "").trim();
  if (!text) return null;
  const upper = text.toUpperCase();
  if (/^V\d+$/.test(upper)) {
    return { label: "Version", value: upper, tone: "version" };
  }
  if (/^TRY\s+\d+\/\d+$/i.test(text)) {
    return { label: "LLM", value: text, tone: "trace" };
  }
  if (/^(accepted|success|selected)$/i.test(text)) {
    return { label: "State", value: upper, tone: "success" };
  }
  if (/^(rejected|failed|error)$/i.test(text)) {
    return { label: "State", value: upper, tone: "danger" };
  }
  if (/[0-9]/.test(text) && /(?:ms|s|sec|elapsed|runtime|min)/i.test(text)) {
    return { label: "Elapsed", value: text, tone: "muted" };
  }
  return { value: text, tone: "muted" };
}

function normalizeProgramDetailBadge(detailBadge) {
  if (detailBadge && typeof detailBadge === "object" && !Array.isArray(detailBadge)) {
    const valueText = String(detailBadge.value || "").trim();
    const labelText = String(detailBadge.label || "").trim();
    if (!valueText && !labelText) {
      return null;
    }
    return {
      label: labelText || null,
      value: valueText || labelText,
      tone: String(detailBadge.tone || "").trim() || "muted",
      title: String(detailBadge.title || "").trim() || null,
    };
  }
  return classifyProgramDetailBadge(detailBadge);
}

function resolveAcceptedSourceBadge(selection) {
  const railSelection = String(selection?.railSelection || "").trim().toLowerCase();
  if (railSelection === "live" || railSelection === "current") {
    return "CURRENT ACCEPTED";
  }
  return "FINAL ACCEPTED";
}

function resolveAttemptResultBadge(attempt) {
  const statusText = String(attempt?.status || "").trim().toLowerCase();
  if (attempt?.accepted === true || statusText === "accepted") {
    return { value: "PATCH ACCEPTED", tone: "success" };
  }
  if (
    attempt?.accepted === false
    || statusText === "rejected"
    || statusText === "failed"
    || statusText === "error"
  ) {
    return { value: "PATCH REJECTED", tone: "danger" };
  }
  return null;
}

function createProgramMetaChip(item) {
  const chip = document.createElement("div");
  chip.className = "discovery-program-meta-chip";
  const tone = String(item?.tone || "").trim();
  if (tone) {
    chip.classList.add(`is-${tone}`);
  }
  const titleText = String(item?.title || "").trim();
  if (titleText) {
    chip.title = titleText;
  }
  const labelText = String(item?.label || "").trim();
  if (labelText) {
    const label = document.createElement("span");
    label.className = "discovery-program-meta-chip-label";
    label.textContent = labelText;
    chip.append(label);
  }
  const value = document.createElement("span");
  value.className = "discovery-program-meta-chip-value";
  value.textContent = String(item?.value || "").trim() || "-";
  chip.append(value);
  return chip;
}

function renderProgramMeta(selection, lineCount) {
  if (!elements.programMeta) return;
  elements.programMeta.innerHTML = "";

  const summaryText = String(selection?.subject || "").trim();
  if (summaryText) {
    const summary = document.createElement("div");
    summary.className = "discovery-program-summary";
    summary.textContent = summaryText;
    summary.title = summaryText;
    elements.programMeta.append(summary);
  }

  const primaryRow = document.createElement("div");
  primaryRow.className = "discovery-program-meta-row is-primary";
  const primaryBadgeText = String(selection?.badge || "").trim();
  if (primaryBadgeText) {
    primaryRow.append(createProgramMetaChip({
      value: primaryBadgeText,
      tone: String(selection?.tone || "").trim() || "muted",
    }));
  }

  for (const badgeText of (selection?.detailBadges || [])) {
    const normalized = normalizeProgramDetailBadge(badgeText);
    if (!normalized) continue;
    primaryRow.append(createProgramMetaChip(normalized));
  }

  if (selection?.showLineCount !== false && lineCount > 0) {
    primaryRow.append(createProgramMetaChip({
      label: "Lines",
      value: String(lineCount),
      tone: "count",
      title: `${lineCount} lines`,
    }));
  }
  if (primaryRow.childElementCount > 0) {
    elements.programMeta.append(primaryRow);
  }

  const filePath = String(selection?.filePath || "").trim();
  if (filePath) {
    const secondaryRow = document.createElement("div");
    secondaryRow.className = "discovery-program-meta-row is-secondary";
    secondaryRow.append(createProgramMetaChip({
      label: "File",
      value: filePath,
      tone: "file",
      title: filePath,
    }));
    elements.programMeta.append(secondaryRow);
  }
}

function resolveAcceptedProgramSelection(runPayload, transitionResolved) {
  const { selection, gallery } = transitionResolved;
  const programVersions = normalizeProgramVersions(runPayload);
  let selectedVersion = null;
  let tone = "success";
  let badge = resolveAcceptedSourceBadge(selection);
  let subject = "Accepted program version";

  if (selection.railSelection === "live" || selection.railSelection === "current") {
    const currentVersionKey = resolveCurrentAcceptedVersionKey(
      runPayload,
      gallery,
      selection.versionKey,
    );
    selectedVersion = programVersions.versionsByKey.get(currentVersionKey)
      || gallery.versionsByKey.get(currentVersionKey)
      || programVersions.versions.find((version) => version.isCurrentExplainer)
      || programVersions.versions.find((version) => version.isCurrentProgram)
      || gallery.versions.find((version) => version.isCurrentExplainer)
      || gallery.versions.find((version) => version.isCurrentProgram)
      || programVersions.versions[0]
      || gallery.versions[0]
      || null;
    subject = selectedVersion
      ? "Current accepted program"
      : "Current accepted source";
  } else if (selection.mode === "version") {
    const selectedVersionKey = normalizeVersionKey(selection.versionKey);
    selectedVersion = programVersions.versionsByKey.get(selectedVersionKey)
      || programVersions.versions[0]
      || null;
    subject = selectedVersion
      ? "Accepted source from program_versions"
      : "Accepted source";
  }

  const versionSource = normalizeSourceText(selectedVersion?.source);
  if (!selectedVersion || !versionSource) {
    return {
      panelTitle: "Accepted Source",
      tone,
      badge,
      subject,
      filePath: selectedVersion?.sourcePath || null,
      lineCount: 0,
      source: null,
      sourceDigest: selectedVersion?.sourceDigest || null,
      contentMode: "python",
      showLineCount: true,
      navGroups: [],
      detailBadges: selectedVersion?.versionKey
        ? [{ label: "Version", value: versionLabel(selectedVersion.versionKey), tone: "version" }]
        : [],
      emptyMessage: "No Python source was attached to the accepted source.",
    };
  }

  return {
    panelTitle: "Accepted Source",
    tone,
    badge,
    subject,
    filePath: selectedVersion.sourcePath || `${selectedVersion.versionKey}.py`,
    lineCount: selectedVersion.sourceLineCount > 0
      ? selectedVersion.sourceLineCount
      : sourceLineCount(versionSource),
    source: versionSource,
    sourceDigest: selectedVersion.sourceDigest || null,
    contentMode: "python",
    showLineCount: true,
    navGroups: [],
    detailBadges: selectedVersion?.versionKey
      ? [{ label: "Version", value: versionLabel(selectedVersion.versionKey), tone: "version" }]
      : [],
  };
}

function resolveAttemptTraceSelection(runPayload, transitionResolved) {
  const { selection, selectedFail } = transitionResolved;
  const inspectorState = state.programInspectorByRun[programInspectorRunKey(runPayload)] || {};
  if (selection.activeVariant !== "fail" || !selectedFail) {
    return {
      panelTitle: "LLM Patch I/O",
      tone: "patch",
      badge: null,
      subject: "Select a fail",
      filePath: null,
      lineCount: 0,
      source: null,
      sourceDigest: null,
      contentMode: "text",
      navGroups: [],
      detailBadges: [],
      emptyMessage: "LLM Patch I/O is available after selecting a Fail target.",
    };
  }

  const attempts = Array.isArray(selectedFail.attempts) ? selectedFail.attempts : [];
  if (!attempts.length) {
    return {
      panelTitle: "LLM Patch I/O",
      tone: "patch",
      badge: null,
      subject: formatFailureLabel(selectedFail),
      filePath: null,
      lineCount: 0,
      source: null,
      sourceDigest: null,
      contentMode: "text",
      navGroups: [],
      detailBadges: [],
      emptyMessage: "No patch attempt artifacts were recorded for this fail.",
    };
  }

  const attemptIndex = clampSelectionIndex(inspectorState.attemptIndex, attempts.length);
  const selectedAttempt = attempts[attemptIndex] || attempts[0] || null;
  const generatorAttempts = Array.isArray(selectedAttempt?.generatorAttempts)
    ? selectedAttempt.generatorAttempts
    : [];
  const generatorIndex = clampSelectionIndex(inspectorState.generatorIndex, generatorAttempts.length);
  const selectedGeneratorAttempt = generatorAttempts[generatorIndex] || null;
  const artifactEntries = buildAttemptArtifactEntries(selectedAttempt, selectedGeneratorAttempt);
  const selectedArtifactKey = String(inspectorState.artifactKey || "").trim();
  const fallbackArtifactKey = chooseDefaultAttemptArtifactKey(selectedAttempt, selectedGeneratorAttempt);
  const selectedArtifact = artifactEntries.find((entry) => entry.key === selectedArtifactKey)
    || artifactEntries.find((entry) => entry.key === fallbackArtifactKey)
    || artifactEntries[0]
    || null;
  const navGroups = [];
  if (attempts.length > 1) {
    navGroups.push({
      label: "Patch Attempt",
      items: attempts.map((attempt, index) => ({
        action: "select-program-attempt",
        attemptIndex: index,
        label: `Attempt ${attempt.unexpectedAttempt || index + 1}`,
        active: index === attemptIndex,
      })),
    });
  }
  if (generatorAttempts.length > 1) {
    navGroups.push({
      label: "LLM Call",
      items: generatorAttempts.map((generatorAttempt, index) => ({
        action: "select-program-generator",
        generatorIndex: index,
        label: `Try ${generatorAttempt.generatorTry}/${Math.max(generatorAttempt.totalAttempts || 0, generatorAttempt.generatorTry || 0, 1)}`,
        active: index === generatorIndex,
      })),
    });
  }
  if (artifactEntries.length > 1) {
    navGroups.push({
      label: "Inspect",
      items: artifactEntries.map((entry) => ({
        action: "select-program-artifact",
        artifactKey: entry.key,
        label: entry.label,
        active: entry.key === selectedArtifact?.key,
      })),
    });
  }
  const detailBadges = [];
  detailBadges.push(resolveAttemptResultBadge(selectedAttempt));
  if (selectedAttempt?.currentVersion) {
    detailBadges.push({
      label: "Version",
      value: versionLabel(selectedAttempt.currentVersion),
      tone: "version",
    });
  }
  if (selectedAttempt?.elapsedText) {
    detailBadges.push(selectedAttempt.elapsedText);
  }
  if (selectedGeneratorAttempt) {
    detailBadges.push(
      `Try ${selectedGeneratorAttempt.generatorTry}/${Math.max(selectedGeneratorAttempt.totalAttempts || 0, selectedGeneratorAttempt.generatorTry || 0, 1)}`,
    );
  }

  return {
    panelTitle: "LLM Patch I/O",
    tone: "patch",
    badge: null,
    subject: `${formatFailureLabel(selectedFail)} · Attempt ${selectedAttempt?.unexpectedAttempt || attemptIndex + 1}`,
    filePath: selectedArtifact?.artifact?.path || null,
    lineCount: 0,
    source: null,
    sourceDigest: null,
    contentMode: selectedArtifact?.contentMode || "text",
    showLineCount: false,
    contentArtifact: selectedArtifact,
    navGroups,
    detailBadges,
    emptyMessage: selectedArtifact
      ? null
      : "No renderable artifacts were recorded for this fail attempt.",
  };
}

function resolveProgramSourceSelection(runPayload, resolved = null) {
  const transitionResolved = resolved || resolveTransitionSelection(runPayload);
  const inspectorState = state.programInspectorByRun[programInspectorRunKey(runPayload)] || {};
  if (inspectorState.mode === "attempt") {
    return resolveAttemptTraceSelection(runPayload, transitionResolved);
  }
  return resolveAcceptedProgramSelection(runPayload, transitionResolved);
}

function renderProgramInspectorTabs(navGroups) {
  if (!elements.programTabs) return;
  if (!Array.isArray(navGroups) || !navGroups.length) {
    elements.programTabs.hidden = true;
    elements.programTabs.innerHTML = "";
    return;
  }
  elements.programTabs.hidden = false;
  elements.programTabs.innerHTML = navGroups.map((group) => `
    <div class="discovery-program-tab-group">
      <div class="discovery-program-tab-label">${escapeHtml(group.label || "")}</div>
      <div class="discovery-program-tab-row">
        ${(Array.isArray(group.items) ? group.items : []).map((item) => {
    const attrs = [`data-discovery-action="${escapeHtml(item.action || "")}"`];
    if (item.attemptIndex !== undefined) {
      attrs.push(`data-attempt-index="${escapeHtml(String(item.attemptIndex))}"`);
    }
    if (item.generatorIndex !== undefined) {
      attrs.push(`data-generator-index="${escapeHtml(String(item.generatorIndex))}"`);
    }
    if (item.artifactKey) {
      attrs.push(`data-artifact-key="${escapeHtml(item.artifactKey)}"`);
    }
    if (item.active) {
      attrs.push('aria-pressed="true"');
    }
    return `
          <button
            type="button"
            class="discovery-program-tab${item.active ? " is-active" : ""}"
            ${attrs.join(" ")}
          >
            ${escapeHtml(item.label || "")}
          </button>
        `;
  }).join("")}
      </div>
    </div>
  `).join("");
}

function clearProgramSourcePanel(message = "Use Accepted Source or LLM Patch I/O from the stage toolbar.") {
  setProgramCopyButton(null);
  if (elements.programPanel) {
    elements.programPanel.hidden = true;
  }
  if (elements.programTitle) {
    elements.programTitle.textContent = "Inspector";
  }
  if (elements.programMeta) {
    elements.programMeta.innerHTML = "";
  }
  if (elements.programTabs) {
    elements.programTabs.hidden = true;
    elements.programTabs.innerHTML = "";
  }
  if (elements.programCode) {
    elements.programCode.innerHTML = `<div class="discovery-program-empty">${escapeHtml(message)}</div>`;
  }
}

function renderProgramSourcePanel(runPayload, resolved = null) {
  if (
    !elements.programPanel
    || !elements.programTitle
    || !elements.programMeta
    || !elements.programTabs
    || !elements.programCode
  ) {
    return;
  }
  const isOpen = Boolean(runPayload) && isProgramInspectorOpen(runPayload);
  if (!isOpen) {
    if (shouldRenderPanel("programSourcePanel", {
      isOpen: false,
      runId: runPayload?.runId || null,
    })) {
      elements.programPanel.hidden = true;
      setProgramCopyButton(null);
    }
    return;
  }
  const selection = resolveProgramSourceSelection(runPayload, resolved);
  let displaySource = typeof selection.source === "string" ? selection.source : null;
  let lineCount = selection.lineCount > 0 ? selection.lineCount : 0;
  let emptyMessage = selection.emptyMessage || null;
  let isLoadingArtifact = false;
  if (!displaySource && selection.contentArtifact?.artifact?.url && runPayload) {
    const revisionToken = transitionGalleryRevisionToken(runPayload);
    const cachedArtifact = readCachedProgramArtifactText(
      selection.contentArtifact.artifact,
      revisionToken,
    );
    if (!cachedArtifact.hasValue) {
      ensureProgramArtifactText(selection.contentArtifact.artifact, revisionToken);
      isLoadingArtifact = true;
    } else if (typeof cachedArtifact.text === "string") {
      displaySource = formatProgramArtifactDisplayText(selection.contentArtifact, cachedArtifact.text);
      lineCount = sourceLineCount(displaySource);
    } else {
      emptyMessage = `${selection.contentArtifact.label} artifact could not be loaded.`;
    }
  }
  const signature = stableSignature({
    isOpen,
    panelTitle: selection.panelTitle || "Inspector",
    tone: selection.tone,
    badge: selection.badge,
    subject: selection.subject,
    filePath: selection.filePath,
    lineCount,
    sourceDigest: selection.sourceDigest,
    emptyMessage,
    loading: isLoadingArtifact,
    contentMode: selection.contentMode || "text",
    artifactPath: selection.contentArtifact?.artifact?.path || null,
    navGroups: (selection.navGroups || []).map((group) => ({
      label: group.label,
      items: (group.items || []).map((item) => ({
        label: item.label,
        active: Boolean(item.active),
        action: item.action,
      })),
    })),
    detailBadges: selection.detailBadges || [],
  });
  if (!shouldRenderPanel("programSourcePanel", signature)) {
    return;
  }
  elements.programPanel.hidden = false;
  elements.programTitle.textContent = selection.panelTitle || "Inspector";
  renderProgramInspectorTabs(selection.navGroups || []);

  renderProgramMeta(selection, lineCount);

  if (isLoadingArtifact) {
    setProgramCopyButton(null);
    elements.programCode.innerHTML = '<div class="discovery-program-empty">Loading artifact...</div>';
    return;
  }

  if (!displaySource) {
    setProgramCopyButton(null);
    elements.programCode.innerHTML = `<div class="discovery-program-empty">${escapeHtml(emptyMessage || "No program source available.")}</div>`;
    return;
  }

  setProgramCopyButton(displaySource);
  const contentMode = selection.contentMode === "python" ? "python" : "text";
  const lines = contentMode === "python"
    ? highlightPythonSource(displaySource)
    : renderPlainTextSource(displaySource);
  elements.programCode.innerHTML = lines.map((line) => `
    <div class="discovery-program-line">
      <span class="discovery-program-line-no">${line.lineNumber}</span>
      <code class="discovery-program-line-code${contentMode === "python" ? "" : " is-text"}">${line.html}</code>
    </div>
  `).join("");
}

function renderTransitionMetaChips(host, items) {
  if (!host) return;
  host.innerHTML = "";
  for (const item of items) {
    host.append(createHudItemChip(item));
  }
}

function buildTransitionFailureEmptyState(selectedFail, bundleMeta) {
  const failureLabel = formatFailureLabel(selectedFail);
  const reasonText = String(
    bundleMeta?.predictionErrorMessage
    || bundleMeta?.predictedParseError
    || "",
  ).trim();
  const phaseText = String(bundleMeta?.predictionErrorPhase || "").trim();
  const phaseKey = phaseText.toLowerCase();
  const planningFailure = phaseKey.includes("planning");
  let detailText = "Reason: No renderable predicted next state was recorded.";
  if (reasonText && phaseText) {
    detailText = `Reason: ${reasonText} (${phaseText})`;
  } else if (reasonText) {
    detailText = `Reason: ${reasonText}`;
  } else if (phaseText) {
    detailText = `Phase: ${phaseText}`;
  }
  return {
    tone: "error",
    eyebrow: "ERROR",
    title: planningFailure ? "Planning Objective Failed" : "Prediction Failed",
    body: planningFailure
      ? `The candidate world model did not meet the planning acceptance threshold for ${failureLabel}.`
      : `No predicted next-state image was generated for ${failureLabel}.`,
    detail: detailText,
  };
}

function formatTransitionWitnessHeader(bundleMeta) {
  const chips = [];
  const rejectionReason = String(bundleMeta?.rejectionReason || "").trim();
  if (rejectionReason) {
    chips.push({ label: "Reject", value: rejectionReason });
  }
  const failGroups = Array.isArray(bundleMeta?.failGroups) ? bundleMeta.failGroups : [];
  const selectedGroupCount = Array.isArray(bundleMeta?.selectedRegressionGroupIds)
    ? bundleMeta.selectedRegressionGroupIds.length
    : 0;
  const brokenGroupCount = Array.isArray(bundleMeta?.brokenGroupIds)
    ? bundleMeta.brokenGroupIds.length
    : 0;
  const failGroupCount = failGroups.length || Math.max(selectedGroupCount, brokenGroupCount);
  if (failGroupCount > 0) {
    chips.push({ label: "Fail Groups", value: String(failGroupCount) });
  }
  const failTransitionCount = failGroups.reduce(
    (total, group) => total + Math.max(0, optionalInteger(group?.transitionCount) ?? 0),
    0,
  );
  if (failTransitionCount > 0) {
    chips.push({ label: "Fail Transitions", value: String(failTransitionCount) });
  }
  const splitCount = Array.isArray(bundleMeta?.splitEvents) ? bundleMeta.splitEvents.length : 0;
  if (splitCount > 0) {
    chips.push({ label: "Splits", value: String(splitCount) });
  }
  const witnessCount = Array.isArray(bundleMeta?.selectedWitnesses) ? bundleMeta.selectedWitnesses.length : 0;
  if (witnessCount > 0) {
    chips.push({ label: "Witnesses", value: String(witnessCount) });
  }
  return chips;
}

function pickTransitionWitnessPreviewArtifact(witness) {
  return witness?.predicted || witness?.expected || witness?.previous || null;
}

function formatTransitionWitnessFocusLabel(witness, runPayload = null) {
  return (
    formatTransitionWitnessGroupDisplayLabel(runPayload, witness?.groupId, witness?.commitVersion)
    || String(witness?.caseLabel || "Witness").trim()
    || "Witness"
  );
}

function formatTransitionWitnessClassLabel(classId) {
  return classId !== null && classId !== undefined ? `C${classId}` : "C-";
}

function formatTransitionWitnessCountLabel(count) {
  const normalizedCount = Math.max(0, optionalInteger(count) ?? 0);
  return `${normalizedCount} transition${normalizedCount === 1 ? "" : "s"}`;
}

function formatTransitionWitnessCountBadge(count) {
  const normalizedCount = Math.max(0, optionalInteger(count) ?? 0);
  return String(normalizedCount);
}

function formatTransitionWitnessSummaryIdentity(
  runPayload,
  groupId,
  fallbackCommitVersion = null,
  preferredClassId = null,
) {
  const displayLabel = formatTransitionWitnessGroupDisplayLabel(
    runPayload,
    groupId,
    fallbackCommitVersion,
  );
  const explicitClassLabel = preferredClassId !== null && preferredClassId !== undefined
    ? formatTransitionWitnessClassLabel(preferredClassId)
    : null;
  if (displayLabel && explicitClassLabel) {
    const normalizedDisplayLabel = String(displayLabel || "").trim();
    if (
      normalizedDisplayLabel === explicitClassLabel
      || normalizedDisplayLabel.startsWith(`${explicitClassLabel} · `)
    ) {
      return normalizedDisplayLabel;
    }
    return `${explicitClassLabel} · ${normalizedDisplayLabel}`;
  }
  if (displayLabel) return displayLabel;
  if (explicitClassLabel) return explicitClassLabel;
  const rawGroupId = String(groupId || "").trim();
  if (rawGroupId) {
    return compactPrefixIdentityLabel(rawGroupId);
  }
  const normalizedCommitVersion = normalizeCommitVersion(fallbackCommitVersion);
  return normalizedCommitVersion ? versionLabel(normalizedCommitVersion) : null;
}

function buildTransitionWitnessFailRows(bundleMeta, runPayload = null) {
  const failGroups = Array.isArray(bundleMeta?.failGroups) ? bundleMeta.failGroups : [];
  if (failGroups.length) {
    return failGroups.map((group) => ({
      tone: "fail",
      label: "Fail",
      subject: formatTransitionWitnessSummaryIdentity(
        runPayload,
        group?.groupId,
        group?.commitVersion,
        group?.classId,
      ) || "-",
      value: formatTransitionWitnessCountBadge(group?.transitionCount),
      valueTitle: formatTransitionWitnessCountLabel(group?.transitionCount),
    }));
  }
  const fallbackLabels = [];
  const brokenGroupIds = Array.isArray(bundleMeta?.brokenGroupIds) ? bundleMeta.brokenGroupIds : [];
  if (brokenGroupIds.length) {
    for (const groupId of brokenGroupIds) {
      const label = formatTransitionWitnessSummaryIdentity(runPayload, groupId);
      if (label) fallbackLabels.push(label);
    }
  } else {
    const selectedWitnesses = Array.isArray(bundleMeta?.selectedWitnesses)
      ? bundleMeta.selectedWitnesses
      : [];
    for (const witness of selectedWitnesses) {
      const label = formatTransitionWitnessSummaryIdentity(
        runPayload,
        witness?.groupId,
        witness?.commitVersion,
      );
      if (label) fallbackLabels.push(label);
    }
  }
  return Array.from(new Set(fallbackLabels)).map((label) => ({
    tone: "fail",
    label: "Fail",
    subject: label,
    value: null,
  }));
}

function buildTransitionWitnessRejectSummary(bundleMeta, runPayload = null) {
  const rejectionReason = String(bundleMeta?.rejectionReason || "").trim();
  const failRows = buildTransitionWitnessFailRows(bundleMeta, runPayload);
  const failGroups = Array.isArray(bundleMeta?.failGroups) ? bundleMeta.failGroups : [];
  const totalTransitionCount = failGroups.reduce(
    (total, group) => total + Math.max(0, optionalInteger(group?.transitionCount) ?? 0),
    0,
  );
  if (!rejectionReason && !failRows.length) return null;
  return {
    caption: rejectionReason || null,
    rows: failRows,
    failGroupCount: failRows.length,
    totalTransitionCount,
  };
}

function buildTransitionWitnessSplitSummary(bundleMeta) {
  const splitEvents = Array.isArray(bundleMeta?.splitEvents) ? bundleMeta.splitEvents : [];
  const totalSplitTransitionCount = splitEvents.reduce(
    (total, event) => total + Math.max(0, optionalInteger(event?.brokenTransitionCount) ?? 0),
    0,
  );
  const totalKeptTransitionCount = splitEvents.reduce(
    (total, event) => total + Math.max(0, optionalInteger(event?.keptTransitionCount) ?? 0),
    0,
  );
  return {
    splitCount: splitEvents.length,
    totalSplitTransitionCount,
    totalKeptTransitionCount,
  };
}

function buildTransitionWitnessSummaryCards(bundleMeta, runPayload = null) {
  const cards = [];
  const splitEvents = Array.isArray(bundleMeta?.splitEvents) ? bundleMeta.splitEvents : [];
  for (const splitEvent of splitEvents) {
    cards.push({
      tone: "split",
      title: formatTransitionWitnessSummaryIdentity(
        runPayload,
        splitEvent?.splitGroupId,
        null,
        splitEvent?.sourceClassId,
      ) || "-",
      split: {
        label: "Split",
        tone: "split",
        subject: formatTransitionWitnessSummaryIdentity(
          runPayload,
          splitEvent?.brokenChildGroupId,
          null,
          splitEvent?.brokenClassId,
        ) || "-",
        value: formatTransitionWitnessCountBadge(splitEvent?.brokenTransitionCount),
        valueTitle: formatTransitionWitnessCountLabel(splitEvent?.brokenTransitionCount),
      },
      kept: {
        label: "Kept",
        tone: "kept",
        subject: formatTransitionWitnessSummaryIdentity(
          runPayload,
          splitEvent?.keptChildGroupId,
          null,
          splitEvent?.keptClassId,
        ) || "-",
        value: formatTransitionWitnessCountBadge(splitEvent?.keptTransitionCount),
        valueTitle: formatTransitionWitnessCountLabel(splitEvent?.keptTransitionCount),
      },
    });
  }
  return cards;
}

function createTransitionWitnessLink(label, artifact) {
  const normalizedLabel = String(label || "").trim();
  if (!normalizedLabel) return null;
  if (!artifact?.url) {
    const placeholder = document.createElement("span");
    placeholder.className = "transition-witness-link is-disabled";
    placeholder.textContent = normalizedLabel;
    placeholder.setAttribute("aria-disabled", "true");
    placeholder.title = `${normalizedLabel} unavailable`;
    return placeholder;
  }
  const link = document.createElement("a");
  link.className = "transition-witness-link";
  link.href = artifact.url;
  link.target = "_blank";
  link.rel = "noreferrer noopener";
  link.textContent = normalizedLabel;
  return link;
}

function renderTransitionWitnessStrip(bundleMeta, selection, runPayload = null) {
  const host = elements.transitionWitnessStrip;
  if (!host) return;
  const selectedWitnesses = Array.isArray(bundleMeta?.selectedWitnesses)
    ? bundleMeta.selectedWitnesses
    : [];
  const rejectSummary = buildTransitionWitnessRejectSummary(bundleMeta, runPayload);
  const splitSummary = buildTransitionWitnessSplitSummary(bundleMeta);
  const summaryCards = buildTransitionWitnessSummaryCards(bundleMeta, runPayload);
  const headerChips = formatTransitionWitnessHeader(bundleMeta);
  const focusedWitnessIndex = selection?.focusMode === "witness"
    ? coerceTransitionWitnessIndex(selection?.witnessIndex)
    : null;
  let summarySection = null;
  if (!selectedWitnesses.length && !rejectSummary && !summaryCards.length && !headerChips.length) {
    host.innerHTML = "";
    host.hidden = true;
    return;
  }

  host.hidden = false;
  host.innerHTML = "";

  if (rejectSummary || summaryCards.length || headerChips.length) {
    summarySection = document.createElement("section");
    summarySection.className = "transition-witness-section is-summary";

    const summaryHead = document.createElement("div");
    summaryHead.className = "transition-witness-head";

    const summarySectionTitle = document.createElement("div");
    summarySectionTitle.className = "transition-witness-title";
    summarySectionTitle.textContent = "Reject Summary";
    summaryHead.append(summarySectionTitle);

    if (headerChips.length) {
      const headActions = document.createElement("div");
      headActions.className = "transition-witness-head-actions";
      const chipRow = document.createElement("div");
      chipRow.className = "transition-stage-meta";
      for (const chip of headerChips) {
        chipRow.append(createHudItemChip(chip));
      }
      headActions.append(chipRow);
      summaryHead.append(headActions);
    }
    summarySection.append(summaryHead);

    if (rejectSummary) {
      const overview = document.createElement("div");
      overview.className = "transition-witness-overview";

      if (rejectSummary.caption) {
        const overviewCaption = document.createElement("div");
        overviewCaption.className = "transition-witness-summary-caption";
        overviewCaption.textContent = rejectSummary.caption;
        overview.append(overviewCaption);
      }

      if (Array.isArray(rejectSummary.rows) && rejectSummary.rows.length) {
        const failExpanded = isRejectFailGroupsExpanded(
          runPayload,
          rejectSummary.failGroupCount,
        );

        const overviewHead = document.createElement("div");
        overviewHead.className = "transition-witness-summary-subhead-row";

        const overviewLabel = document.createElement("div");
        overviewLabel.className = "transition-witness-summary-subhead";
        overviewLabel.textContent = "Failed Groups";
        overviewHead.append(overviewLabel);

        const failMeta = document.createElement("div");
        failMeta.className = "transition-witness-summary-submeta";
        const failGroupCountLabel = `${rejectSummary.failGroupCount} group${rejectSummary.failGroupCount === 1 ? "" : "s"}`;
        const failTransitionLabel = rejectSummary.totalTransitionCount > 0
          ? `${rejectSummary.totalTransitionCount} fail`
          : null;
        failMeta.textContent = failTransitionLabel
          ? `${failGroupCountLabel} · ${failTransitionLabel}`
          : failGroupCountLabel;
        overviewHead.append(failMeta);

        if (rejectSummary.failGroupCount > 4) {
          const toggleButton = document.createElement("button");
          toggleButton.type = "button";
          toggleButton.className = "transition-witness-summary-toggle";
          toggleButton.dataset.discoveryAction = "toggle-reject-fail-groups";
          toggleButton.dataset.failGroupCount = String(rejectSummary.failGroupCount);
          toggleButton.setAttribute("aria-expanded", failExpanded ? "true" : "false");
          toggleButton.textContent = failExpanded ? "Hide Groups" : "Show Groups";
          overviewHead.append(toggleButton);
        }

        overview.append(overviewHead);

        if (failExpanded) {
          const failGrid = document.createElement("div");
          failGrid.className = "transition-witness-overview-grid";
          for (const row of rejectSummary.rows) {
            const rowElement = document.createElement("div");
            rowElement.className = "transition-witness-summary-row";
            if (row?.tone) {
              rowElement.classList.add(`is-${row.tone}`);
            }

            const rowBadge = document.createElement("div");
            rowBadge.className = "transition-witness-summary-badge";
            rowBadge.textContent = String(row?.label || "-");
            rowElement.append(rowBadge);

            const rowSubject = document.createElement("div");
            rowSubject.className = "transition-witness-summary-subject";
            rowSubject.textContent = String(row?.subject || "-");
            rowElement.append(rowSubject);

            if (row?.value) {
              const rowValue = document.createElement("div");
              rowValue.className = "transition-witness-summary-value";
              rowValue.textContent = String(row.value);
              if (row?.valueTitle) {
                rowValue.title = String(row.valueTitle);
              }
              rowElement.append(rowValue);
            }
            failGrid.append(rowElement);
          }
          overview.append(failGrid);
        }
      }
      summarySection.append(overview);
    }

    if (summaryCards.length) {
      const splitsExpanded = isRejectSplitsExpanded(
        runPayload,
        splitSummary.splitCount,
      );

      const splitHead = document.createElement("div");
      splitHead.className = "transition-witness-summary-subhead-row";

      const splitLabel = document.createElement("div");
      splitLabel.className = "transition-witness-summary-subhead";
      splitLabel.textContent = "Split Outcomes";
      splitHead.append(splitLabel);

      const splitMeta = document.createElement("div");
      splitMeta.className = "transition-witness-summary-submeta";
      const splitCountLabel = `${splitSummary.splitCount} split${splitSummary.splitCount === 1 ? "" : "s"}`;
      const splitTransitionLabel = splitSummary.totalSplitTransitionCount > 0
        ? `split ${splitSummary.totalSplitTransitionCount}`
        : null;
      const keptTransitionLabel = splitSummary.totalKeptTransitionCount > 0
        ? `kept ${splitSummary.totalKeptTransitionCount}`
        : null;
      splitMeta.textContent = [
        splitCountLabel,
        splitTransitionLabel,
        keptTransitionLabel,
      ].filter(Boolean).join(" · ");
      splitHead.append(splitMeta);

      if (splitSummary.splitCount > 4) {
        const toggleButton = document.createElement("button");
        toggleButton.type = "button";
        toggleButton.className = "transition-witness-summary-toggle";
        toggleButton.dataset.discoveryAction = "toggle-reject-splits";
        toggleButton.dataset.splitCount = String(splitSummary.splitCount);
        toggleButton.setAttribute("aria-expanded", splitsExpanded ? "true" : "false");
        toggleButton.textContent = splitsExpanded ? "Hide Splits" : "Show Splits";
        splitHead.append(toggleButton);
      }

      summarySection.append(splitHead);

      if (splitsExpanded) {
        const summary = document.createElement("div");
        summary.className = "transition-witness-split-grid";
        for (const summaryCard of summaryCards) {
          const item = document.createElement("div");
          item.className = "transition-witness-split-card";
          if (summaryCard.tone) {
            item.classList.add(`is-${summaryCard.tone}`);
          }

          const summaryTitle = document.createElement("div");
          summaryTitle.className = "transition-witness-summary-title";
          summaryTitle.textContent = summaryCard.title;
          item.append(summaryTitle);

          const laneGrid = document.createElement("div");
          laneGrid.className = "transition-witness-split-rows";
          for (const lane of [summaryCard.split, summaryCard.kept]) {
            const laneElement = document.createElement("div");
            laneElement.className = "transition-witness-summary-row";
            if (lane?.tone) {
              laneElement.classList.add(`is-${lane.tone}`);
            }

            const laneBadge = document.createElement("div");
            laneBadge.className = "transition-witness-summary-badge";
            laneBadge.textContent = String(lane?.label || "-");
            laneElement.append(laneBadge);

            const laneSubject = document.createElement("div");
            laneSubject.className = "transition-witness-summary-subject";
            laneSubject.textContent = String(lane?.subject || "-");
            laneElement.append(laneSubject);

            if (lane?.value) {
              const laneValue = document.createElement("div");
              laneValue.className = "transition-witness-summary-value";
              laneValue.textContent = String(lane.value);
              if (lane?.valueTitle) {
                laneValue.title = String(lane.valueTitle);
              }
              laneElement.append(laneValue);
            }
            laneGrid.append(laneElement);
          }
          item.append(laneGrid);
          summary.append(item);
        }
        summarySection.append(summary);
      }
    }
  }

  if (selectedWitnesses.length) {
    const witnessSection = document.createElement("section");
    witnessSection.className = "transition-witness-section is-witnesses";

    const witnessHead = document.createElement("div");
    witnessHead.className = "transition-witness-head";

    const witnessTitle = document.createElement("div");
    witnessTitle.className = "transition-witness-title";
    witnessTitle.textContent = "Selected Witnesses";
    witnessHead.append(witnessTitle);

    if (focusedWitnessIndex !== null) {
      const headActions = document.createElement("div");
      headActions.className = "transition-witness-head-actions";
      const resetButton = document.createElement("button");
      resetButton.type = "button";
      resetButton.className = "transition-witness-reset";
      resetButton.dataset.discoveryAction = "reset-witness-focus";
      resetButton.textContent = "Target View";
      headActions.append(resetButton);
      witnessHead.append(headActions);
    }
    witnessSection.append(witnessHead);

    const cards = document.createElement("div");
    cards.className = "transition-witness-cards";
    if (selectedWitnesses.length <= 2) {
      cards.classList.add("is-sparse");
    }
    selectedWitnesses.forEach((witness, witnessIndex) => {
      const card = document.createElement("article");
      card.className = "transition-witness-card";
      card.dataset.discoveryAction = "select-witness";
      card.dataset.witnessIndex = String(witnessIndex);
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      if (focusedWitnessIndex === witnessIndex) {
        card.classList.add("is-active");
      }

      const cardHead = document.createElement("div");
      cardHead.className = "transition-witness-card-head";

      const titleRow = document.createElement("div");
      titleRow.className = "transition-witness-card-title-row";

      const cardTitle = document.createElement("div");
      cardTitle.className = "transition-witness-card-title";
      cardTitle.textContent = formatTransitionWitnessFocusLabel(witness, runPayload);
      titleRow.append(cardTitle);

      const cardBadge = document.createElement("div");
      cardBadge.className = "transition-witness-card-badge";
      cardBadge.textContent = focusedWitnessIndex === witnessIndex ? "Viewing" : "Inspect";
      titleRow.append(cardBadge);
      cardHead.append(titleRow);

      const subtitleParts = [
        witness.caseLabel,
        formatTransitionLocationText(witness),
        witness.commitVersion ? versionLabel(witness.commitVersion) : null,
        witness.caseSource,
      ].filter(Boolean);
      if (subtitleParts.length) {
        const cardSubtitle = document.createElement("div");
        cardSubtitle.className = "transition-witness-card-subtitle";
        cardSubtitle.textContent = subtitleParts.join(" | ");
        cardHead.append(cardSubtitle);
      }
      card.append(cardHead);

      const previewHost = document.createElement("div");
      previewHost.className = "transition-witness-preview";
      const previewArtifact = pickTransitionWitnessPreviewArtifact(witness);
      if (previewArtifact?.url) {
        const previewImage = document.createElement("img");
        previewImage.className = "transition-witness-preview-image";
        previewImage.src = previewArtifact.url;
        previewImage.alt = `${witness.caseLabel} preview`;
        previewHost.append(previewImage);
      } else {
        const previewEmpty = document.createElement("div");
        previewEmpty.className = "transition-witness-preview-empty";
        previewEmpty.textContent = "No renderable witness preview available.";
        previewHost.append(previewEmpty);
      }
      card.append(previewHost);

      const links = document.createElement("div");
      links.className = "transition-witness-links";
      for (const [label, artifact] of [
        ["Prev", witness.previous],
        ["GT", witness.expected],
        ["Pred", witness.predicted],
        ["Bundle", witness.bundle],
      ]) {
        const link = createTransitionWitnessLink(label, artifact);
        if (link) {
          links.append(link);
        }
      }
      card.append(links);

      cards.append(card);
    });
    witnessSection.append(cards);
    host.append(witnessSection);
  }

  if (summarySection) {
    host.append(summarySection);
  }
}

function renderTransitionImageFrame(host, artifact, statePayload, visualConfig, emptyText) {
  if (!host) return;
  if (!artifact?.url && statePayload && typeof statePayload === "object") {
    const scene = buildStateScene(statePayload, visualConfig || null);
    renderBoard(scene, host, {
      tileSize: 36,
      fitToHost: true,
      minTileSize: 12,
      maxTileSize: 44,
      fallbackHostHeight: 420,
      emptyMessage: typeof emptyText === "string" ? emptyText : "state unavailable",
    });
    return;
  }
  host.innerHTML = "";
  delete host.dataset.sceneSignature;
  if (!artifact?.url) {
    if (emptyText && typeof emptyText === "object") {
      const toneText = String(emptyText.tone || "").trim().toLowerCase();
      const toneClass = toneText ? ` is-${escapeHtml(toneText)}` : "";
      const eyebrow = String(emptyText.eyebrow || "").trim();
      const title = String(emptyText.title || "").trim();
      const body = String(emptyText.body || "").trim();
      const detail = String(emptyText.detail || "").trim();
      host.innerHTML = `
        <div class="transition-empty transition-empty-detailed${toneClass}">
          ${eyebrow ? `<div class="transition-empty-eyebrow">${escapeHtml(eyebrow)}</div>` : ""}
          ${title ? `<div class="transition-empty-title">${escapeHtml(title)}</div>` : ""}
          ${body ? `<div class="transition-empty-body">${escapeHtml(body)}</div>` : ""}
          ${detail ? `<div class="transition-empty-detail">${escapeHtml(detail)}</div>` : ""}
        </div>
      `;
      return;
    }
    host.innerHTML = `<div class="transition-empty">${escapeHtml(emptyText)}</div>`;
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
  image.alt = String(artifact.path || "transition image");
  image.loading = "eager";
  image.decoding = "async";
  if ("fetchPriority" in image) {
    image.fetchPriority = "high";
  }

  link.append(image);
  host.append(link);
}

function normalizeProgressPhase(runPayload) {
  const metrics = runPayload?.dashboard?.metrics;
  const rawPhase = metrics?.progress_phase ?? metrics?.progressPhase;
  const phase = String(rawPhase || "").trim().toLowerCase();
  return phase || null;
}

function isCurrentPatchAvailable(runPayload, galleryInput = null) {
  const gallery = galleryInput || normalizeTransitionGallery(runPayload);
  const liveReviewId = String(gallery.live?.defaultReview?.reviewId || "").trim() || null;
  const liveFailureId = String(gallery.live?.defaultReview?.failureId || "").trim() || null;
  if (!(gallery.live?.isFail && liveReviewId && liveFailureId)) {
    return false;
  }
  return normalizeProgressPhase(runPayload) === "patch";
}

// Keep the terminal CURRENT patch visible after discovery ends so the last
// failed patch remains inspectable in the dashboard.
function isCurrentPatchVisible(runPayload, galleryInput = null) {
  const gallery = galleryInput || normalizeTransitionGallery(runPayload);
  const liveReviewId = String(gallery.live?.defaultReview?.reviewId || "").trim() || null;
  const liveFailureId = String(gallery.live?.defaultReview?.failureId || "").trim() || null;
  if (!(gallery.live?.isFail && liveReviewId && liveFailureId)) {
    return false;
  }
  const phase = normalizeProgressPhase(runPayload);
  const isLiveRun = Boolean(runPayload?.isLive);
  return phase === "patch" || !isLiveRun;
}

function resolveWorldViewerLiveState(runPayload) {
  const phase = normalizeProgressPhase(runPayload);
  if (phase) {
    return phase === "collect" ? "live_on" : "standby";
  }
  return Boolean(runPayload?.dashboard?.metrics?.resume_pending) ? "standby" : "live_on";
}

function setTransitionNavigatorSlotText(host, slotName, value) {
  const slot = host?.querySelector(`[data-discovery-slot="${slotName}"]`);
  if (!slot) return;
  const nextText = String(value ?? "");
  if (slot.textContent !== nextText) {
    slot.textContent = nextText;
  }
}

function setTransitionNavigatorStatus(host, statusClass, statusText) {
  const status = host?.querySelector('[data-discovery-slot="status"]');
  if (!status) return;
  status.className = `discovery-version-status ${statusClass}`;
  setTransitionNavigatorSlotText(host, "status-text", statusText);
}

function setTransitionNavigatorEvalState(host, evalBadge) {
  if (!host) return;
  const progress = Math.max(0, Math.min(100, Number(evalBadge?.progressPercent || 0)));
  host.style.setProperty("--eval-progress", `${progress.toFixed(2)}%`);
  setTransitionNavigatorSlotText(host, "eval-caption", evalBadge?.caption || "RUN");
}

function renderTransitionNavigator(runPayload, resolved = null) {
  if (!elements.versionList || !elements.versionSummary) return;
  const transitionResolved = resolved || resolveTransitionSelection(runPayload);
  const { gallery, selection } = transitionResolved;
  const programVersions = normalizeProgramVersions(runPayload);
  const liveReviewId = String(gallery.live?.defaultReview?.reviewId || "").trim() || null;
  const liveFailureId = String(gallery.live?.defaultReview?.failureId || "").trim() || null;
  const currentPatchVisible = isCurrentPatchVisible(runPayload, gallery);
  const currentPatchAvailable = isCurrentPatchAvailable(runPayload, gallery);
  const currentPatchActive = currentPatchAvailable && Boolean(runPayload?.isLive);
  const livePatchSelected = selection.railSelection === "current";
  const liveRuntimeState = resolveWorldViewerLiveState(runPayload);
  const liveStatusClass = !runPayload?.isLive
    ? "is-stopped"
    : isPausedView()
      ? "is-paused"
      : liveRuntimeState === "standby"
        ? "is-standby"
        : "is-active";
  const liveStatusText = !runPayload?.isLive
    ? "STOPPED"
    : isPausedView()
      ? "PAUSED"
      : liveRuntimeState === "standby"
        ? "STANDBY"
        : "LIVE ON";
  const liveIdentityText = resolveLiveNavigatorIdentity(runPayload, gallery);
  const summaryText = formatLines([
    `live: ${liveIdentityText}`,
    `program versions: ${programVersions.versions.length}`,
  ]);
  if (shouldRenderPanel("transitionVersionSummary", summaryText)) {
    elements.versionSummary.textContent = summaryText;
  }

  const versionButtonModels = programVersions.versions.map((entry) => {
    const review = resolvePrimaryReview(entry, gallery);
    const stepText = review ? `STEP ${String(review.stepIndex).padStart(3, "0")}` : "STEP ---";
    const failCount = Array.isArray(review?.fails) ? review.fails.length : 0;
    return {
      versionKey: entry.versionKey,
      versionLabel: versionLabel(entry.versionKey),
      sourcePath: entry.sourcePath || null,
      sourceDigest: entry.sourceDigest || null,
      stepText,
      failText: `FAIL ${failCount}`,
      isSelected: (
        selection.railSelection !== "current"
        && selection.mode === "version"
        && selection.versionKey === entry.versionKey
      ),
    };
  });
  const listStructureSignature = stableSignature({
    versionKeys: versionButtonModels.map((entry) => entry.versionKey),
    currentPatchVisible,
  });
  const listSignature = stableSignature({
    versionButtons: versionButtonModels,
    selection,
    runIsLive: Boolean(runPayload?.isLive),
    progressPhase: normalizeProgressPhase(runPayload),
    liveIdentityText,
    liveStatusClass,
    liveStatusText,
    currentPatchVisible,
    currentPatchActive,
    live: {
      currentVersionKey: gallery.live?.currentVersionKey || null,
      currentGroupKey: gallery.live?.currentGroupKey || null,
      currentClassIndex: gallery.live?.currentClassIndex ?? null,
      isFail: Boolean(gallery.live?.isFail),
      defaultReview: gallery.live?.defaultReview || null,
    },
    offlineEvalRuns: state.offlineEval.runs.map((row) => ({
      runId: row.runId || null,
      runMode: normalizeOfflineEvalRunMode(row?.runMode),
      versionKey: normalizeVersionKey(row?.program?.versionKey),
      sourceDigest: row?.program?.sourceDigest || null,
      sourcePath: row?.program?.sourcePath || null,
      status: row?.status || null,
      completedCount: row?.progress?.completedCount ?? null,
      totalCount: row?.progress?.totalCount ?? null,
      failedCount: row?.metrics?.failedCount ?? null,
      startedAt: row?.startedAt || null,
    })),
  });
  const shouldRebuildList = shouldRenderPanel("transitionVersionListStructure", listStructureSignature);
  const shouldSyncList = shouldRenderPanel("transitionVersionList", listSignature);
  if (!shouldRebuildList && !shouldSyncList) {
    return;
  }

  // Keep navigator buttons mounted across live scene updates so hover/focus
  // states do not reset while the world viewer is streaming.
  if (shouldRebuildList) {
    elements.versionList.innerHTML = "";

    const liveButton = document.createElement("button");
    liveButton.type = "button";
    liveButton.dataset.discoveryAction = "select-live";
    liveButton.className = "discovery-version-main-button";
    liveButton.innerHTML = `
      <span class="discovery-version-shell">
        <span class="discovery-version-main">
          <span class="discovery-version-v" data-discovery-slot="primary"></span>
          <span class="discovery-version-g" data-discovery-slot="secondary"></span>
        </span>
        <span class="discovery-version-metrics">
          <span class="discovery-version-status" data-discovery-slot="status">
            <span class="discovery-version-status-dot"></span>
            <span data-discovery-slot="status-text"></span>
          </span>
        </span>
      </span>
    `;
    elements.versionList.append(liveButton);

    for (const entry of versionButtonModels) {
      const wrapper = document.createElement("div");
      wrapper.className = "discovery-version-entry";
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.discoveryAction = "select-version";
      button.dataset.versionKey = entry.versionKey;
      button.className = "discovery-version-main-button";
      button.innerHTML = `
        <span class="discovery-version-shell">
          <span class="discovery-version-main">
            <span class="discovery-version-v" data-discovery-slot="primary"></span>
          </span>
          <span class="discovery-version-metrics">
            <span class="discovery-version-metric" data-discovery-slot="step"></span>
            <span class="discovery-version-metric" data-discovery-slot="fail"></span>
          </span>
        </span>
      `;
      const evalButton = document.createElement("button");
      evalButton.type = "button";
      evalButton.className = "button discovery-version-eval-button";
      evalButton.dataset.discoveryAction = "open-offline-eval";
      evalButton.dataset.versionKey = entry.versionKey;
      evalButton.setAttribute("aria-label", `${entry.versionLabel} offline eval`);
      evalButton.innerHTML = `
        <span class="discovery-version-eval-shell">
          <span class="discovery-version-eval-fill" aria-hidden="true"></span>
          <span class="discovery-version-eval-content">
            <span class="discovery-version-eval-icon" aria-hidden="true">
              <svg viewBox="0 0 16 16" focusable="false">
                <path d="M6 2.5h4"></path>
                <path d="M7 2.5v3.2L3.6 11a1.8 1.8 0 0 0 1.5 2.8h5.8a1.8 1.8 0 0 0 1.5-2.8L9 5.7V2.5"></path>
                <path d="M5.2 10.2h5.6"></path>
              </svg>
            </span>
            <span class="discovery-version-eval-caption" data-discovery-slot="eval-caption">RUN</span>
          </span>
        </span>
      `;
      const classButton = document.createElement("button");
      classButton.type = "button";
      classButton.className = "button discovery-version-eval-button discovery-version-class-button";
      classButton.dataset.discoveryAction = "open-offline-class-analysis";
      classButton.dataset.versionKey = entry.versionKey;
      classButton.setAttribute("aria-label", `${entry.versionLabel} dynamics class analysis`);
      classButton.innerHTML = `
        <span class="discovery-version-eval-shell">
          <span class="discovery-version-eval-fill" aria-hidden="true"></span>
          <span class="discovery-version-eval-content">
            <span class="discovery-version-eval-icon" aria-hidden="true">
              <svg viewBox="0 0 16 16" focusable="false">
                <path d="M3 4.5h10"></path>
                <path d="M3 8h10"></path>
                <path d="M3 11.5h10"></path>
                <path d="M5.5 3v10"></path>
                <path d="M10.5 3v10"></path>
              </svg>
            </span>
            <span class="discovery-version-eval-caption" data-discovery-slot="eval-caption">CLASS</span>
          </span>
        </span>
      `;
      wrapper.append(button, evalButton, classButton);
      elements.versionList.append(wrapper);
    }

    if (currentPatchVisible) {
      const patchButton = document.createElement("button");
      patchButton.type = "button";
      patchButton.dataset.discoveryAction = "select-live-patch";
      patchButton.className = "discovery-version-main-button";
      patchButton.innerHTML = `
        <span class="discovery-version-shell">
          <span class="discovery-version-main">
            <span class="discovery-version-v" data-discovery-slot="primary"></span>
            <span class="discovery-version-g" data-discovery-slot="secondary"></span>
          </span>
          <span class="discovery-version-metrics">
            <span class="discovery-version-metric" data-discovery-slot="fail"></span>
            <span class="discovery-version-status" data-discovery-slot="status">
              <span class="discovery-version-status-dot"></span>
              <span data-discovery-slot="status-text"></span>
            </span>
          </span>
        </span>
      `;
      elements.versionList.append(patchButton);
    }
  }

  const liveButton = elements.versionList.querySelector('[data-discovery-action="select-live"]');
  if (liveButton) {
    liveButton.className = `discovery-version-button is-live ${liveStatusClass}`;
    liveButton.classList.toggle("is-selected", selection.railSelection === "live");
    setTransitionNavigatorSlotText(liveButton, "primary", "LIVE");
    setTransitionNavigatorSlotText(liveButton, "secondary", liveIdentityText);
    setTransitionNavigatorStatus(liveButton, liveStatusClass, liveStatusText);
  }

  const versionButtonsByKey = new Map(
    Array.from(elements.versionList.querySelectorAll('[data-discovery-action="select-version"]'))
      .map((button) => [String(button.dataset.versionKey || ""), button]),
  );
  for (const entry of versionButtonModels) {
    const button = versionButtonsByKey.get(entry.versionKey);
    if (!button) continue;
    button.className = "discovery-version-button discovery-version-main-button";
    button.classList.toggle("is-selected", entry.isSelected);
    setTransitionNavigatorSlotText(button, "primary", entry.versionLabel);
    setTransitionNavigatorSlotText(button, "step", entry.stepText);
    setTransitionNavigatorSlotText(button, "fail", entry.failText);
    const evalButton = button.parentElement?.querySelector('[data-discovery-action="open-offline-eval"]');
    if (evalButton) {
      const evalBadge = resolveOfflineEvalVersionBadge(entry);
      evalButton.className = `button discovery-version-eval-button is-${evalBadge.tone}`;
      evalButton.title = evalBadge.title;
      evalButton.setAttribute("aria-label", `${entry.versionLabel} ${evalBadge.title}`);
      setTransitionNavigatorEvalState(evalButton, evalBadge);
    }
    const classButton = button.parentElement?.querySelector('[data-discovery-action="open-offline-class-analysis"]');
    if (classButton) {
      const classBadge = resolveOfflineEvalVersionBadge(entry, "class_purity");
      classButton.className = `button discovery-version-eval-button discovery-version-class-button is-${classBadge.tone}`;
      classButton.title = classBadge.title;
      classButton.setAttribute("aria-label", `${entry.versionLabel} ${classBadge.title}`);
      setTransitionNavigatorEvalState(classButton, classBadge);
    }
  }

  const patchButton = elements.versionList.querySelector('[data-discovery-action="select-live-patch"]');
  if (patchButton && currentPatchVisible) {
    const currentPatchLabel = currentPatchActive ? "CURRENT" : "FAILED";
    const patchStepLabel = `STEP ${String(gallery.live.defaultReview.stepIndex || 0).padStart(3, "0")}`;
    const patchFailLabel = formatFailureLabel({
      failureId: liveFailureId,
      failureIndex: gallery.live.defaultReview.failureIndex,
    });
    const currentPatchStatusLabel = currentPatchActive ? "PATCH" : "FAILED";
    const currentPatchStatusClass = currentPatchActive ? "is-patching" : "is-failed";
    patchButton.className = `discovery-version-button is-patch ${currentPatchActive ? "is-active" : "is-failed"}`;
    patchButton.classList.toggle("is-selected", livePatchSelected);
    setTransitionNavigatorSlotText(patchButton, "primary", currentPatchLabel);
    setTransitionNavigatorSlotText(patchButton, "secondary", patchStepLabel);
    setTransitionNavigatorSlotText(patchButton, "fail", patchFailLabel);
    setTransitionNavigatorStatus(patchButton, currentPatchStatusClass, currentPatchStatusLabel);
  }
}

function renderTransitionStage(runPayload, resolved = null) {
  if (!elements.transitionStage || !elements.transitionToolbar) return;
  const transitionResolved = resolved || resolveTransitionSelection(runPayload);
  const {
    selection,
    review,
    previous,
    bundleMeta,
    failOptions,
    selectedFail,
    successArtifact,
    successBundle,
    focusedWitness,
    focusedWitnessIndex,
    focusedWitnessBundle,
    previousStatePayload,
    nextStatePayload,
    previousAction,
    previousTerminated,
    nextTerminated,
    nextTone,
    nextArtifact,
    nextLabel,
    metadataBundle,
  } = transitionResolved;
  const isCurrentSelection = selection.railSelection === "current";
  const bundleUrl = String(metadataBundle?.url || "").trim();
  const hasCachedBundleMeta = bundleUrl
    ? Object.prototype.hasOwnProperty.call(state.transitionBundleMetaByUrl, bundleUrl)
    : false;
  const cachedBundleMeta = bundleMeta ?? (
    bundleUrl
      ? state.transitionBundleMetaByUrl[bundleUrl] ?? null
      : null
  );
  const focusedWitnessBundleUrl = String(focusedWitnessBundle?.url || "").trim();
  const hasCachedFocusedWitnessBundleMeta = focusedWitnessBundleUrl
    ? Object.prototype.hasOwnProperty.call(state.transitionBundleMetaByUrl, focusedWitnessBundleUrl)
    : false;

  const isTransitionMode = selection.mode === "version";
  if (!isTransitionMode) return;
  const bundleRevision = transitionGalleryRevisionToken(runPayload);
  const cachedBundleRevision = bundleUrl
    ? state.transitionBundleMetaRevisionByUrl[bundleUrl] || null
    : null;
  const hasFreshCachedBundleMeta = hasCachedBundleMeta && cachedBundleRevision === bundleRevision;
  const focusedWitnessBundleRevision = focusedWitnessBundleUrl
    ? state.transitionBundleMetaRevisionByUrl[focusedWitnessBundleUrl] || null
    : null;
  const hasFreshCachedFocusedWitnessBundleMeta = (
    hasCachedFocusedWitnessBundleMeta
    && focusedWitnessBundleRevision === bundleRevision
  );
  const requiresFailBundleMeta = selection.activeVariant === "fail" && !hasFreshCachedBundleMeta;
  if (
    metadataBundle?.url
    && (
      requiresFailBundleMeta
      || (
      previousAction === null
      || previousTerminated === null
      || nextTerminated === null
      || (
        selection.activeVariant === "fail"
        && !selectedFail?.image?.url
        && !hasFreshCachedBundleMeta
      )
      )
    )
  ) {
    ensureTransitionBundleMetadata(metadataBundle, bundleRevision);
  }
  if (focusedWitnessBundle?.url && !hasFreshCachedFocusedWitnessBundleMeta) {
    ensureTransitionBundleMetadata(focusedWitnessBundle, bundleRevision);
  }

  const toolbarSignature = stableSignature({
    galleryRevision: transitionGalleryRevisionToken(runPayload),
    reviewId: review?.reviewId || null,
    reviewWorldIndex: review?.worldIndex ?? null,
    reviewMapName: review?.mapName ?? null,
    activeVariant: selection.activeVariant,
    failureId: selection.failureId,
    success: artifactKey(successArtifact),
    successBundle: artifactKey(successBundle),
    successWorldIndex: review?.explained?.worldIndex ?? null,
    successMapName: review?.explained?.mapName ?? null,
    fails: failOptions.map((fail) => ({
      failureId: fail.failureId,
      failureIndex: fail.failureIndex,
      worldIndex: fail.worldIndex ?? null,
      mapName: fail.mapName ?? null,
      image: artifactKey(fail.image),
    })),
    inspectorOpen: isProgramInspectorOpen(runPayload),
    inspectorMode: state.programInspectorByRun[programInspectorRunKey(runPayload)]?.mode || "program",
  });
  if (shouldRenderPanel("transitionStageToolbar", toolbarSignature)) {
    elements.transitionToolbar.innerHTML = "";
    const inspectorState = state.programInspectorByRun[programInspectorRunKey(runPayload)] || {};
    const inspectorOpen = isProgramInspectorOpen(runPayload);
    const toolbarShell = document.createElement("div");
    toolbarShell.className = "discovery-transition-toolbar-shell";

    const primaryGroup = document.createElement("div");
    primaryGroup.className = "discovery-transition-toolbar-group is-primary";
    const primaryLabel = document.createElement("div");
    primaryLabel.className = "discovery-transition-toolbar-label";
    primaryLabel.textContent = "State View";
    const primaryRow = document.createElement("div");
    primaryRow.className = "discovery-transition-toolbar-row is-primary";
    primaryGroup.append(primaryLabel, primaryRow);

    const expectedButton = document.createElement("button");
    expectedButton.type = "button";
    expectedButton.className = "discovery-transition-chip is-expected";
    expectedButton.dataset.discoveryAction = "select-target";
    expectedButton.dataset.targetVariant = "expected";
    expectedButton.textContent = "Expected";
    if (selection.activeVariant === "expected") {
      expectedButton.classList.add("is-active");
    }
    primaryRow.append(expectedButton);

    const successButton = document.createElement("button");
    successButton.type = "button";
    successButton.className = "discovery-transition-chip is-success";
    successButton.dataset.discoveryAction = "select-target";
    successButton.dataset.targetVariant = "success";
    successButton.textContent = "Success";
    if (!isCurrentSelection) {
      successButton.toggleAttribute("disabled", !(successArtifact?.url || successBundle?.url));
      if (selection.activeVariant === "success") {
        successButton.classList.add("is-active");
      }
      primaryRow.append(successButton);
    }

    for (const fail of failOptions) {
      const failButton = document.createElement("button");
      failButton.type = "button";
      failButton.className = "discovery-transition-chip is-fail";
      failButton.dataset.discoveryAction = "select-target";
      failButton.dataset.targetVariant = "fail";
      failButton.dataset.failureId = fail.failureId;
      failButton.textContent = formatFailureLabel(fail, "Fail");
      if (selection.activeVariant === "fail" && selectedFail?.failureId === fail.failureId) {
        failButton.classList.add("is-active");
      }
      primaryRow.append(failButton);
    }

    const secondaryGroup = document.createElement("div");
    secondaryGroup.className = "discovery-transition-toolbar-group is-secondary";
    const secondaryLabel = document.createElement("div");
    secondaryLabel.className = "discovery-transition-toolbar-label";
    secondaryLabel.textContent = "Inspector";
    const secondaryRow = document.createElement("div");
    secondaryRow.className = "discovery-transition-toolbar-row is-secondary";
    secondaryGroup.append(secondaryLabel, secondaryRow);

    if (selection.activeVariant === "fail") {
      const attemptTraceButton = document.createElement("button");
      attemptTraceButton.type = "button";
      attemptTraceButton.className = "discovery-transition-chip is-inspector";
      attemptTraceButton.dataset.discoveryAction = "open-attempt-trace";
      attemptTraceButton.textContent = "Fail LLM I/O";
      if (inspectorOpen && inspectorState.mode === "attempt") {
        attemptTraceButton.classList.add("is-active");
      }
      secondaryRow.append(attemptTraceButton);
    } else {
      const acceptedProgramButton = document.createElement("button");
      acceptedProgramButton.type = "button";
      acceptedProgramButton.className = "discovery-transition-chip is-inspector";
      acceptedProgramButton.dataset.discoveryAction = "open-accepted-program";
      acceptedProgramButton.textContent = "Successful Version";
      if (inspectorOpen && inspectorState.mode !== "attempt") {
        acceptedProgramButton.classList.add("is-active");
      }
      secondaryRow.append(acceptedProgramButton);
    }

    toolbarShell.append(primaryGroup, secondaryGroup);
    elements.transitionToolbar.append(toolbarShell);
  }

  const frameSignature = stableSignature({
    galleryRevision: transitionGalleryRevisionToken(runPayload),
    previous: artifactKey(previous),
    next: artifactKey(nextArtifact),
    nextLabel,
    previousStatePayload: stableSignature(previousStatePayload),
    nextStatePayload: stableSignature(nextStatePayload),
    previousWorldIndex: normalizeTransitionLocation(focusedWitness || review).worldIndex,
    previousMapName: normalizeTransitionLocation(focusedWitness || review).mapName,
    nextWorldIndex: normalizeTransitionLocation(
      focusedWitness
        || (selection.activeVariant === "fail" ? selectedFail : null)
        || (selection.activeVariant === "success" ? review?.explained : null)
        || review,
    ).worldIndex,
    nextMapName: normalizeTransitionLocation(
      focusedWitness
        || (selection.activeVariant === "fail" ? selectedFail : null)
        || (selection.activeVariant === "success" ? review?.explained : null)
        || review,
    ).mapName,
    focusMode: selection.focusMode || "target",
    focusedWitnessIndex,
    focusedWitnessLabel: focusedWitness ? formatTransitionWitnessFocusLabel(focusedWitness, runPayload) : null,
    previousAction,
    previousTerminated,
    nextTerminated,
    nextTone,
    activeVariant: selection.activeVariant,
    failureId: selectedFail?.failureId || null,
    nextErrorMessage: (
      selection.activeVariant === "fail" && !selectedFail?.image?.url
        ? (
          cachedBundleMeta?.predictionErrorMessage
          || cachedBundleMeta?.predictedParseError
          || cachedBundleMeta?.predictionErrorPhase
          || null
        )
        : null
    ),
  });
  if (shouldRenderPanel("transitionStageFrames", frameSignature)) {
    if (elements.transitionNextLabel) {
      elements.transitionNextLabel.textContent = nextLabel;
    }
    const previousMetaItems = [
      { label: "Action", value: formatActionLabel(previousAction) },
      { label: "Done", value: formatTerminatedLabel(previousTerminated) },
    ];
    appendTransitionLocationChips(previousMetaItems, focusedWitness || review);
    renderTransitionMetaChips(elements.transitionPreviousMeta, previousMetaItems);
    const nextMetaItems = [
      { label: "Done", value: formatTerminatedLabel(nextTerminated) },
    ];
    if (!focusedWitness) {
      appendTransitionLocationChips(
        nextMetaItems,
        (selection.activeVariant === "fail" ? selectedFail : null)
          || (selection.activeVariant === "success" ? review?.explained : null)
          || review,
      );
    }
    if (focusedWitness) {
      nextMetaItems.push({ label: "Focus", value: formatTransitionWitnessFocusLabel(focusedWitness, runPayload) });
    }
    renderTransitionMetaChips(elements.transitionNextMeta, nextMetaItems);
    elements.transitionNextPanel?.classList.remove(
      "is-variant-expected",
      "is-variant-success",
      "is-variant-fail",
    );
    elements.transitionNextPanel?.classList.add(`is-variant-${nextTone}`);
    renderTransitionImageFrame(
      elements.transitionPreviousFrame,
      previous,
      previousStatePayload,
      runPayload?.dashboard?.visualConfig || null,
      "Previous state image not available",
    );
    renderTransitionImageFrame(
      elements.transitionNextFrame,
      nextArtifact,
      nextStatePayload,
      runPayload?.dashboard?.visualConfig || null,
      focusedWitness
        ? "Selected witness image not available for this focus"
        : selection.activeVariant === "fail" && !selectedFail?.image?.url
        ? buildTransitionFailureEmptyState(selectedFail, cachedBundleMeta)
        : "Next state image not available for this selection",
    );
  }
  if (shouldRenderPanel("transitionStageWitnesses", {
    galleryRevision: transitionGalleryRevisionToken(runPayload),
    activeVariant: selection.activeVariant,
    failureId: selectedFail?.failureId || null,
    focusMode: selection.focusMode || "target",
    focusedWitnessIndex,
    bundleMeta: transitionBundleMetaSignature(cachedBundleMeta),
    groupPresentation: buildTransitionWitnessGroupPresentationSignature(runPayload, cachedBundleMeta),
  })) {
    if (selection.activeVariant === "fail") {
      renderTransitionWitnessStrip(cachedBundleMeta, selection, runPayload);
    } else if (elements.transitionWitnessStrip) {
      elements.transitionWitnessStrip.innerHTML = "";
      elements.transitionWitnessStrip.hidden = true;
    }
  }
}

function renderActive() {
  const payload = activeRun();
  ensureRunPanelCache(payload?.runId || "none");
  const currentStatus = normalizeRunStatus(payload);
  const statusClasses = `status-pill ${statusClassName(currentStatus)}`;
  if (shouldRenderPanel("status", { text: currentStatus, className: statusClasses })) {
    elements.status.textContent = currentStatus;
    elements.status.className = statusClasses;
  }

  if (!payload) {
    applyLiveFooterLayout(null);
    if (elements.hudTitle && shouldRenderPanel("hudTitle", "HUD Metrics")) {
      elements.hudTitle.textContent = "HUD Metrics";
    }
    if (shouldRenderPanel("meta", "discovery run unavailable")) {
      elements.meta.textContent = "discovery run unavailable";
    }
    if (shouldRenderPanel("configSnapshots", "")) {
      renderConfigSnapshots(null);
    }
    if (shouldRenderPanel("log", "")) {
      elements.log.textContent = "";
    }
    if (shouldRenderPanel("dashboardEmpty", "dashboard unavailable")) {
      clearDashboardPanels();
    }
    if (shouldRenderPanel("transitionVersionSummary", "")) {
      clearTransitionNavigator();
    }
    if (shouldRenderPanel("transitionStageMode", "")) {
      clearTransitionStage();
    }
    if (shouldRenderPanel("programSourcePanel", "unavailable")) {
      clearProgramSourcePanel("Discovery run unavailable.");
    }
    renderRunDeleteModal(null);
    return;
  }

  const dashboard = payload.dashboard || {};
  applyLiveFooterLayout(dashboard);
  const transitionResolved = resolveTransitionSelection(payload);
  const programVersions = normalizeProgramVersions(payload);
  const isTransitionMode = transitionResolved.selection.mode === "version";
  const dashboardRevision = dashboardRevisionToken(dashboard, payload.runId);
  const intrinsicCurrentPanelSpec = deriveIntrinsicCurrentPanelSpec(dashboard.dashboardSpec?.bottom_left || {});
  const intrinsicMeanPanelSpec = deriveIntrinsicMeanPanelSpec(dashboard.dashboardSpec?.bottom_left || {});
  const panelEnabled = syncDashboardPanelVisibility(dashboard);
  const compositeWidth = Math.round(Math.max(
    0,
    Number(elements.dashboardComposite?.clientWidth || elements.dashboardComposite?.getBoundingClientRect().width || 0),
  ));
  const layoutSignature = stableSignature({
    compositeWidth,
    viewportWidth: window.innerWidth,
    panelEnabled,
  });
  if (shouldRenderPanel("layout", layoutSignature)) {
    applyDashboardLayout(dashboard, panelEnabled);
  }
  setViewerStageMode(isTransitionMode ? "transition" : "live");
  const hudTitleText = formatAgentHeading(
    dashboard?.agent?.name || dashboard?.diagnostics?.name,
  );
  if (elements.hudTitle && shouldRenderPanel("hudTitle", hudTitleText)) {
    elements.hudTitle.textContent = hudTitleText;
  }
  const topLeftTitle = dashboard.dashboardSpec?.top_left?.title || "Top Left";
  if (shouldRenderPanel("topLeftTitle", topLeftTitle)) {
    elements.topLeftTitle.textContent = topLeftTitle;
  }
  const middleLeftTitle = intrinsicCurrentPanelSpec.title || "Intrinsic Current Reward";
  if (shouldRenderPanel("middleLeftTitle", middleLeftTitle)) {
    elements.middleLeftTitle.textContent = middleLeftTitle;
  }
  const bottomLeftTitle = intrinsicMeanPanelSpec.title || "Intrinsic Mean Reward";
  if (shouldRenderPanel("bottomLeftTitle", bottomLeftTitle)) {
    elements.bottomLeftTitle.textContent = bottomLeftTitle;
  }
  const summary = payload.summary || {};
  const llmCallCount = resolveLlmCallCount(summary);
  const llmRuntimeLine = formatLlmRuntimeLine(summary, payload);
  const agentTypeLine = formatAgentTypeLine(payload);
  const currentVersionId = String(dashboard?.agent?.current_version_id || "").trim();
  const prototypeCount = resolvePrototypeCount(payload);
  const metaLines = [
    `run output dir: ${payload.runOutputDir || "-"}`,
    `current version: ${currentVersionId || summary.latestVersion || "-"}`,
    `program versions: ${programVersions.versions.length}`,
    `prototype cnt: ${prototypeCount ?? "-"}`,
    `new transition bundles: ${summary.newTransitionBundleCount ?? 0}`,
    `llm calls: ${llmCallCount ?? "-"}`,
    llmRuntimeLine,
    agentTypeLine,
  ];
  const metaText = formatLines(metaLines);
  if (shouldRenderPanel("meta", metaText)) {
    elements.meta.textContent = metaText;
  }
  const configSnapshotSignature = stableSignature(
    buildConfigSnapshotRenderSnapshot(summary, payload.runId),
  );
  if (shouldRenderPanel("configSnapshots", configSnapshotSignature)) {
    renderConfigSnapshots(payload);
  }

  const normalizedLog = normalizeTerminalLog(payload.logTail || "");
  if (shouldRenderPanel("log", { runId: payload.runId || null, text: normalizedLog })) {
    elements.log.textContent = normalizedLog;
  }
  restoreLogScrollState(payload);

  const boardScene = buildStateScene(dashboard.boardState || null, dashboard.visualConfig || null);
  const boardSignature = stableSignature({
    scene: boardScene,
    host: hostSizeSignature(elements.board),
    mode: isTransitionMode ? "transition" : "live",
  });
  if (!isTransitionMode && shouldRenderPanel("board", boardSignature)) {
    renderBoard(boardScene, elements.board, {
      tileSize: 44,
      fitToHost: true,
      minTileSize: 14,
      maxTileSize: 56,
      fallbackHostHeight: Math.min(window.innerHeight * 0.72, 820),
      emptyMessage: "current scene unavailable",
    });
  }

  if (shouldRenderPanel("targetWorlds", buildTargetWorldsRenderSnapshot(payload))) {
    renderTargetWorldPanel(payload);
  }

  if (shouldRenderPanel("hud", hudSignature(dashboard))) {
    renderHudSections(dashboard);
  }

  let shouldResizeCharts = false;
  const topLeftSignature = stableSignature({
    enabled: panelEnabled.top_left,
    chart: lineChartSignature(dashboard.dashboardSpec?.top_left || {}, dashboard.history || {}, dashboard),
  });
  if (panelEnabled.top_left && shouldRenderPanel("topLeftChart", topLeftSignature)) {
    renderLineChart(elements.topLeftChart, dashboard.dashboardSpec?.top_left || {}, dashboard.history || {});
    shouldResizeCharts = true;
  }
  const middleLeftSignature = stableSignature({
    enabled: panelEnabled.bottom_left,
    chart: lineChartSignature(intrinsicCurrentPanelSpec, dashboard.history || {}, dashboard),
  });
  if (panelEnabled.bottom_left && shouldRenderPanel("middleLeftChart", middleLeftSignature)) {
    renderLineChart(elements.middleLeftChart, intrinsicCurrentPanelSpec, dashboard.history || {});
    shouldResizeCharts = true;
  }
  const bottomLeftSignature = stableSignature({
    enabled: panelEnabled.bottom_left,
    chart: lineChartSignature(intrinsicMeanPanelSpec, dashboard.history || {}, dashboard),
  });
  if (panelEnabled.bottom_left && shouldRenderPanel("bottomLeftChart", bottomLeftSignature)) {
    renderLineChart(elements.bottomLeftChart, intrinsicMeanPanelSpec, dashboard.history || {});
    shouldResizeCharts = true;
  }
  const projectionPanelSignature = stableSignature({
    enabled: panelEnabled.projection,
    chart: projectionSignature(payload),
  });
  const projectionTargetWorldsSignature = stableSignature({
    enabled: panelEnabled.projection,
    selector: buildProjectionTargetWorldLegendSnapshot(payload),
  });
  if (panelEnabled.projection && shouldRenderPanel("projectionTargetWorlds", projectionTargetWorldsSignature)) {
    renderProjectionTargetWorldLegend(payload);
  } else if (!panelEnabled.projection) {
    clearProjectionTargetWorldLegend();
  }
  if (panelEnabled.projection && shouldRenderPanel("projection", projectionPanelSignature)) {
    renderProjection(payload);
    shouldResizeCharts = true;
  }
  const probabilitiesPanelSignature = stableSignature({
    enabled: panelEnabled.class_probability,
    chart: probabilitiesSignature(payload),
  });
  if (panelEnabled.class_probability && shouldRenderPanel("probabilities", probabilitiesPanelSignature)) {
    renderProbabilities(payload);
    shouldResizeCharts = true;
  }
  const classRowsSignature = stableSignature({
    revision: dashboardRevision,
    rows: buildClassRowsRenderSnapshot(dashboard.classRows || null),
  });
  if (panelEnabled.projection && shouldRenderPanel("prototypeLegend", buildPrototypeLegendSnapshot(payload))) {
    renderPrototypeLegend(payload);
  }
  if (shouldRenderPanel("classRows", classRowsSignature)) {
    renderClassRows(payload);
  }
  renderTransitionNavigator(payload, transitionResolved);
  renderProgramSourcePanel(payload, transitionResolved);
  renderTransitionStage(payload, transitionResolved);
  renderRunDeleteModal(payload);
  if (shouldResizeCharts) {
    window.requestAnimationFrame(() => {
      resizeMountedCharts();
    });
  }
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  const previousActiveRunId = state.activeRunId;
  state.runs = payload.runs || [];
  state.canShutdownServer = Boolean(payload.canShutdownServer);
  state.activeRunId = chooseActiveRun(payload);
  setQueryRunId(state.activeRunId);
  if (state.activeRunId) {
    try {
      const activeRunSummary = state.runs.find((run) => run.runId === state.activeRunId) || null;
      const shouldRefreshPausedTerminalPayload = isPausedView()
        && Boolean(state.activeRunPayload)
        && previousActiveRunId === state.activeRunId
        && isTerminalRunStatus(activeRunSummary)
        && normalizeRunStatus(activeRunSummary) !== normalizeRunStatus(state.activeRunPayload);
      const shouldFetchActivePayload = !isPausedView()
        || !state.activeRunPayload
        || previousActiveRunId !== state.activeRunId
        || shouldRefreshPausedTerminalPayload;
      if (shouldFetchActivePayload) {
        const nextPayload = await fetchJson(`/api/runs/${encodeURIComponent(state.activeRunId)}`);
        state.activeRunPayload = nextPayload;
      }
      await sendViewerState(
        state.activeRunId,
        state.viewMode,
        previousActiveRunId !== state.activeRunId,
      );
    } catch (_error) {
      state.activeRunPayload = null;
    }
  } else {
    state.activeRunPayload = null;
  }
  updateShutdownButton();
  updateTabs();
  renderActive();
}

async function pollRuns() {
  if (state.pollIntervalEditing) return;
  if (state.pollInFlight) return;
  if (state.pollHandle) {
    window.clearTimeout(state.pollHandle);
    state.pollHandle = null;
  }
  state.pollInFlight = true;
  try {
    await loadRuns();
  } catch (error) {
    elements.meta.textContent = formatError(error);
  } finally {
    state.pollInFlight = false;
    scheduleNextPoll(livePollIntervalMs());
  }
}

window.addEventListener("resize", () => {
  if (state.activeRunPayload) {
    invalidatePanelCache(
      "layout",
      "board",
      "classRows",
      "topLeftChart",
      "middleLeftChart",
      "bottomLeftChart",
      "projection",
      "probabilities",
      "prototypeLegend",
    );
    renderActive();
    window.requestAnimationFrame(() => {
      resizeMountedCharts();
    });
  }
});

elements.viewModeButton?.addEventListener("click", () => {
  toggleViewMode().catch((error) => {
    elements.meta.textContent = formatError(error);
  });
});

elements.projectionModeButton?.addEventListener("click", () => {
  toggleProjectionMode().catch((error) => {
    elements.meta.textContent = formatError(error);
  });
});

elements.projectionSparseFilterInput?.addEventListener("change", () => {
  toggleProjectionSparseFilter().catch((error) => {
    elements.meta.textContent = formatError(error);
  });
});

elements.shutdownButton?.addEventListener("click", () => {
  requestServerShutdown().catch((error) => {
    state.shutdownInFlight = false;
    updateShutdownButton();
    elements.meta.textContent = formatError(error);
  });
});

window.addEventListener("focus", () => {
  scheduleLiveViewAutoPause();
  sendViewerState(state.activeRunId, state.viewMode, true).catch(() => {});
  if (isPausedView()) return;
  pollRuns().catch((error) => {
    elements.meta.textContent = formatError(error);
  });
});

document.addEventListener("visibilitychange", () => {
  scheduleLiveViewAutoPause();
  sendViewerState(state.activeRunId, state.viewMode, true).catch(() => {});
  if (document.hidden || isPausedView()) return;
  pollRuns().catch((error) => {
    elements.meta.textContent = formatError(error);
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!elements.evalModal?.hidden) {
    closeOfflineEvalModal();
    return;
  }
  if (!elements.runDeleteModal?.hidden) {
    closeRunDeleteModal();
    return;
  }
  if (!state.activeRunPayload || !isProgramInspectorOpen(state.activeRunPayload)) return;
  closeProgramInspector(state.activeRunPayload);
  renderActive();
});

elements.livePollInput?.addEventListener("focus", () => {
  state.pollIntervalEditing = true;
  if (state.pollHandle) {
    window.clearTimeout(state.pollHandle);
    state.pollHandle = null;
  }
});

elements.livePollInput?.addEventListener("blur", (event) => {
  state.pollIntervalEditing = false;
  updatePollInterval(event.currentTarget?.value);
});

elements.log?.addEventListener("scroll", () => {
  rememberLogScrollState();
});

elements.programCopy?.addEventListener("click", async () => {
  if (!state.programInspectorText) return;
  try {
    await navigator.clipboard.writeText(state.programInspectorText);
    if (!elements.programCopy) return;
    resetProgramCopyButton();
    elements.programCopy.classList.add("is-copied");
    elements.programCopy.title = "Copied";
    elements.programCopy.setAttribute("aria-label", "Copied");
    state.programCopyResetHandle = window.setTimeout(() => {
      resetProgramCopyButton();
    }, 1400);
  } catch (_error) {
    resetProgramCopyButton();
    if (elements.programCopy) {
      elements.programCopy.classList.add("is-copy-failed");
      elements.programCopy.title = "Copy failed";
      elements.programCopy.setAttribute("aria-label", "Copy failed");
    }
  }
});

elements.programClose?.addEventListener("click", () => {
  if (!state.activeRunPayload) return;
  closeProgramInspector(state.activeRunPayload);
  renderActive();
});

elements.evalModalClose?.addEventListener("click", () => {
  closeOfflineEvalModal();
});

elements.evalDatasetRoot?.addEventListener("change", (event) => {
  syncOfflineEvalDiscoveryJsonOptions(String(event.currentTarget?.value || "").trim());
  updateOfflineEvalModalStatus();
});

elements.evalDiscoveryJson?.addEventListener("change", () => {
  updateOfflineEvalModalStatus();
});

elements.evalScenarioSplit?.addEventListener("change", () => {
  syncOfflineEvalDiscoveryJsonOptions(
    String(elements.evalDatasetRoot?.value || "").trim(),
    String(elements.evalDiscoveryJson?.value || "").trim(),
  );
  updateOfflineEvalModalStatus();
});

elements.evalWorkers?.addEventListener("input", () => {
  updateOfflineEvalModalStatus();
});

elements.evalStartButton?.addEventListener("click", () => {
  startOfflineEvalRun().catch((error) => {
    elements.evalModalStatus.textContent = formatError(error);
  });
});

elements.compareStartButton?.addEventListener("click", () => {
  startCompareSession().catch((error) => {
    elements.evalModalStatus.textContent = formatError(error);
  });
});

elements.evalModal?.addEventListener("click", (event) => {
  if (event.target === elements.evalModal) {
    closeOfflineEvalModal();
  }
});

elements.runDeleteClose?.addEventListener("click", () => {
  closeRunDeleteModal();
});

elements.runDeleteCancel?.addEventListener("click", () => {
  closeRunDeleteModal();
});

elements.runDeleteInput?.addEventListener("input", (event) => {
  state.runDelete.inputText = String(event.currentTarget?.value || "");
  renderRunDeleteModal(state.activeRunPayload);
});

elements.runDeleteInput?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  submitRunDelete().catch((error) => {
    state.runDelete.busy = false;
    state.runDelete.notice = formatError(error);
    renderRunDeleteModal(state.activeRunPayload);
  });
});

elements.runDeleteConfirm?.addEventListener("click", () => {
  submitRunDelete().catch((error) => {
    state.runDelete.busy = false;
    state.runDelete.notice = formatError(error);
    renderRunDeleteModal(state.activeRunPayload);
  });
});

elements.runDeleteModal?.addEventListener("click", (event) => {
  if (event.target === elements.runDeleteModal && !state.runDelete.busy) {
    closeRunDeleteModal();
  }
});

elements.versionList?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action]");
  if (!button || !state.activeRunPayload) return;
  const action = String(button.dataset.discoveryAction || "").trim();
  if (action === "open-offline-eval") {
    openOfflineEvalFromSource(
      resolveOfflineEvalSource(state.activeRunPayload, button.dataset.versionKey),
    ).catch((error) => {
      elements.meta.textContent = formatError(error);
    });
    return;
  }
  if (action === "open-offline-class-analysis") {
    openOfflineEvalFromSource(
      resolveOfflineEvalSource(state.activeRunPayload, button.dataset.versionKey),
      { runMode: "class_purity" },
    ).catch((error) => {
      elements.meta.textContent = formatError(error);
    });
    return;
  }
  if (action === "select-live") {
    selectTransitionLive(state.activeRunPayload);
    return;
  }
  if (action === "select-live-patch") {
    selectTransitionLivePatch(state.activeRunPayload);
    return;
  }
  if (action === "select-version") {
    selectTransitionVersion(state.activeRunPayload, button.dataset.versionKey);
  }
});

elements.evalRerunButton?.addEventListener("click", () => {
  const source = resolveOfflineEvalSourceForRunSummary(state.offlineEval.runSummary);
  if (!source) {
    state.offlineEval.notice = "Current discovery run does not expose source for this version.";
    renderOfflineEvalDrawer();
    return;
  }
  openOfflineEvalModal(source, {
    runId: state.activeRunPayload?.runId,
    runMode: state.offlineEval.runSummary?.runMode || "accuracy",
  }).catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.evalCloseDrawer?.addEventListener("click", () => {
  clearOfflineEvalActiveState({ keepRuns: true });
  renderOfflineEvalDrawer();
});

elements.evalStatusFilter?.addEventListener("change", (event) => {
  state.offlineEval.statusFilter = String(event.currentTarget?.value || "failed");
  resetOfflineEvalClassResults();
  state.offlineEval.selectedFailureId = null;
  state.offlineEval.selectedClassRowKey = null;
  state.offlineEval.selectedClassRepresentativeId = null;
  state.offlineEval.failureDetail = null;
  state.offlineEval.loadingFailureId = null;
  state.offlineEval.classRepresentativeDetail = null;
  state.offlineEval.loadingClassRepresentativeId = null;
  renderOfflineEvalDrawer();
  loadOfflineEvalClassResults({ reset: true, forceDetailRefresh: true }).catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.evalClassList?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action='select-offline-eval-class']");
  if (!button) return;
  if (isOfflineEvalClassPurity(state.offlineEval.runSummary)) {
    selectOfflineEvalClassRow(button.dataset.rowKey);
    return;
  }
  const failureId = String(button.dataset.failureId || "").trim();
  selectOfflineEvalFailure(failureId).catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.evalDetailMeta?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action='select-offline-eval-class-representative']");
  if (!button) return;
  const representativeId = String(button.dataset.representativeId || "").trim();
  selectOfflineEvalClassRepresentative(representativeId).catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.evalClassLoadMore?.addEventListener("click", () => {
  if (!state.offlineEval.classPageHasMore || state.offlineEval.classPageLoading) return;
  loadOfflineEvalClassResults({ reset: false, forceDetailRefresh: false }).catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.evalImageGrid?.addEventListener("click", (event) => {
  const card = event.target.closest("[data-discovery-action='select-offline-eval-image']");
  if (!card) return;
  const kind = String(card.dataset.kind || "").trim();
  if (!OFFLINE_EVAL_IMAGE_KINDS.includes(kind)) return;
  const isActive = card.classList.contains("is-active");
  if (event.target.closest("a")) {
    if (!isActive) {
      event.preventDefault();
      state.offlineEval.activeImageKind = kind;
      updateOfflineEvalImageGridLayout(resolveOfflineEvalAvailableImages(currentOfflineEvalImageDetail()));
      applyOfflineEvalImageMeta(currentOfflineEvalImageDetail());
    }
    return;
  }
  state.offlineEval.activeImageKind = kind;
  updateOfflineEvalImageGridLayout(resolveOfflineEvalAvailableImages(currentOfflineEvalImageDetail()));
  applyOfflineEvalImageMeta(currentOfflineEvalImageDetail());
});

elements.evalImageGrid?.addEventListener("keydown", (event) => {
  const card = event.target.closest("[data-discovery-action='select-offline-eval-image']");
  if (!card) return;
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  const kind = String(card.dataset.kind || "").trim();
  if (!OFFLINE_EVAL_IMAGE_KINDS.includes(kind)) return;
  state.offlineEval.activeImageKind = kind;
  updateOfflineEvalImageGridLayout(resolveOfflineEvalAvailableImages(currentOfflineEvalImageDetail()));
  applyOfflineEvalImageMeta(currentOfflineEvalImageDetail());
});

elements.evalSaveAll?.addEventListener("click", () => {
  const action = isOfflineEvalExportActive(state.offlineEval.exportStatus)
    ? cancelOfflineEvalExport
    : exportOfflineEvalFailures;
  action().catch((error) => {
    state.offlineEval.notice = formatError(error);
    renderOfflineEvalDrawer();
  });
});

elements.targetWorldList?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action='select-target-world']");
  if (!button || !state.activeRunPayload) return;
  selectTargetWorld(state.activeRunPayload, button.dataset.worldIndex);
});

elements.projectionTargetWorldList?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action='toggle-projection-target-world']");
  if (!button || !state.activeRunPayload || button.hasAttribute("disabled")) return;
  toggleProjectionPinnedWorld(state.activeRunPayload, button.dataset.worldIndex);
});

elements.projectionTargetWorldsHead?.addEventListener("click", () => {
  if (!state.activeRunPayload) return;
  toggleProjectionTargetWorldPanel(state.activeRunPayload);
});

elements.projectionTargetWorldsHead?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  if (!state.activeRunPayload) return;
  event.preventDefault();
  toggleProjectionTargetWorldPanel(state.activeRunPayload);
});

elements.targetWorldsHead?.addEventListener("click", () => {
  if (!state.activeRunPayload) return;
  toggleTargetWorldPanel(state.activeRunPayload);
});

elements.targetWorldsHead?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  if (!state.activeRunPayload) return;
  event.preventDefault();
  toggleTargetWorldPanel(state.activeRunPayload);
});

elements.transitionToolbar?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action]");
  if (!button) return;
  if (!state.activeRunPayload) return;
  if (button.hasAttribute("disabled")) return;
  const action = String(button.dataset.discoveryAction || "").trim();
  if (action === "select-target") {
    selectTransitionTarget(
      state.activeRunPayload,
      button.dataset.targetVariant,
      button.dataset.failureId,
    );
    return;
  }
  if (action === "open-accepted-program") {
    openAcceptedProgramInspector(state.activeRunPayload);
    renderActive();
    return;
  }
  if (action === "open-attempt-trace") {
    openAttemptTraceInspector(state.activeRunPayload);
    renderActive();
  }
});

elements.programTabs?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-discovery-action]");
  if (!button || !state.activeRunPayload) return;
  const action = String(button.dataset.discoveryAction || "").trim();
  if (action === "select-program-attempt") {
    updateProgramInspector(state.activeRunPayload, {
      attemptIndex: optionalInteger(button.dataset.attemptIndex) ?? 0,
      generatorIndex: 0,
      artifactKey: null,
    });
    renderActive();
    return;
  }
  if (action === "select-program-generator") {
    updateProgramInspector(state.activeRunPayload, {
      generatorIndex: optionalInteger(button.dataset.generatorIndex) ?? 0,
      artifactKey: null,
    });
    renderActive();
    return;
  }
  if (action === "select-program-artifact") {
    updateProgramInspector(state.activeRunPayload, {
      artifactKey: String(button.dataset.artifactKey || "").trim() || null,
    });
    renderActive();
  }
});

elements.transitionWitnessStrip?.addEventListener("click", (event) => {
  const actionTarget = event.target.closest("[data-discovery-action]");
  if (!actionTarget || !state.activeRunPayload) return;
  if (event.target.closest("a")) return;
  const action = String(actionTarget.dataset.discoveryAction || "").trim();
  if (action === "toggle-reject-fail-groups") {
    toggleRejectFailGroups(
      state.activeRunPayload,
      actionTarget.dataset.failGroupCount,
    );
    return;
  }
  if (action === "toggle-reject-splits") {
    toggleRejectSplits(
      state.activeRunPayload,
      actionTarget.dataset.splitCount,
    );
    return;
  }
  if (action === "reset-witness-focus") {
    resetTransitionWitnessFocus(state.activeRunPayload);
    return;
  }
  if (action === "select-witness") {
    selectTransitionWitness(
      state.activeRunPayload,
      actionTarget.dataset.witnessIndex,
    );
  }
});

elements.transitionWitnessStrip?.addEventListener("keydown", (event) => {
  const actionTarget = event.target.closest("[data-discovery-action='select-witness']");
  if (!actionTarget || !state.activeRunPayload) return;
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  selectTransitionWitness(
    state.activeRunPayload,
    actionTarget.dataset.witnessIndex,
  );
});

elements.pageTitle?.addEventListener("click", () => {
  triggerDiscoveryPageTitleRotationNow();
});

elements.pageTitle?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  triggerDiscoveryPageTitleRotationNow();
});

syncPollIntervalInputs();
updateViewModeButton();
scheduleLiveViewAutoPause();
updateProjectionModeButton();
updateProjectionSparseFilterInput();
updateShutdownButton();
initializeDiscoveryPageTitleRotation();
renderOfflineEvalDrawer();
loadMapDisplayNames()
  .catch(() => null)
  .finally(() => {
    loadLatestOfflineEvalRunMaybe().catch(() => {});
    pollRuns().catch((error) => {
      elements.meta.textContent = formatError(error);
    });
  });
