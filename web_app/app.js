const DEFAULT_STATE = {
  active: false,
  phase: "idle",
  startedAt: null,
  endsAt: null,
  lastNotificationCreatedAt: null,
  lastCalibrationKey: null,
  calibrationNotificationPending: false,
  latestSessionSummary: null,
  latestTrace: null,
  privacyLedger: null,
  memoryProfile: null,
  summaryLoading: false,
  pi: {
    ip: "",
    user: "",
    path: ""
  },
  connectInfo: null,
  moreOpen: false,
  settings: {
    intent: "",
    tone: "neutral",
    notification_level: "minimal",
    focus_areas: ["posture", "restlessness", "movement", "declutter"],
    pomodoro_minutes: 25,
    break_minutes: 5,
    work_struggles: {
      selected: [],
      notes: "",
      support_preferences: ""
    }
  }
};

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "flowpilot.pwa.state";
const CALIBRATION_SECONDS = 120;

let state = loadState();
let latestStatus = null;
let calibration = null;
let connectInfo = state.connectInfo || null;
let tickTimer = null;
let pollTimer = null;

function loadState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return mergeState(stored);
  } catch (_error) {
    return mergeState({});
  }
}

function mergeState(value) {
  return {
    ...DEFAULT_STATE,
    ...value,
    settings: {
      ...DEFAULT_STATE.settings,
      ...(value.settings || {}),
      work_struggles: {
        ...DEFAULT_STATE.settings.work_struggles,
        ...((value.settings && value.settings.work_struggles) || {})
      }
    },
    pi: {
      ...DEFAULT_STATE.pi,
      ...(value.pi || {})
    }
  };
}

function saveState(nextState = state) {
  state = mergeState(nextState);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function selectedValues(containerId) {
  return [...document.querySelectorAll(`#${containerId} input:checked`)].map((item) => item.value);
}

function settingsFromForm() {
  return {
    intent: $("intent").value.trim(),
    tone: $("tone").value,
    notification_level: $("notificationLevel").value,
    focus_areas: selectedValues("focusAreas"),
    pomodoro_minutes: Math.max(1, Number($("pomodoroMinutes").value || 25)),
    break_minutes: Math.max(1, Number($("breakMinutes").value || 5)),
    work_struggles: {
      selected: selectedValues("struggles"),
      notes: $("struggleNotes").value.trim(),
      support_preferences: $("supportPreferences").value.trim()
    }
  };
}

function applyForm() {
  $("intent").value = state.settings.intent || "";
  $("tone").value = state.settings.tone || "neutral";
  $("notificationLevel").value = state.settings.notification_level || "minimal";
  $("pomodoroMinutes").value = state.settings.pomodoro_minutes || 25;
  $("breakMinutes").value = state.settings.break_minutes || 5;
  document.querySelectorAll("#focusAreas input").forEach((input) => {
    input.checked = state.settings.focus_areas.includes(input.value);
  });
  const struggles = state.settings.work_struggles || {};
  document.querySelectorAll("#struggles input").forEach((input) => {
    input.checked = (struggles.selected || []).includes(input.value);
  });
  $("struggleNotes").value = struggles.notes || "";
  $("supportPreferences").value = struggles.support_preferences || "";
  $("piIp").value = state.pi.ip || "";
  $("piUser").value = state.pi.user || "";
  $("piPath").value = state.pi.path || "";
  renderPiCommands();
}

function syncFormToState() {
  saveState({
    ...state,
    settings: settingsFromForm(),
    pi: {
      ip: $("piIp").value.trim(),
      user: $("piUser").value.trim(),
      path: $("piPath").value.trim()
    }
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function formatRemaining(endsAt) {
  if (!endsAt) return `${String(state.settings.pomodoro_minutes).padStart(2, "0")}:00`;
  const remaining = Math.max(0, new Date(endsAt).getTime() - Date.now());
  const totalSeconds = Math.ceil(remaining / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function calibrationIsBusy() {
  return calibration && ["collecting", "classifying_objects"].includes(calibration.status);
}

function calibrationIsCollecting() {
  return calibration && calibration.status === "collecting";
}

function statusText() {
  if (!latestStatus) return "Connecting to Bamboo Focus app...";
  if (calibration && calibration.status === "collecting") {
    return `Calibrating baseline: ${calibration.event_count || 0} events received.`;
  }
  if (calibration && calibration.status === "classifying_objects") {
    return "Finishing calibration and object policy.";
  }
  if (state.active && state.phase === "focus") return "Focus session is running. Nudges are enabled.";
  if (state.active && state.phase === "break") return "Break is running. Nudges are paused.";
  if (calibration && calibration.status !== "complete") return "Calibrate before starting a focus session.";
  return "Ready.";
}

function render() {
  const calibrating = calibrationIsCollecting();
  const finishingCalibration = calibration && calibration.status === "classifying_objects";
  $("timerValue").textContent = calibrating ? formatRemaining(calibration.ends_at) : finishingCalibration ? "00:00" : formatRemaining(state.endsAt);
  $("phaseLabel").textContent = calibrating ? "calibrating" : finishingCalibration ? "finishing" : state.phase === "break" ? "break" : state.active ? "focus" : "ready";
  $("primaryAction").textContent = calibrationIsBusy() ? "Calibrating" : state.active ? "Stop session" : "Start focus";
  $("primaryAction").disabled = calibrationIsBusy();
  $("calibrate").disabled = calibrationIsBusy();
  $("statusLine").textContent = statusText();
  const modePill = $("modePill");
  if (modePill) {
    modePill.textContent = latestStatus
      ? `local app: ${latestStatus.nudge_agent && latestStatus.nudge_agent.mode ? latestStatus.nudge_agent.mode : "ready"}`
      : "local app: offline";
  }

  const calibrationStatus = calibration && calibration.status ? calibration.status : "unknown";
  $("calibrationStatus").textContent = calibrationStatus;
  $("nudgeMode").textContent = latestStatus && latestStatus.nudge_agent ? latestStatus.nudge_agent.mode || "auto" : "unknown";
  const postureCount = latestStatus && latestStatus.posture_monitor ? latestStatus.posture_monitor.posture_windows : null;
  const objectCount = latestStatus && latestStatus.object_monitor ? latestStatus.object_monitor.snapshots : null;
  $("piStatus").textContent = postureCount || objectCount ? `${postureCount || 0} posture / ${objectCount || 0} object` : "waiting";
  $("cloudStatus").textContent = "optional sync off";
  if (latestStatus && latestStatus.privacy) {
    $("cloudStatus").textContent = latestStatus.privacy.cloud_attempted_for_latest_decision ? "Qwen summaries" : "local only";
  }
  const summary = state.latestSessionSummary;
  $("latestInsight").textContent = summary && summary.paragraph ? summary.paragraph : "No session summary yet.";
  renderPrivacy();
  renderTrace();
  renderMemory();
  const showLoadingInsight = !state.active && state.summaryLoading;
  const calibrationInsight = calibrationIsBusy()
    ? "Start working how you normally would while your posture and surroundings are scanned."
    : "";
  const showMainInsight = !state.active && !state.summaryLoading && !calibrationInsight && summary && summary.paragraph;
  $("mainInsight").textContent = calibrationInsight || (showLoadingInsight ? "..." : showMainInsight ? summary.paragraph : "");
  $("mainInsight").classList.toggle("hidden", !calibrationInsight && !showLoadingInsight && !showMainInsight);
  $("mainInsight").classList.toggle("loading-dots", showLoadingInsight && !calibrationInsight);
  $("moreSections").classList.toggle("hidden", !state.moreOpen);
  $("moreToggle").textContent = state.moreOpen ? "less ↑" : "more ↓";
  updatePanda();
  renderPiCommands();
}

function renderPrivacy() {
  const privacy = state.privacyLedger;
  const hardware = privacy && privacy.hardware_first ? privacy.hardware_first : {};
  const boundaries = privacy && privacy.data_boundaries ? privacy.data_boundaries : {};
  $("privacyVideo").textContent = hardware.raw_video_sent_off_device === false ? "never leaves edge" : "unknown";
  $("privacyFrames").textContent = hardware.raw_frames_persisted_by_backend === false ? "not persisted" : "unknown";
  $("privacyCloud").textContent = boundaries.cloud_attempted_for_latest_decision ? "compact summaries only" : "not used for latest decision";
  $("privacyProvider").textContent = boundaries.cloud_provider || "local only";
}

function renderTrace() {
  const trace = state.latestTrace;
  const decision = trace && trace.decision ? trace.decision : {};
  const why = trace && trace.why ? trace.why : {};
  const edge = trace && trace.edge_evidence ? trace.edge_evidence : {};
  const agentPath = trace && trace.agent_path ? trace.agent_path : {};
  const memory = trace && trace.memory_influence ? trace.memory_influence : {};
  const rag = trace && trace.history_rag ? trace.history_rag : {};
  $("traceDecision").textContent = decision.category
    ? `${decision.should_nudge ? "nudged" : "suppressed"}: ${decision.category}`
    : "No decision yet.";
  $("traceWhy").textContent = why.rationale || why.suppress_reason || "Waiting for the first nudge cycle.";
  $("traceEvidence").textContent = trace
    ? `${edge.posture_analysis_count || 0} posture / ${edge.object_snapshot_count || 0} object / ${edge.object_dwell_candidate_count || 0} dwell`
    : "No edge evidence yet.";
  $("traceMemory").textContent = trace
    ? `${memory.used ? "used" : "not used"} (${memory.session_count || 0} sessions): ${memory.guidance_summary || "No memory guidance yet."}`
    : "not used yet";
  $("traceRag").textContent = trace
    ? rag.available
      ? `${rag.match_count || 0} matches for "${rag.query || "history"}"`
      : "not searched in this path"
    : "not searched yet";
  $("tracePath").textContent = agentPath.local_fallback_used ? "local fallback/rules" : agentPath.mode ? `${agentPath.mode} agent` : "not run yet";
}

function renderMemory() {
  const memory = state.memoryProfile;
  if (!memory || !memory.session_count) {
    $("memoryProfile").textContent = "No completed sessions in memory yet.";
    return;
  }
  const totals = memory.totals || {};
  const guidance = memory.adaptive_guidance || {};
  $("memoryProfile").textContent = `${memory.session_count} sessions remembered, ${totals.nudge_count || 0} nudges, ${totals.slouching || 0} posture flags, ${totals.restless || 0} restless flags. ${guidance.summary || ""}`.trim();
}

function updatePanda() {
  const image = $("pandaImage");
  if (!image) return;
  if (calibrationIsBusy()) {
    image.src = "/assets/panda_1.png";
    return;
  }
  if (state.phase === "break") {
    image.src = "/assets/panda_8.png";
    return;
  }
  if (!state.active || state.phase !== "focus" || !state.endsAt) {
    image.src = "/assets/panda_1.png";
    return;
  }
  const totalMs = Math.max(1, state.settings.pomodoro_minutes * 60 * 1000);
  const remainingMs = Math.max(0, new Date(state.endsAt).getTime() - Date.now());
  const elapsed = Math.min(totalMs, totalMs - remainingMs);
  const frame = Math.min(8, Math.floor((elapsed / totalMs) * 8) + 1);
  image.src = `/assets/panda_${frame}.png`;
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

function renderPiCommands() {
  if (!$("laptopApiBase")) return;
  const apiBase = connectInfo && connectInfo.api_base
    ? connectInfo.api_base
    : `${window.location.protocol}//${window.location.host}`;
  const token = connectInfo && connectInfo.token ? connectInfo.token : "dev-local-token";
  const piPath = $("piPath") ? $("piPath").value.trim() || "<PROJECT_PATH_ON_PI>" : state.pi.path || "<PROJECT_PATH_ON_PI>";
  const piUser = $("piUser") ? $("piUser").value.trim() || "<PI_SSH_USER>" : state.pi.user || "<PI_SSH_USER>";
  const piIp = $("piIp") ? $("piIp").value.trim() : state.pi.ip;
  const piCommand = `cd ${piPath} && python pi_start.py --laptop-api-base ${shellQuote(apiBase)} --token ${shellQuote(token)} --download-object-model --download-pose-model`;
  const webcamCommand = connectInfo && connectInfo.webcam_command
    ? connectInfo.webcam_command
    : `python webcam_edge.py --laptop-api-base ${shellQuote(`${window.location.protocol}//${window.location.host}`)} --token ${shellQuote(token)} --download-object-model --download-pose-model --debug-stream`;
  const sshTarget = piIp ? `${piUser}@${piIp}` : `${piUser}@<PI_IP_ADDRESS>`;
  $("laptopApiBase").value = apiBase;
  $("piCommand").value = piCommand;
  $("sshCommand").value = `ssh ${sshTarget} ${shellQuote(piCommand)}`;
  if ($("webcamCommand")) $("webcamCommand").value = webcamCommand;
}

async function syncSessionSettings(activeOverride = null) {
  syncFormToState();
  const active = activeOverride === null ? state.active : activeOverride;
  const phase = active ? (activeOverride === true ? "focus" : state.phase) : "idle";
  const startedAt = activeOverride === true ? new Date().toISOString() : active ? state.startedAt : null;
  const endsAt = activeOverride === true
    ? new Date(Date.now() + state.settings.pomodoro_minutes * 60 * 1000).toISOString()
    : active ? state.endsAt : null;
  saveState({ ...state, active, phase, startedAt, endsAt });
  await api("/api/session-settings", {
    method: "POST",
    body: JSON.stringify({
      active: active && phase === "focus",
      intent: state.settings.intent,
      tone: state.settings.tone,
      notification_level: state.settings.notification_level,
      focus_areas: state.settings.focus_areas,
      pomodoro_minutes: state.settings.pomodoro_minutes,
      break_minutes: state.settings.break_minutes,
      work_struggles: state.settings.work_struggles,
      started_at: active && phase === "focus" ? startedAt : null,
      ends_at: active && phase === "focus" ? endsAt : null
    })
  });
  render();
}

async function startSession() {
  await refreshCalibration();
  if (!calibration || calibration.status !== "complete") {
    toast("Calibrate baseline before starting.");
    return;
  }
  await requestNotificationsQuietly();
  await syncSessionSettings(true);
  toast("Focus session started.");
}

async function stopSession() {
  const endingState = { ...state };
  const endedAt = new Date().toISOString();
  saveState({ ...state, summaryLoading: Boolean(endingState.startedAt) });
  render();
  await syncSessionSettings(false);
  if (endingState.startedAt) {
    try {
      await createSessionSummary(endingState, endedAt);
    } catch (error) {
      saveState({ ...state, summaryLoading: false });
      render();
      throw error;
    }
  }
  toast("Session stopped.");
}

async function completeFocusPeriod() {
  if (!state.active || state.phase !== "focus") return;
  const breakEndsAt = new Date(Date.now() + state.settings.break_minutes * 60 * 1000).toISOString();
  saveState({ ...state, phase: "break", endsAt: breakEndsAt });
  await api("/api/session-settings", {
    method: "POST",
    body: JSON.stringify({ ...state.settings, active: false, started_at: null, ends_at: null })
  });
  notify("Focus time is done", `Time for a ${state.settings.break_minutes}-minute break.`);
  toast("Break started.");
  render();
}

async function completeBreakPeriod() {
  if (!state.active || state.phase !== "break") return;
  const endingState = { ...state };
  const endedAt = new Date().toISOString();
  saveState({ ...state, active: false, phase: "idle", startedAt: null, endsAt: null, summaryLoading: true });
  await api("/api/session-settings", {
    method: "POST",
    body: JSON.stringify({ ...state.settings, active: false, started_at: null, ends_at: null })
  });
  notify("Break complete", "Time to resume when you are ready.");
  try {
    await createSessionSummary(endingState, endedAt);
  } catch (error) {
    saveState({ ...state, summaryLoading: false });
    render();
    throw error;
  }
  render();
}

async function handleTimer() {
  if (!state.active || !state.endsAt) return;
  if (Date.now() < new Date(state.endsAt).getTime()) return;
  if (state.phase === "focus") {
    await completeFocusPeriod();
  } else if (state.phase === "break") {
    await completeBreakPeriod();
  }
}

async function createSessionSummary(sessionState, endedAt) {
  const result = await api("/api/session-summary", {
    method: "POST",
    body: JSON.stringify({
      started_at: sessionState.startedAt,
      ended_at: endedAt,
      settings: sessionState.settings
    })
  });
  if (result.summary) {
    saveState({ ...state, latestSessionSummary: result.summary, summaryLoading: false });
    render();
  } else {
    saveState({ ...state, summaryLoading: false });
    render();
  }
}

async function refreshStatus() {
  try {
    latestStatus = await api("/api/status");
    calibration = latestStatus.calibration || calibration;
    render();
  } catch (error) {
    latestStatus = null;
    $("statusLine").textContent = `Local app unavailable: ${error.message}`;
    render();
  }
}

async function refreshConnectInfo() {
  try {
    const result = await api("/api/connect-info");
    connectInfo = result && result.connect_info ? result.connect_info : null;
    saveState({ ...state, connectInfo });
    renderPiCommands();
  } catch (_error) {
    renderPiCommands();
  }
}

async function refreshCalibration() {
  const result = await api("/api/calibration");
  const previousStatus = calibration && calibration.status;
  calibration = result.calibration || null;
  const key = calibration && (calibration.completed_at || calibration.latest_baseline_path);
  const completedAfterStart = state.calibrationNotificationPending && calibration && calibration.status === "complete";
  const completedTransition = calibration && calibration.status === "complete" && previousStatus && previousStatus !== "complete";
  if ((completedAfterStart || completedTransition) && state.lastCalibrationKey !== key) {
    saveState({ ...state, lastCalibrationKey: key, calibrationNotificationPending: false });
    notify("Calibration complete", "Baseline is ready. Bamboo Focus can start monitoring from the new setup.");
  } else if (calibration && calibration.status === "complete" && state.calibrationNotificationPending) {
    saveState({ ...state, calibrationNotificationPending: false });
  }
  render();
}

async function recalibrate() {
  syncFormToState();
  saveState({ ...state, calibrationNotificationPending: true });
  await api("/api/recalibrate", {
    method: "POST",
    body: JSON.stringify({ seconds: CALIBRATION_SECONDS })
  });
  await refreshCalibration();
  toast("Calibration started.");
}

async function pollNotification() {
  if (!state.active || state.phase !== "focus") return;
  const result = await api("/api/latest-notification");
  const record = result.notification;
  if (!record || !record.notification || record.created_at === state.lastNotificationCreatedAt) return;
  saveState({ ...state, lastNotificationCreatedAt: record.created_at });
  notify(record.notification.title || "Bamboo Focus", record.notification.message || "Quick check-in.");
}

async function fetchLatestSummary() {
  try {
    const result = await api("/api/latest-session-summary");
    if (result.summary) {
      saveState({ ...state, latestSessionSummary: result.summary });
    }
  } catch (_error) {
    // Optional.
  }
}

async function refreshPrivacyLedger() {
  try {
    const result = await api("/api/privacy-ledger");
    saveState({ ...state, privacyLedger: result.privacy || null });
  } catch (_error) {
    // Optional.
  }
}

async function refreshExplainability() {
  try {
    const result = await api("/api/explainability");
    saveState({ ...state, latestTrace: result.trace || null });
  } catch (_error) {
    // Optional.
  }
}

async function refreshMemoryProfile() {
  try {
    const result = await api("/api/memory-profile");
    saveState({ ...state, memoryProfile: result.memory || null });
  } catch (_error) {
    // Optional.
  }
}

function notify(title, message) {
  if (!("Notification" in window) || Notification.permission !== "granted") {
    toast(`${title}: ${message}`);
    return;
  }
  new Notification(title, {
    body: message,
    icon: "panda_head.png"
  });
}

async function requestNotificationsQuietly() {
  if (!("Notification" in window) || Notification.permission !== "default") return;
  try {
    await Notification.requestPermission();
  } catch (_error) {
    // The explicit button remains available.
  }
}

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.remove("hidden");
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => element.classList.add("hidden"), 4200);
}

function bindEvents() {
  $("settingsToggle").addEventListener("click", () => setDrawerOpen(true));
  $("closeSettings").addEventListener("click", () => setDrawerOpen(false));
  $("drawerBackdrop").addEventListener("click", () => setDrawerOpen(false));
  $("moreToggle").addEventListener("click", () => {
    saveState({ ...state, moreOpen: !state.moreOpen });
    render();
  });
  $("copySshCommand").addEventListener("click", () => copyText($("sshCommand").value, "SSH command copied."));
  $("copyPiCommand").addEventListener("click", () => copyText($("piCommand").value, "Pi command copied."));
  $("copyWebcamCommand").addEventListener("click", () => copyText($("webcamCommand").value, "Webcam demo command copied."));
  ["piIp", "piUser", "piPath"].forEach((id) => {
    $(id).addEventListener("input", () => {
      syncFormToState();
      renderPiCommands();
    });
  });
  $("primaryAction").addEventListener("click", async () => {
    try {
      if (state.active) {
        await stopSession();
      } else {
        await startSession();
      }
    } catch (error) {
      toast(error.message || String(error));
    }
  });
  $("calibrate").addEventListener("click", async () => {
    try {
      await recalibrate();
    } catch (error) {
      toast(error.message || String(error));
    }
  });
  $("enableNotifications").addEventListener("click", async () => {
    if (!("Notification" in window)) {
      toast("This browser does not support notifications.");
      return;
    }
    const permission = await Notification.requestPermission();
    toast(permission === "granted" ? "Notifications enabled." : "Notifications were not enabled.");
  });
  document.querySelectorAll("input, select, textarea").forEach((input) => {
    if (["piIp", "piUser", "piPath"].includes(input.id)) return;
    input.addEventListener("change", async () => {
      syncFormToState();
      if (state.active && state.phase === "focus") {
        try {
          await syncSessionSettings(null);
        } catch (error) {
          toast(error.message || String(error));
        }
      }
      render();
    });
  });
}

function setDrawerOpen(open) {
  $("settingsDrawer").classList.toggle("open", open);
  $("settingsDrawer").setAttribute("aria-hidden", open ? "false" : "true");
  $("drawerBackdrop").classList.toggle("hidden", !open);
}

async function copyText(value, message) {
  try {
    await navigator.clipboard.writeText(value);
    toast(message);
  } catch (_error) {
    toast("Could not copy. Select the command text manually.");
  }
}

async function init() {
  applyForm();
  bindEvents();
  render();
  await refreshStatus();
  await refreshConnectInfo();
  await refreshCalibration().catch(() => undefined);
  await fetchLatestSummary();
  await refreshPrivacyLedger();
  await refreshExplainability();
  await refreshMemoryProfile();
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("service-worker.js").catch(() => undefined);
  }
  tickTimer = setInterval(() => {
    render();
    handleTimer().catch((error) => toast(error.message || String(error)));
  }, 1000);
  pollTimer = setInterval(() => {
    refreshStatus().catch(() => undefined);
    refreshCalibration().catch(() => undefined);
    refreshPrivacyLedger().catch(() => undefined);
    refreshExplainability().catch(() => undefined);
    pollNotification().catch(() => undefined);
  }, 5000);
}

window.addEventListener("beforeunload", () => {
  clearInterval(tickTimer);
  clearInterval(pollTimer);
});

init();
