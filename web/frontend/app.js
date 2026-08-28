/* Dental Implant Planner · Web — frontend logic */
(() => {
  "use strict";

  // ------------------------------------------------------------- state
  const state = {
    sid: null,
    slices: 0,
    shape: null,
    curSlice: 0,
    viewMode: "base",        // base | overlay
    blend: false,
    missing: { start: null, end: null },
    candidates: [],
    applied: null,
    logs: [],
    noImplantReason: null,
    plants: [],
    busy: false,
    pickedFile: null,        // File object from the picker, not yet uploaded
  };

  // ------------------------------------------------------------- dom
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const el = {
    dropzone:     $("#dropzone"),
    dzFile:       $("#dzFile"),
    filePick:     $("#filePick"),
    btnPick:      $("#btnPick"),
    btnOpenFile:  $("#btnOpenFile"),
    btnClearFile: $("#btnClearFile"),
    patientDir:   $("#patientDir"),
    btnLoad:      $("#btnLoadPatient"),
    sessionMeta:  $("#sessionMeta"),
    metaShape:    $("#metaShape"),
    metaSlices:   $("#metaSlices"),
    metaSource:   $("#metaSource"),
    metaRange:    $("#metaRange"),
    patientPill:  $("#patientPill"),

    sliceSlider:  $("#sliceSlider"),
    sliceLabel:   $("#sliceLabel"),
    imgBase:      $("#imgBase"),
    imgOverlay:   $("#imgOverlay"),
    viewerEmpty:  $("#viewerEmpty"),
    legend:       $("#legend"),
    statusDot:    $("#statusDot"),
    statusLabel:  $("#statusLabel"),
    progressWrap: $("#progressWrap"),
    progressBar:  $("#progressBar"),

    angle:        $("#angle"),
    axis:         $("#axis"),
    btnRotL:      $("#btnRotateLeft"),
    btnRotR:      $("#btnRotateRight"),
    btnResetA:    $("#btnResetAngle"),

    missStart:    $("#missStart"),
    missEnd:      $("#missEnd"),
    btnSetRange:  $("#btnSetRange"),
    btnAuto:      $("#btnAutoRange"),

    btnRun:       $("#btnRun"),
    kpiImplants:  $("#kpiImplants"),
    kpiCand:      $("#kpiCandidates"),

    noCand:       $("#noCandidates"),
    noCandWhy:    $("#noCandidatesWhy"),
    appliedBadge: $("#appliedBadge"),
    appliedId:    $("#appliedId"),
    appliedD:     $("#appliedD"),
    appliedH:     $("#appliedH"),
    appliedMeta:  $("#appliedMeta"),
    candHead:     $("#candidatesHead"),
    candCount:    $("#candCount"),
    candList:     $("#candidates"),
    logDetails:   $("#logDetails"),
    logPre:       $("#logPre"),

    plantForm:    $("#plantForm"),
    plantList:    $("#plantList"),
    toast:        $("#toast"),
    footerSess:   $("#footerSess"),
  };

  // ------------------------------------------------------------- api
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-json */ }
    if (!res.ok) {
      const msg = data && (data.detail || data.message) ? (data.detail || data.message) : `HTTP ${res.status}`;
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  let toastTimer = null;
  function toast(msg, kind = "info", ms = 4200) {
    el.toast.textContent = msg;
    el.toast.className = "toast show " + kind;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.remove("show"), ms);
  }

  // ------------------------------------------------------------- status
  function setStatus(kind, label, progress = null) {
    el.statusDot.className = "dot" + (kind ? " " + kind : "");
    el.statusLabel.textContent = label;
    if (progress === null) {
      el.progressWrap.hidden = true;
      el.progressBar.style.width = "0%";
    } else {
      el.progressWrap.hidden = false;
      el.progressBar.style.width = Math.round(progress * 100) + "%";
    }
  }

  // ------------------------------------------------------------- steps
  let currentStep = 0;
  function setStep(n, opts = {}) {
    currentStep = n;
    $$(".step").forEach((s, i) => {
      s.classList.toggle("is-active", i === n);
      s.classList.toggle("is-done", i < n || (opts.done === true && i === n - 1) ? false : i < n);
    });
    $$(".step-panel").forEach((p) => p.classList.toggle("is-active", +p.dataset.step === n));
  }
  $$(".step").forEach((s) => s.addEventListener("click", () => setStep(+s.dataset.step)));

  // ------------------------------------------------------------- viewer
  function sliceUrl(z) { return `/api/session/${state.sid}/slice/${z}/png`; }
  function overlayUrl(z) { return `/api/session/${state.sid}/overlay/${z}.png?t=${Date.now()}`; }

  function updateViewer() {
    if (!state.sid) return;
    if (state.viewMode === "base") {
      el.imgBase.src = sliceUrl(state.curSlice);
      el.imgOverlay.classList.add("hide");
    } else {
      el.imgBase.src = sliceUrl(state.curSlice);
      el.imgOverlay.classList.remove("hide");
      el.imgOverlay.src = overlayUrl(state.curSlice);
      el.imgOverlay.addEventListener("error", () => {
        el.imgOverlay.classList.add("hide");
        toast("No overlay for this slice yet — run the pipeline first.", "info");
      }, { once: true });
    }
    if (state.blend) el.imgOverlay.classList.remove("img-blend");
    el.sliceLabel.textContent = `${state.curSlice} / ${state.slices - 1}`;
    el.sliceSlider.value = state.curSlice;
    // If the 3D view is active, refresh it with the new volume too
    if (window.Vol3D && window.Vol3D.active && state.sid) {
      window.Vol3D.refresh(state.sid).catch(() => {});
    }
  }

  // view-mode segment buttons (2D / Overlay / 3D)
  $$(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    if (b.id === "btn2d3d") { enter3D(); return; }
    // 2D buttons: exit 3D first if active
    if (window.Vol3D && window.Vol3D.active) {
      window.Vol3D.hide();
      $("#imgStack").style.display = "";
    }
    $$(".seg-btn").forEach((x) => x.classList.remove("is-active"));
    b.classList.add("is-active");
    state.viewMode = b.dataset.view;
    showToolbar3d(false);
    updateViewer();
  }));

  function showToolbar3d(on) {
    $("#ctl3dThreshold").style.display = on ? "inline-flex" : "none";
    $("#ctl3dOpacity").style.display = on ? "inline-flex" : "none";
  }

  let _vol3dPromise = null;
  function ensureVol3d() {
    if (!window.Vol3D) return null;
    if (!window.Vol3D.ready) {
      // lazy-import the module if it hasn't booted yet
      if (!_vol3dPromise) _vol3dPromise = import("./vol3d.js");
      return _vol3dPromise;
    }
    return Promise.resolve(window.Vol3D);
  }

  async function enter3D() {
    if (!state.sid) { toast("Open a session first.", "err"); return; }
    $$(".seg-btn").forEach((x) => x.classList.remove("is-active"));
    $("#btn2d3d").classList.add("is-active");
    showToolbar3d(true);
    if (window.Vol3D) {
      const V = await ensureVol3d();
      if (!V) { toast("3D module failed to load.", "err"); return; }
      try {
        await V.show(state.sid);
        V.setThreshold(parseInt($("#volThreshold").value, 10));
        V.setOpacity(parseInt($("#volOpacity").value, 10));
        $("#imgStack").style.display = "none";
      } catch (e) {
        toast("3D failed: " + e.message, "err", 7000);
      }
    } else {
      toast("3D module not ready — reload the page.", "err", 6000);
    }
  }

  $("#volThreshold").addEventListener("input", (e) => {
    if (window.Vol3D) window.Vol3D.setThreshold(parseInt(e.target.value, 10));
  });
  $("#volOpacity").addEventListener("input", (e) => {
    if (window.Vol3D) window.Vol3D.setOpacity(parseInt(e.target.value, 10));
  });

  $("#chkBlend").addEventListener("change", (e) => {
    state.blend = e.target.checked;
    el.imgOverlay.classList.toggle("img-blend", !state.blend);
  });

  el.sliceSlider.addEventListener("input", (e) => {
    state.curSlice = +e.target.value;
    el.sliceLabel.textContent = `${state.curSlice} / ${state.slices - 1}`;
    updateViewer();
  });

  // ------------------------------------------------------------- session
  function setupSession(data, label, sourceLabel) {
    state.sid = data.sid;
    state.slices = data.slices;
    state.shape = data.shape;
    state.curSlice = Math.floor(data.slices / 2);
    el.sliceSlider.max = data.slices - 1;
    el.sliceSlider.value = state.curSlice;
    el.viewerEmpty.classList.add("hidden");
    el.patientPill.textContent = label;
    el.patientPill.classList.add("pill-live");
    el.footerSess.textContent = `session ${data.sid}`;
    el.sessionMeta.hidden = false;
    el.metaShape.textContent = data.shape.join(" × ");
    el.metaSlices.textContent = data.slices;
    el.metaSource.textContent = sourceLabel;
    updateViewer();
  }

  function resetPickUI() {
    state.pickedFile = null;
    el.filePick.value = "";
    el.dzFile.hidden = true;
    el.btnOpenFile.hidden = true;
    el.btnOpenFile.disabled = true;
    el.btnClearFile.hidden = true;
  }

  function showFileButtons() {
    el.btnOpenFile.hidden = false;
    el.btnOpenFile.disabled = false;
    el.btnClearFile.hidden = false;
  }

  async function openFromFile() {
    const f = state.pickedFile;
    if (!f) return;
    setBusy(true, "Uploading…", 0.05);
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-json */ }
      if (!res.ok) {
        const msg = data && (data.detail || data.message) ? (data.detail || data.message) : `HTTP ${res.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      setupSession(data, f.name, "uploaded");
      const info = await api(`/api/session/${state.sid}`);
      applySessionState(info);
      resetPickUI();
      toast("Session opened (uploaded).", "ok");
      setStep(1);
    } catch (err) {
      toast(err.message, "err", 7000);
    } finally {
      setBusy(false, "Ready");
    }
  }

  async function openFromServerPath() {
    const dir = el.patientDir.value.trim();
    if (!dir) { toast("Isi path folder dulu.", "err"); return; }
    setBusy(true, "Opening session…");
    try {
      const data = await api("/api/session", { method: "POST", body: { patient_dir: dir } });
      setupSession(data, dir.split("/").slice(-2).join(" / "), "original");
      const info = await api(`/api/session/${state.sid}`);
      applySessionState(info);
      toast("Session opened.", "ok");
      setStep(1);
    } catch (err) {
      toast(err.message, "err", 7000);
    } finally {
      setBusy(false, "Ready");
    }
  }

  // -- file picker wiring -------------------------------------------
  function pickFile() { el.filePick.click(); }
  el.dropzone.addEventListener("click", pickFile);
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFile(); }
  });
  el.btnPick.addEventListener("click", pickFile);
  el.filePick.addEventListener("change", () => {
    const f = el.filePick.files && el.filePick.files[0];
    if (!f) return;
    state.pickedFile = f;
    el.dzFile.textContent = `${f.name}  (${(f.size / 1048576).toFixed(1)} MB)`;
    el.dzFile.hidden = false;
    showFileButtons();
  });
  el.btnClearFile.addEventListener("click", resetPickUI);
  el.btnOpenFile.addEventListener("click", openFromFile);

  // drag & drop
  ["dragenter", "dragover"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      el.dropzone.classList.add("is-over");
    }));
  ["dragleave", "drop"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      el.dropzone.classList.remove("is-over");
    }));
  el.dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) {
      const f = files[0];
      if (f.name.toLowerCase().endsWith(".npy") || f.type.startsWith("application") || f.type === "") {
        state.pickedFile = f;
        el.dzFile.textContent = `${f.name}  (${(f.size / 1048576).toFixed(1)} MB)`;
        el.dzFile.hidden = false;
        showFileButtons();
      } else {
        toast(`File "${f.name}" bukan .npy — pilih file hasil pipeline desktop.`, "err", 6000);
      }
    }
  });

  el.btnLoad.addEventListener("click", openFromServerPath);

  function applySessionState(info) {
    if (info.shape) {
      state.slices = info.slices;
      state.shape = info.shape;
      el.sliceSlider.max = state.slices - 1;
      el.metaShape.textContent = info.shape.join(" × ");
      el.metaSlices.textContent = info.slices;
      el.metaSource.textContent = info.volume_source;
    }
    if (info.missing_start != null || info.missing_end != null) {
      state.missing.start = info.missing_start;
      state.missing.end = info.missing_end;
      el.missStart.value = info.missing_start ?? "";
      el.missEnd.value = info.missing_end ?? "";
      el.metaRange.textContent = `[${info.missing_start}, ${info.missing_end}]`;
    }
    if (info.status && info.status !== "idle" && !state.busy) {
      setStatus(info.status === "done" ? "ok" : info.status === "error" ? "err" : "busy", info.message || info.status, info.progress);
    }
    renderResultsFromState(info);
  }

  // ------------------------------------------------------------- rotation
  function doRotate(side) {
    if (!state.sid) { toast("Open a session first.", "err"); return; }
    const angle = parseFloat(el.angle.value);
    const axis = el.axis.value;
    const sideLabel = side === "left" ? "Left" : "Right";
    setBusy(true, `Rotating (${sideLabel})…`);
    api(`/api/session/${state.sid}/rotate`, { method: "POST", body: { angle, axis, side } })
      .then((d) => {
        state.slices = d.shape[0];
        state.shape = d.shape;
        state.curSlice = Math.floor(state.slices / 2);
        el.sliceSlider.max = state.slices - 1;
        el.metaShape.textContent = d.shape.join(" × ");
        el.metaSlices.textContent = state.slices;
        el.metaSource.textContent = "rotated";
        el.metaRange.textContent = `[0, ${state.slices - 1}]`;
        el.missStart.value = 0;
        el.missEnd.value = state.slices - 1;
        state.missing.start = 0;
        state.missing.end = state.slices - 1;
        state.viewMode = "base";
        $$(".seg-btn").forEach((x) => x.classList.toggle("is-active", x.dataset.view === "base"));
        updateViewer();
        toast(`Rotated ${sideLabel} ${d.applied_angle}° around ${axis.toUpperCase()}.`, "ok");
        setStep(2);
      })
      .catch((e) => toast(e.message, "err", 6000))
      .finally(() => setBusy(false, "Ready"));
  }
  el.btnRotL.addEventListener("click", () => doRotate("left"));
  el.btnRotR.addEventListener("click", () => doRotate("right"));

  el.btnResetA.addEventListener("click", () => {
    setBusy(true, "Resetting…");
    api(`/api/session/${state.sid}/reset-rotation`, { method: "POST" })
      .then((d) => {
        state.slices = d.shape[0];
        state.shape = d.shape;
        el.sliceSlider.max = state.slices - 1;
        state.curSlice = Math.floor(state.slices / 2);
        updateViewer();
        toast("Volume reset to original.", "ok");
      })
      .catch((e) => toast(e.message, "err"))
      .finally(() => setBusy(false, "Ready"));
  });

  // ------------------------------------------------------------- range
  function validateRange() {
    const s = parseInt(el.missStart.value, 10);
    const e = parseInt(el.missEnd.value, 10);
    if (Number.isNaN(s) || Number.isNaN(e)) { toast("Provide both start and end slices.", "err"); return null; }
    if (s < 0 || e < 0 || s >= state.slices || e >= state.slices) {
      toast(`Slices must be within [0, ${state.slices - 1}].`, "err"); return null;
    }
    if (e < s) { toast("End must be ≥ start.", "err"); return null; }
    return { s, e };
  }

  el.btnSetRange.addEventListener("click", () => {
    const r = validateRange(); if (!r) return;
    setBusy(true, "Setting range…");
    api(`/api/session/${state.sid}/missing-range`, { method: "POST", body: { start: r.s, end: r.e } })
      .then(() => {
        state.missing = { start: r.s, end: r.e };
        el.metaRange.textContent = `[${r.s}, ${r.e}]`;
        toast(`Missing range set to [${r.s}, ${r.e}].`, "ok");
        setStep(3);
      })
      .catch((e) => toast(e.message, "err", 6000))
      .finally(() => setBusy(false, "Ready"));
  });

  el.btnAuto.addEventListener("click", () => {
    if (!state.sid) return;
    setBusy(true, "Auto-detecting missing slices (this can take ~1 min)…", 0.1);
    api(`/api/session/${state.sid}/missing-auto`, { method: "POST" })
      .then((d) => {
        state.missing = { start: d.start, end: d.end };
        el.missStart.value = d.start;
        el.missEnd.value = d.end;
        el.metaRange.textContent = `[${d.start}, ${d.end}]`;
        toast(`Auto-detected missing range [${d.start}, ${d.end}].`, "ok");
        setStep(3);
      })
      .catch((e) => toast(e.message, "err", 8000))
      .finally(() => setBusy(false, "Ready", null));
  });

  // ------------------------------------------------------------- run pipeline
  el.btnRun.addEventListener("click", () => {
    if (!state.sid) { toast("Open a session first.", "err"); return; }
    if (state.missing.start == null || state.missing.end == null) {
      const r = validateRange(); if (!r) return;
      el.missStart.value = r.s; el.missEnd.value = r.e;
    } else {
      el.missStart.value = state.missing.start;
      el.missEnd.value = state.missing.end;
    }
    setBusy(true, "Starting pipeline…", 0.05);
    el.btnRun.disabled = true;
    api(`/api/session/${state.sid}/run`, { method: "POST" })
      .then(() => pollStatus(0))
      .catch((e) => {
        toast(e.message, "err", 8000);
        el.btnRun.disabled = false;
        setBusy(false, "Ready");
      });
  });

  let pollTimer = null;
  function pollStatus(attempts = 0) {
    clearTimeout(pollTimer);
    if (attempts > 600) { // ~15 min safety net
      el.btnRun.disabled = false;
      setBusy(false, "Timed out waiting for backend — check server logs.", "err");
      return;
    }
    api(`/api/session/${state.sid}`)
      .then((info) => {
        applySessionState(info);
        const st = info.status;
        if (st === "running") {
          setStatus("busy", info.message || "Working…", info.progress);
          pollTimer = setTimeout(() => pollStatus(attempts + 1), 1200);
        } else if (st === "done") {
          el.btnRun.disabled = false;
          setStatus("ok", "Pipeline complete", 1);
          state.candidates = info.candidates || [];
          state.applied = info.applied;
          state.logs = info.logs || [];
          state.noImplantReason = info.no_implant_reason;
          $$(".step").forEach((s, i) => s.classList.toggle("is-done", i <= 3));
          state.viewMode = "overlay";
          $$(".seg-btn").forEach((x) => x.classList.toggle("is-active", x.dataset.view === "overlay"));
          state.curSlice = Math.max(0, (state.missing.start + state.missing.end) >> 1);
          updateViewer();
          renderResults();
          setStep(4);
          toast(state.applied ? `Applied ${state.applied.id}` : "Pipeline done — no candidate fit.", "ok");
        } else if (st === "error") {
          el.btnRun.disabled = false;
          setStatus("err", info.message || "failed");
          toast(info.message || "Pipeline failed", "err", 8000);
        } else if (st === "idle") {
          // could be a fresh session; nothing more to do
          el.btnRun.disabled = false;
          setStatus("", "Ready");
        }
      })
      .catch(() => {
        el.btnRun.disabled = false;
        setStatus("err", "lost connection to server");
      });
  }

  // ------------------------------------------------------------- results rendering
  function renderResultsFromState(info) {
    state.candidates = info.candidates || [];
    state.applied = info.applied || null;
    state.logs = info.logs || [];
    state.noImplantReason = info.no_implant_reason || null;
    renderResults();
  }

  function renderResults() {
    const hasCand = state.candidates.length > 0;
    el.noCand.hidden = hasCand;
    el.candHead.hidden = !hasCand;
    el.logDetails.hidden = state.logs.length === 0;

    if (!hasCand) {
      el.appliedBadge.hidden = true;
      el.candCount.textContent = "0";
      el.candList.innerHTML = "";
      const why = {
        sharp: "The alveolar bone is too sharp / narrow — no implant diameter fits safely.",
        short: "The alveolar bone is too short above the mandibular canal — no implant height fits safely.",
      }[state.noImplantReason] || "See the selection log below for per-implant reasons.";
      el.noCandWhy.textContent = why;
      el.kpiCand.textContent = "0";
      return;
    }

    el.kpiCand.textContent = String(state.candidates.length);
    el.candCount.textContent = String(state.candidates.length);

    if (state.applied) {
      el.appliedBadge.hidden = false;
      el.appliedId.textContent = state.applied.id;
      el.appliedD.textContent = fmtNum(state.applied.diameter_mm);
      el.appliedH.textContent = fmtNum(state.applied.height_mm);
      el.appliedMeta.textContent =
        `slices ${state.applied.start_slice}–${state.applied.end_slice} · vol ${fmtNum(state.applied.volume_mm3)} mm³`;
    }

    el.candList.innerHTML = "";
    state.candidates.forEach((c, i) => {
      const [rid, d, h, , , si, ei, vol] = c;
      const appliedIdx = state.applied ? state.candidates.findIndex(
        (x) => x[0] === state.applied.id && x[1] === state.applied.diameter_mm
          && x[2] === state.applied.height_mm) : -1;
      const isApplied = i === appliedIdx;
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cand" + (isApplied ? " is-applied" : "");
      btn.innerHTML = `
        <span class="cand-top">
          <span class="cand-rank">${i + 1}</span>
          <span class="cand-id">${escapeHtml(rid)}</span>
          <span class="cand-spec">Ø${fmtNum(d)} × ${fmtNum(h)} mm</span>
          ${isApplied ? '<span class="cand-tag">APPLIED</span>' : ""}
        </span>
        <span class="cand-vol">${fmtNum(vol)} mm³</span>
        <span class="cand-foot">slices ${si + state.missing.start}–${ei + state.missing.start}</span>`;
      btn.addEventListener("click", () => applyCandidate(i));
      li.appendChild(btn);
      el.candList.appendChild(li);
    });

    el.logPre.textContent = state.logs.join("\n");
  }

  function applyCandidate(i) {
    if (!state.sid) return;
    setBusy(true, "Applying implant…");
    api(`/api/session/${state.sid}/apply`, { method: "POST", body: { index: i } })
      .then((d) => {
        state.applied = d.applied;
        state.viewMode = "overlay";
        $$(".seg-btn").forEach((x) => x.classList.toggle("is-active", x.dataset.view === "overlay"));
        if (state.applied.start_slice != null) {
          state.curSlice = Math.max(0, Math.min(state.slices - 1,
            (state.applied.start_slice + state.applied.end_slice) >> 1));
        }
        updateViewer();
        renderResults();
        toast(`Applied ${d.applied.id}.`, "ok");
      })
      .catch((e) => toast(e.message, "err", 6000))
      .finally(() => setBusy(false, "Ready"));
  }

  // ------------------------------------------------------------- plants
  async function loadPlants() {
    try {
      const data = await api("/api/plants");
      state.plants = data.plants;
      renderPlants();
    } catch (e) { toast(e.message, "err"); }
  }

  function renderPlants() {
    el.kpiImplants.textContent = String(state.plants.length);
    el.plantList.innerHTML = "";
    const head = document.createElement("div");
    head.className = "plant-head plant-row";
    head.innerHTML = `<span>ID</span><span>Ø mm</span><span>H mm</span><span></span>`;
    el.plantList.appendChild(head);
    const sorted = [...state.plants].sort((a, b) =>
      a.id.localeCompare(b.id, undefined, { numeric: true }));
    sorted.forEach((p) => {
      const row = document.createElement("div");
      row.className = "plant-row";
      const id = document.createElement("span"); id.className = "plant-id"; id.textContent = p.id;
      const d = document.createElement("span"); d.className = "plant-num"; d.textContent = fmtNum(p.diameter_mm);
      const h = document.createElement("span"); h.className = "plant-num"; h.textContent = fmtNum(p.height_mm);
      const act = document.createElement("span"); act.style.display = "flex"; act.style.gap = "4px";
      const ed = document.createElement("button");
      ed.className = "btn btn-ghost btn-sm"; ed.textContent = "Edit";
      ed.addEventListener("click", () => editPlant(p.id));
      const del = document.createElement("button");
      del.className = "btn btn-danger btn-sm"; del.textContent = "×";
      del.addEventListener("click", () => {
        if (confirm(`Delete implant ${p.id}?`)) deletePlant(p.id);
      });
      act.appendChild(ed); act.appendChild(del);
      row.append(id, d, h, act);
      el.plantList.appendChild(row);
    });
  }

  function editPlant(id) {
    const p = state.plants.find((x) => x.id === id);
    if (!p) return;
    el.plantForm.id.value = p.id;
    el.plantForm.diameter_mm.value = p.diameter_mm;
    el.plantForm.height_mm.value = p.height_mm;
    el.plantForm._editId = p.id;
    el.plantForm.id.focus();
  }

  async function deletePlant(id) {
    try {
      await api(`/api/plants/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast(`Deleted ${id}.`, "ok");
      loadPlants();
    } catch (e) { toast(e.message, "err"); }
  }

  el.plantForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const fd = new FormData(el.plantForm);
    const body = {
      id: String(fd.get("id") || "").trim(),
      diameter_mm: parseFloat(fd.get("diameter_mm")),
      height_mm: parseFloat(fd.get("height_mm")),
    };
    if (!body.id || !(body.diameter_mm > 0) || !(body.height_mm > 0)) {
      toast("Fill in ID and positive Ø / H values.", "err");
      return;
    }
    const editId = el.plantForm._editId;
    if (editId) {
      api(`/api/plants/${encodeURIComponent(editId)}`, { method: "PUT", body })
        .then(() => { el.plantForm.reset(); el.plantForm._editId = null; toast("Updated.", "ok"); loadPlants(); })
        .catch((e) => toast(e.message, "err", 6000));
    } else {
      api("/api/plants", { method: "POST", body })
        .then(() => { el.plantForm.reset(); toast("Added.", "ok"); loadPlants(); })
        .catch((e) => toast(e.message, "err", 6000));
    }
  });

  // ------------------------------------------------------------- utils
  function fmtNum(n) {
    const v = Number(n);
    if (Number.isNaN(v)) return "—";
    return String(Math.round(v * 100) / 100);
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function setBusy(busy, label = "", progress = null) {
    state.busy = busy;
    if (busy) {
      setStatus("busy", label, progress);
      el.btnLoad.disabled = true;
      el.btnRun.disabled = false; // controlled by run/poll flow
    } else {
      setStatus(progress === "err" ? "err" : "ok", label);
      el.btnLoad.disabled = false;
    }
  }

  // ------------------------------------------------------------- init
  loadPlants();
  setStatus("", "Ready");
  setStep(0);
})();
