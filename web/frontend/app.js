/* Dental Implant Planner · Web — frontend logic */
(() => {
  "use strict";

  const V = window.NapariView;

  // ------------------------------------------------------------- state
  let pollTimer = null;
  const state = {
    sid: null,
    slices: 0,
    shape: null,
    curSlice: 0,
    missing: { start: null, end: null },
    candidates: [],
    applied: null,
    logs: [],
    noImplantReason: null,
    plants: [],
    busy: false,
    pickedFile: null,        // File object from the picker, not yet uploaded
    selectedLayer: "base",
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
    dimsBand:     $("#dimsBand"),
    toast:        $("#toast"),
    footerSess:   $("#footerSess"),

    // napari dock
    layerList:    $("#layerList"),
    dockCount:    $("#dockCount"),
    lc:           $("#layerControls"),
    lcName:       $("#lcName"),
    lcOpacity:    $("#lcOpacity"),
    lcOpacityOut: $("#lcOpacityOut"),
    lcClimLo:     $("#lcClimLo"),
    lcClimHi:     $("#lcClimHi"),
    lcClimOut:    $("#lcClimOut"),
    lcGamma:      $("#lcGamma"),
    lcGammaOut:   $("#lcGammaOut"),
    lcColormap:   $("#lcColormap"),
    lcColormapBar:$("#lcColormapBar"),
    lcBlending:   $("#lcBlending"),
    lcRendering:  $("#lcRendering"),
    lcRenderingRow: $("#lcRenderingRow"),
    lcIso:        $("#lcIso"),
    lcIsoOut:     $("#lcIsoOut"),
    lcIsoRow:     $("#lcIsoRow"),
    lcAtten:      $("#lcAtten"),
    lcAttenOut:   $("#lcAttenOut"),
    lcAttenRow:   $("#lcAttenRow"),
    lc2dNote:     $("#lc2dNote"),
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
      const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      err.status = res.status;
      throw err;
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
  function setStep(n) {
    $$(".step").forEach((s, i) => {
      s.classList.toggle("is-active", i === n);
      s.classList.toggle("is-done", i < n);
    });
    $$(".step-panel").forEach((p) => p.classList.toggle("is-active", +p.dataset.step === n));
  }
  $$(".step").forEach((s) => s.addEventListener("click", () => setStep(+s.dataset.step)));

  // ============================================================== viewer
  const EYE_ON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12Z"/><circle cx="12" cy="12" r="2.8"/></svg>`;
  const EYE_OFF = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 4l16 16M9.9 5.7A10.6 10.6 0 0 1 12 5.5c6.4 0 10 6.5 10 6.5a18 18 0 0 1-3.3 4M6.4 7.9A17.6 17.6 0 0 0 2 12s3.6 6.5 10 6.5a10.9 10.9 0 0 0 3.9-.7"/></svg>`;

  async function bootViewer() {
    try {
      V.quality = parseInt($("#volQuality").value, 10) || V.quality;
      await V.boot($("#vol3d"));
      V.onChange(renderDock);
      setNdisplay(V.ndisplay);
    } catch (e) {
      toast(e.message, "err", 9000);
    }
  }

  function renderDock() {
    const layers = V.layers || [];
    el.dockCount.textContent = String(layers.filter((l) => l.available).length);
    el.layerList.innerHTML = "";

    if (!layers.length) {
      const li = document.createElement("li");
      li.className = "layer-empty";
      li.textContent = "No volume loaded yet.";
      el.layerList.appendChild(li);
      el.lc.hidden = true;
      return;
    }

    // napari draws the topmost layer first in its list
    [...layers].reverse().forEach((L) => {
      const li = document.createElement("li");
      li.className = "layer-row"
        + (L.kind === state.selectedLayer ? " is-selected" : "")
        + (L.available ? "" : " is-missing");

      const eye = document.createElement("button");
      eye.type = "button";
      eye.className = "layer-eye";
      eye.innerHTML = L.visible ? EYE_ON : EYE_OFF;
      eye.title = L.visible ? "Hide layer" : "Show layer";
      eye.setAttribute("aria-pressed", String(L.visible));
      eye.disabled = !L.available;
      eye.addEventListener("click", (ev) => {
        ev.stopPropagation();
        V.set(L.kind, { visible: !L.visible });
      });

      const thumb = document.createElement("span");
      thumb.className = "layer-thumb";
      thumb.style.background = L.synthetic
        ? `linear-gradient(90deg, ${V.rangeColors.start} 0 50%, ${V.rangeColors.end} 50% 100%)`
        : L.rendering === "screw"
          ? "linear-gradient(135deg,#f2f6fa 0%,#aeb9c4 45%,#eef3f8 60%,#96a3b0 100%)"
          : V.cmapCss(L.colormap);

      const name = document.createElement("span");
      name.className = "layer-name";
      name.textContent = L.name;

      const tag = document.createElement("span");
      tag.className = "layer-tag";
      tag.textContent = L.synthetic ? "range" : (L.available ? L.rendering : "pending");

      li.append(eye, thumb, name, tag);
      li.addEventListener("click", () => {
        state.selectedLayer = L.kind;
        renderDock();
      });
      el.layerList.appendChild(li);
    });

    syncLayerControls();
  }

  function selectedLayer() {
    return (V.layers || []).find((l) => l.kind === state.selectedLayer) || null;
  }

  let syncing = false;
  function syncLayerControls() {
    const L = selectedLayer();
    if (!L || !L.available) { el.lc.hidden = true; return; }
    syncing = true;
    el.lc.hidden = false;
    el.lcName.textContent = L.name;

    el.lcOpacity.value = Math.round(L.opacity * 100);
    el.lcOpacityOut.value = L.opacity.toFixed(2);

    // the missing-range marker is geometry, not a volume — opacity is all it has
    if (L.synthetic) {
      for (const row of [el.lcClimLo.closest(".lc-row"), el.lcGamma.closest(".lc-row"),
                         el.lcColormap.closest(".lc-row"), el.lcBlending.closest(".lc-row"),
                         el.lcRenderingRow, el.lcIsoRow, el.lcAttenRow]) {
        if (row) row.hidden = true;
      }
      el.lc2dNote.hidden = false;
      el.lc2dNote.textContent =
        "Red planes mark the first and last slice of the missing-teeth range.";
      syncing = false;
      return;
    }
    for (const row of [el.lcClimLo.closest(".lc-row"), el.lcGamma.closest(".lc-row"),
                       el.lcColormap.closest(".lc-row"), el.lcBlending.closest(".lc-row")]) {
      if (row) row.hidden = false;
    }
    el.lc2dNote.textContent =
      "Rendering, iso threshold and attenuation apply to the 3D view only.";

    const [lo, hi] = L.range;
    const toSlider = (v) => Math.round(((v - lo) / Math.max(hi - lo, 1e-9)) * 1000);
    el.lcClimLo.value = toSlider(L.climLo);
    el.lcClimHi.value = toSlider(L.climHi);
    el.lcClimOut.value = `${fmtNum(L.climLo)} – ${fmtNum(L.climHi)}`;

    el.lcGamma.value = Math.round(L.gamma * 100);
    el.lcGammaOut.value = L.gamma.toFixed(2);

    fillSelect(el.lcColormap, V.cmaps, L.colormap);
    el.lcColormapBar.style.background = V.cmapCss(L.colormap);
    fillSelect(el.lcBlending, V.blendings, L.blending);
    fillSelect(el.lcRendering,
               L.kind === "implant" ? V.implantModes : V.renderModes,
               L.rendering);

    el.lcIso.value = Math.round(L.iso * 100);
    el.lcIsoOut.value = L.iso.toFixed(2);
    el.lcAtten.value = Math.round(L.attenuation * 200);
    el.lcAttenOut.value = L.attenuation.toFixed(2);

    const is3d = V.ndisplay === 3;
    el.lcRenderingRow.hidden = false;
    el.lcIsoRow.hidden = !(is3d && L.rendering === "iso");
    if (L.rendering === "screw") {
      el.lc2dNote.hidden = false;
      el.lc2dNote.textContent =
        `Titanium screw model at the planned position — Ø${fmtNum(state.applied?.diameter_mm)}`
        + ` × ${fmtNum(state.applied?.height_mm)} mm. Switch rendering to see the raw mask.`;
    }
    el.lcAttenRow.hidden = !(is3d && L.rendering === "attenuated_mip");
    el.lcRenderingRow.classList.toggle("is-disabled", !is3d);
    el.lcRendering.disabled = !is3d;
    el.lc2dNote.hidden = is3d;
    syncing = false;
  }

  function fillSelect(sel, options, value) {
    if (sel.dataset.filled !== options.join("|")) {
      sel.innerHTML = "";
      options.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = o;
        opt.textContent = o;
        sel.appendChild(opt);
      });
      sel.dataset.filled = options.join("|");
    }
    sel.value = value;
  }

  function patch(p) {
    if (syncing) return;
    const L = selectedLayer();
    if (L) V.set(L.kind, p);
  }

  el.lcOpacity.addEventListener("input", (e) => patch({ opacity: +e.target.value / 100 }));
  el.lcGamma.addEventListener("input", (e) => patch({ gamma: +e.target.value / 100 }));
  el.lcColormap.addEventListener("change", (e) => patch({ colormap: e.target.value }));
  el.lcBlending.addEventListener("change", (e) => patch({ blending: e.target.value }));
  el.lcRendering.addEventListener("change", (e) => patch({ rendering: e.target.value }));
  el.lcIso.addEventListener("input", (e) => patch({ iso: +e.target.value / 100 }));
  el.lcAtten.addEventListener("input", (e) => patch({ attenuation: +e.target.value / 200 }));

  function climFromSliders() {
    const L = selectedLayer();
    if (!L) return;
    const [lo, hi] = L.range;
    const span = Math.max(hi - lo, 1e-9);
    let a = lo + (+el.lcClimLo.value / 1000) * span;
    let b = lo + (+el.lcClimHi.value / 1000) * span;
    if (a > b) { const t = a; a = b; b = t; }
    if (b - a < span * 1e-3) b = a + span * 1e-3;
    patch({ climLo: a, climHi: b });
  }
  el.lcClimLo.addEventListener("input", climFromSliders);
  el.lcClimHi.addEventListener("input", climFromSliders);

  // -- ndisplay / camera / capture -----------------------------------
  function setNdisplay(n) {
    V.setNdisplay(n);
    $("#btnNd2").classList.toggle("is-active", n === 2);
    $("#btnNd3").classList.toggle("is-active", n === 3);
    $("#dimsBar").classList.toggle("is-idle", n === 3);
    el.sliceSlider.disabled = n === 3;
    syncLayerControls();
  }
  $("#btnNd2").addEventListener("click", () => setNdisplay(2));
  $("#btnNd3").addEventListener("click", () => setNdisplay(3));
  $("#btnResetView").addEventListener("click", () => V.resetView());
  $("#btnShot").addEventListener("click", () => {
    const url = V.screenshot();
    if (!url) { toast("Nothing to capture yet.", "err"); return; }
    const a = document.createElement("a");
    a.href = url;
    a.download = `implant-view-${state.sid || "session"}-z${state.curSlice}.png`;
    a.click();
    toast("Screenshot saved.", "ok");
  });
  $("#volQuality").addEventListener("change", async (e) => {
    if (!state.sid) return;
    setBusy(true, "Reloading volume…");
    try {
      await V.setQuality(parseInt(e.target.value, 10));
    } catch (err) {
      if (isSessionGone(err)) sessionLost("The server no longer has this session");
      else toast(err.message, "err", 6000);
    } finally { setBusy(false, "Ready"); }
  });
  $("#btnToggleDock").addEventListener("click", () => {
    document.getElementById("appMain").classList.toggle("dock-hidden");
  });

  el.sliceSlider.addEventListener("input", (e) => {
    state.curSlice = +e.target.value;
    el.sliceLabel.textContent = `${state.curSlice} / ${Math.max(0, state.slices - 1)}`;
    V.setSlice(state.curSlice, state.slices);
  });

  /** The backend forgot this session (restart with no volume on disk). */
  function sessionLost(msg) {
    clearTimeout(pollTimer);
    state.sid = null;
    state.slices = 0;
    state.candidates = [];
    state.applied = null;
    state.logs = [];
    el.viewerEmpty.classList.remove("hidden");
    el.legend.hidden = true;
    el.sessionMeta.hidden = true;
    el.patientPill.textContent = "no session";
    el.patientPill.classList.remove("pill-live");
    el.footerSess.textContent = "no session";
    el.btnRun.disabled = false;
    setMissingRange(null, null);
    renderResults();
    setStatus("err", "Session ended");
    toast(msg + " — open the volume again to continue.", "err", 8000);
    setStep(0);
  }

  function isSessionGone(e) {
    return e && (e.status === 404 || /unknown session/i.test(e.message || ""));
  }

  /** Push the applied implant to the viewer so it can draw the 3D model. */
  function syncImplantModel() {
    V.setImplant(state.applied && state.shape ? state.applied : null, state.shape);
  }

  /** Reload every layer from the backend (after rotate / run / apply). */
  async function refreshLayers() {
    if (!state.sid) return;
    try {
      await V.reload();
    } catch (e) {
      if (isSessionGone(e)) sessionLost("The server no longer has this session");
      else toast("Viewer reload failed: " + e.message, "err", 6000);
    }
  }

  /** Record the missing range and show its boundary markers + slider band. */
  function setMissingRange(start, end) {
    state.missing.start = start;
    state.missing.end = end;
    const has = start != null && end != null;
    el.metaRange.textContent = has ? `[${start}, ${end}]` : "—";
    if (has) {
      el.missStart.value = start;
      el.missEnd.value = end;
    }
    V.setMissingRange(has ? start : null, has ? end : null);
    const band = el.dimsBand;
    if (!band) return;
    const last = Math.max(1, state.slices - 1);
    band.hidden = !has;
    if (has) {
      band.style.left = `${(start / last) * 100}%`;
      band.style.width = `${((end - start) / last) * 100}%`;
    }
  }

  function syncSliceUi() {
    el.sliceSlider.max = Math.max(0, state.slices - 1);
    el.sliceSlider.value = state.curSlice;
    el.sliceLabel.textContent = `${state.curSlice} / ${Math.max(0, state.slices - 1)}`;
    V.setSlice(state.curSlice, state.slices);
  }

  // ------------------------------------------------------------- session
  async function setupSession(data, label, sourceLabel) {
    state.sid = data.sid;
    state.slices = data.slices;
    state.shape = data.shape;
    state.curSlice = Math.floor(data.slices / 2);
    el.viewerEmpty.classList.add("hidden");
    el.legend.hidden = false;
    el.patientPill.textContent = label;
    el.patientPill.classList.add("pill-live");
    el.footerSess.textContent = `session ${data.sid}`;
    el.sessionMeta.hidden = false;
    el.metaShape.textContent = data.shape.join(" × ");
    el.metaSlices.textContent = data.slices;
    el.metaSource.textContent = sourceLabel;
    await V.attach(state.sid, $("#vol3d"));
    syncSliceUi();
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
      await setupSession(data, f.name, "uploaded");
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
      await setupSession(data, dir.split("/").slice(-2).join(" / "), "original");
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
      el.metaShape.textContent = info.shape.join(" × ");
      el.metaSlices.textContent = info.slices;
      el.metaSource.textContent = info.volume_source;
      state.curSlice = Math.min(state.curSlice, Math.max(0, info.slices - 1));
      syncSliceUi();
    }
    if (info.missing_start != null && info.missing_end != null) {
      setMissingRange(info.missing_start, info.missing_end);
    }
    if (info.status && info.status !== "idle" && !state.busy) {
      setStatus(info.status === "done" ? "ok" : info.status === "error" ? "err" : "busy",
                info.message || info.status, info.progress);
    }
    renderResultsFromState(info);
    syncImplantModel();
  }

  // ------------------------------------------------------------- rotation
  function doRotate(side) {
    if (!state.sid) { toast("Open a session first.", "err"); return; }
    const angle = parseFloat(el.angle.value);
    const axis = el.axis.value;
    const sideLabel = side === "left" ? "Left" : "Right";
    setBusy(true, `Rotating (${sideLabel})…`);
    api(`/api/session/${state.sid}/rotate`, { method: "POST", body: { angle, axis, side } })
      .then(async (d) => {
        state.slices = d.shape[0];
        state.shape = d.shape;
        state.curSlice = Math.floor(state.slices / 2);
        el.metaShape.textContent = d.shape.join(" × ");
        el.metaSlices.textContent = state.slices;
        el.metaSource.textContent = "rotated";
        state.applied = null;
        await refreshLayers();
        setMissingRange(null, null);
        syncImplantModel();
        el.missStart.value = 0;
        el.missEnd.value = state.slices - 1;
        syncSliceUi();
        toast(`Rotated ${sideLabel} ${d.applied_angle}° around ${axis.toUpperCase()}.`, "ok");
        setStep(2);
      })
      .catch((e) => toast(e.message, "err", 6000))
      .finally(() => setBusy(false, "Ready"));
  }
  el.btnRotL.addEventListener("click", () => doRotate("left"));
  el.btnRotR.addEventListener("click", () => doRotate("right"));

  el.btnResetA.addEventListener("click", () => {
    if (!state.sid) { toast("Open a session first.", "err"); return; }
    setBusy(true, "Resetting…");
    api(`/api/session/${state.sid}/reset-rotation`, { method: "POST" })
      .then(async (d) => {
        state.slices = d.shape[0];
        state.shape = d.shape;
        state.curSlice = Math.floor(state.slices / 2);
        el.metaShape.textContent = d.shape.join(" × ");
        el.metaSlices.textContent = state.slices;
        el.metaSource.textContent = "original";
        await refreshLayers();
        setMissingRange(null, null);
        syncSliceUi();
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
        setMissingRange(r.s, r.e);
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
        setMissingRange(d.start, d.end);
        state.curSlice = Math.max(0, (d.start + d.end) >> 1);
        syncSliceUi();
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

  function pollStatus(attempts = 0) {
    clearTimeout(pollTimer);
    if (attempts > 600) { // ~15 min safety net
      el.btnRun.disabled = false;
      setStatus("err", "Timed out waiting for backend — check server logs.");
      return;
    }
    api(`/api/session/${state.sid}`)
      .then(async (info) => {
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
          state.curSlice = Math.max(0, (state.missing.start + state.missing.end) >> 1);
          await refreshLayers();          // AB / MC / implant layers now exist
          syncSliceUi();
          syncImplantModel();
          renderResults();
          setStep(4);
          toast(state.applied ? `Applied ${state.applied.id}` : "Pipeline done — no candidate fit.", "ok");
        } else if (st === "error") {
          el.btnRun.disabled = false;
          setStatus("err", info.message || "failed");
          toast(info.message || "Pipeline failed", "err", 8000);
        } else if (st === "idle") {
          el.btnRun.disabled = false;
          setStatus("", "Ready");
        }
      })
      .catch((e) => {
        el.btnRun.disabled = false;
        if (isSessionGone(e)) sessionLost("The server no longer has this session");
        else setStatus("err", "lost connection to server");
      });
  }

  // ------------------------------------------------------------- results
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
      el.logPre.textContent = state.logs.join("\n");
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
    const appliedIdx = state.applied ? state.candidates.findIndex(
      (x) => x[0] === state.applied.id && x[1] === state.applied.diameter_mm
        && x[2] === state.applied.height_mm) : -1;
    state.candidates.forEach((c, i) => {
      const [rid, d, h, , , si, ei, vol] = c;
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
      .then(async (d) => {
        state.applied = d.applied;
        if (state.applied.start_slice != null) {
          state.curSlice = Math.max(0, Math.min(state.slices - 1,
            (state.applied.start_slice + state.applied.end_slice) >> 1));
        }
        await refreshLayers();
        syncSliceUi();
        syncImplantModel();
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
      const act = document.createElement("span"); act.className = "plant-actions";
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
    } else {
      setStatus("ok", label);
      el.btnLoad.disabled = false;
    }
  }

  // ------------------------------------------------------------- keyboard
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    if (e.key === "2") setNdisplay(2);
    else if (e.key === "3") setNdisplay(3);
    else if (e.key === "r") V.resetView();
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      if (!state.sid) return;
      const step = e.key === "ArrowLeft" ? -1 : 1;
      if (V.ndisplay === 3) setNdisplay(2);
      state.curSlice = Math.max(0, Math.min(state.slices - 1, state.curSlice + step));
      syncSliceUi();
      e.preventDefault();
    }
  });

  // ------------------------------------------------------------- init
  bootViewer();
  setNdisplay(3);
  loadPlants();
  renderDock();
  setStatus("", "Ready");
  setStep(0);
})();
