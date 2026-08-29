/* vol3d.js — napari-style layered CBCT viewer for the browser.
 *
 * Mirrors the desktop app's napari canvas (main46.py):
 *   · a layer stack — CBCT base + final_AB / final_implant / final_MC
 *   · per-layer colormap, rendering mode, blending, opacity, contrast
 *     limits, gamma and (for iso) an iso threshold
 *   · an ndisplay toggle: 2D single-slice view ⇄ 3D free-orbit volume
 *   · orientation gizmo, reset view, screenshot
 *
 * Everything is drawn in one WebGL2 canvas: each layer owns a box mesh
 * (3D ray-marching) and a plane mesh (2D slice), both reading the same
 * Data3DTexture through the same shader.
 *
 * Exposes window.NapariView.
 */
(() => {
  "use strict";

  // ------------------------------------------------------------ colormaps
  // Anchor stops sampled from the matplotlib / napari colormaps of the same
  // name, then linearly interpolated into a 256×1 LUT texture.
  const CMAPS = {
    gray:    [[0, 0, 0], [255, 255, 255]],
    magma:   [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
              [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]],
    inferno: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
              [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]],
    plasma:  [[13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150], [203, 70, 121],
              [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33]],
    viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
              [31, 158, 137], [53, 183, 121], [109, 205, 89], [253, 231, 37]],
    turbo:   [[48, 18, 59], [70, 107, 227], [54, 183, 229], [63, 233, 150], [150, 255, 80],
              [220, 232, 55], [254, 173, 42], [238, 84, 20], [122, 4, 3]],
    bone:    [[0, 0, 0], [55, 55, 76], [110, 121, 138], [169, 200, 200], [255, 255, 255]],
    hot:     [[0, 0, 0], [140, 0, 0], [255, 30, 0], [255, 160, 0], [255, 255, 90], [255, 255, 255]],
    red:     [[0, 0, 0], [255, 0, 0]],
    green:   [[0, 0, 0], [0, 255, 0]],
    blue:    [[0, 0, 0], [0, 0, 255]],
    cyan:    [[0, 0, 0], [0, 255, 255]],
    yellow:  [[0, 0, 0], [255, 255, 0]],
    magenta: [[0, 0, 0], [255, 0, 255]],
  };
  const CMAP_NAMES = Object.keys(CMAPS);
  const RENDER_MODES = ["mip", "attenuated_mip", "translucent", "additive", "iso"];
  const BLENDINGS = ["translucent", "additive", "opaque"];

  /** Colour at position t (0..1) of a colormap, as [r,g,b] 0..255. */
  function cmapSample(name, t) {
    const stops = CMAPS[name] || CMAPS.gray;
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  function cmapCss(name) {
    const stops = [];
    for (let i = 0; i <= 8; i++) {
      const [r, g, b] = cmapSample(name, i / 8);
      stops.push(`rgb(${r | 0},${g | 0},${b | 0}) ${(i / 8 * 100).toFixed(0)}%`);
    }
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }

  function makeLut(THREE, name) {
    const data = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = cmapSample(name, i / 255);
      data[i * 4 + 0] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255;
    }
    const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  // -------------------------------------------------------------- shaders
  const COMMON = /* glsl */`
    precision highp float;
    precision highp sampler3D;
    uniform sampler3D volTex;
    uniform sampler2D lutTex;
    uniform vec3  uCamLocal;     // camera in object space (box is [-.5,.5]^3)
    uniform vec2  uClim;         // contrast limits, normalised 0..1
    uniform float uGamma;
    uniform float uOpacity;
    uniform float uIso;
    uniform float uAtten;
    uniform int   uMode;         // 0 mip · 1 attenuated_mip · 2 translucent · 3 additive · 4 iso
    uniform int   uSteps;
    uniform float uSlice;        // 2D: normalised Z of the displayed plane
    uniform vec3  uTexel;        // 1 / texture dims, for iso gradients

    /* raw sample -> contrast-limited, gamma-corrected intensity */
    float intensity(vec3 uvw) {
      float raw = texture(volTex, uvw).r;
      float v = (raw - uClim.x) / max(uClim.y - uClim.x, 1e-5);
      v = clamp(v, 0.0, 1.0);
      return pow(v, 1.0 / max(uGamma, 1e-3));
    }

    vec3 shade(float v) { return texture(lutTex, vec2(v, 0.5)).rgb; }
  `;

  const VOL_VERT = /* glsl */`
    out vec3 vLocal;
    void main() {
      vLocal = position;                       // BoxGeometry(1,1,1) -> [-.5,.5]
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const VOL_FRAG = COMMON + /* glsl */`
    in vec3 vLocal;
    out vec4 fragColor;

    vec3 gradient(vec3 uvw) {
      return vec3(
        intensity(uvw + vec3(uTexel.x, 0.0, 0.0)) - intensity(uvw - vec3(uTexel.x, 0.0, 0.0)),
        intensity(uvw + vec3(0.0, uTexel.y, 0.0)) - intensity(uvw - vec3(0.0, uTexel.y, 0.0)),
        intensity(uvw + vec3(0.0, 0.0, uTexel.z)) - intensity(uvw - vec3(0.0, 0.0, uTexel.z)));
    }

    void main() {
      vec3 ro = uCamLocal;
      vec3 rd = normalize(vLocal - ro);
      vec3 inv = 1.0 / rd;
      vec3 ta = (-0.5 - ro) * inv;
      vec3 tb = ( 0.5 - ro) * inv;
      vec3 tmin = min(ta, tb), tmax = max(ta, tb);
      float tNear = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));
      float tFar  = min(tmax.x, min(tmax.y, tmax.z));
      if (tNear >= tFar) discard;

      float span = tFar - tNear;
      float dt = span / float(uSteps);

      float acc = 0.0;             // mip / additive accumulator
      vec3  col = vec3(0.0);       // translucent / iso colour
      float alpha = 0.0;
      bool  hit = false;

      for (int i = 0; i < 512; i++) {
        if (i >= uSteps) break;
        float t = tNear + (float(i) + 0.5) * dt;
        vec3 p = ro + rd * t;
        vec3 uvw = vec3(p.x + 0.5, 0.5 - p.y, p.z + 0.5);
        float v = intensity(uvw);

        if (uMode == 0) {                       // mip
          acc = max(acc, v);
        } else if (uMode == 1) {                // attenuated_mip
          acc = max(acc, v * exp(-uAtten * float(i) * dt * 12.0));
        } else if (uMode == 2) {                // translucent
          float a = v * uOpacity * dt * 40.0;
          if (a > 0.001) {
            col += (1.0 - alpha) * a * shade(v);
            alpha += (1.0 - alpha) * a;
            if (alpha > 0.995) break;
          }
        } else if (uMode == 3) {                // additive
          acc += v * dt * 30.0;
          if (acc > 1.0) { acc = 1.0; break; }
        } else {                                // iso
          if (v >= uIso) {
            vec3 n = normalize(gradient(uvw) + vec3(1e-5));
            float lambert = 0.42 + 0.58 * abs(dot(n, -rd));
            col = shade(clamp(v, 0.0, 1.0)) * lambert;
            alpha = uOpacity;
            hit = true;
            break;
          }
        }
      }

      if (uMode == 0 || uMode == 1 || uMode == 3) {
        if (acc <= 0.004) discard;
        fragColor = vec4(shade(acc), acc * uOpacity);
      } else if (uMode == 2) {
        if (alpha <= 0.004) discard;
        fragColor = vec4(col / max(alpha, 1e-4), alpha);
      } else {
        if (!hit) discard;
        fragColor = vec4(col, alpha);
      }
    }`;

  const SLICE_VERT = /* glsl */`
    out vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const SLICE_FRAG = COMMON + /* glsl */`
    in vec2 vUv;
    out vec4 fragColor;
    void main() {
      float v = intensity(vec3(vUv.x, 1.0 - vUv.y, uSlice));
      if (v <= 0.002) discard;                 // keep overlays transparent
      fragColor = vec4(shade(v), v * uOpacity);
    }`;

  // ---------------------------------------------------------------- state
  const API = {
    ready: false,
    sid: null,
    ndisplay: 3,
    slice: 0,
    sliceCount: 1,            // session Z — the texture may be downsampled
    range: { start: null, end: null },   // missing-teeth slice range
    rangeColors: { start: "#2ee6ff", end: "#ff5ce1" },
    implant: null,                      // applied implant, for the 3D model
    implantModes: [...RENDER_MODES, "screw"],
    dims: [1, 1, 1],          // texture w,h,d of the base layer
    layers: [],               // [{kind, name, available, loaded, ...props}]
    cmaps: CMAP_NAMES,
    renderModes: RENDER_MODES,
    blendings: BLENDINGS,
    cmapCss,
    quality: 256,
    _onChange: null,
  };
  window.NapariView = API;

  const G = {                 // three.js scene graph
    THREE: null, renderer: null, scene: null, gizmoScene: null, gizmoCam: null,
    cam3: null, cam2: null, controls3: null, controls2: null,
    host: null, luts: {}, group3: null, group2: null, box: null,
    token: 0, needsRender: true,
  };

  const emit = () => { if (API._onChange) API._onChange(); };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ---------------------------------------------------------------- boot
  async function boot(hostEl) {
    if (G.bootPromise) return G.bootPromise;
    G.bootPromise = bootOnce(hostEl);
    return G.bootPromise;
  }

  async function bootOnce(hostEl) {
    const THREE = await import("three");
    const { OrbitControls } = await import("./vendor/OrbitControls.js");
    G.THREE = THREE;
    G.host = hostEl;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    if (!renderer.capabilities.isWebGL2) {
      throw new Error("WebGL2 is required for volume rendering — please use a current browser.");
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.autoClear = false;
    renderer.domElement.className = "napari-gl";
    hostEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    // Volume layers are ShaderMaterial and range markers MeshBasicMaterial, so
    // these only light the implant model.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x1b2733, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.35);
    key.position.set(1.2, 1.6, 1.0);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fd8ff, 0.55);
    rim.position.set(-1.0, -0.4, -0.9);
    scene.add(rim);

    const group3 = new THREE.Group();
    const group2 = new THREE.Group();
    const rangeGroup = new THREE.Group();     // missing-range markers (3D)
    const rangeGroup2 = new THREE.Group();    // missing-range markers (2D)
    const screwGroup = new THREE.Group();     // the implant model (both modes)
    group3.add(rangeGroup);
    group2.add(rangeGroup2);
    scene.add(group3, group2, screwGroup);

    const cam3 = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    cam3.position.set(1.5, 1.05, 1.75);
    const cam2 = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    cam2.position.set(0, 0, 2);

    const c3 = new OrbitControls(cam3, renderer.domElement);
    c3.enableDamping = true; c3.dampingFactor = 0.09; c3.rotateSpeed = 0.85;
    c3.minDistance = 0.55; c3.maxDistance = 9;
    const c2 = new OrbitControls(cam2, renderer.domElement);
    c2.enableRotate = false; c2.enableDamping = false;
    c2.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    c2.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
    c2.enabled = false;
    [c3, c2].forEach((c) => c.addEventListener("change", () => { G.needsRender = true; }));

    // dashed bounding box, like napari's 3D volume outline
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x4d6a94, transparent: true, opacity: 0.55 }));
    group3.add(box);

    // orientation gizmo (bottom-left corner overlay)
    const gizmoScene = new THREE.Scene();
    const gizmoCam = new THREE.PerspectiveCamera(45, 1, 0.1, 10);
    gizmoCam.position.set(0, 0, 2.6);
    const AX = [[0xff5a6e, 1, 0, 0], [0x53d38a, 0, 1, 0], [0x4aa8ff, 0, 0, 1]];
    AX.forEach(([c, x, y, z]) => {
      const g = new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(0, 0, 0), new THREE.Vector3(x, y, z)]);
      gizmoScene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: c })));
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 12),
        new THREE.MeshBasicMaterial({ color: c }));
      tip.position.set(x, y, z);
      gizmoScene.add(tip);
    });

    Object.assign(G, { renderer, scene, group3, group2, rangeGroup, rangeGroup2,
                       screwGroup, cam3, cam2, controls3: c3, controls2: c2,
                       box, gizmoScene, gizmoCam });
    CMAP_NAMES.forEach((n) => { G.luts[n] = makeLut(THREE, n); });

    new ResizeObserver(resize).observe(hostEl);
    resize();
    API.ready = true;

    (function loop() {
      requestAnimationFrame(loop);
      const controls = API.ndisplay === 3 ? G.controls3 : G.controls2;
      if (controls.enableDamping) { controls.update(); G.needsRender = true; }
      if (!G.needsRender) return;
      G.needsRender = controls.enableDamping;
      render();
    })();
  }

  function resize() {
    if (!G.renderer || !G.host) return;
    const w = Math.max(1, G.host.clientWidth), h = Math.max(1, G.host.clientHeight);
    G.renderer.setSize(w, h, false);
    G.cam3.aspect = w / h;
    G.cam3.updateProjectionMatrix();
    fitOrtho();
    G.needsRender = true;
  }

  function fitOrtho() {
    const w = Math.max(1, G.host.clientWidth), h = Math.max(1, G.host.clientHeight);
    const a = w / h;
    const half = 0.62;
    G.cam2.left = -half * a; G.cam2.right = half * a;
    G.cam2.top = half; G.cam2.bottom = -half;
    G.cam2.updateProjectionMatrix();
  }

  function render() {
    const THREE = G.THREE;
    const is3d = API.ndisplay === 3;
    const cam = is3d ? G.cam3 : G.cam2;
    G.group3.visible = is3d;
    G.group2.visible = !is3d;

    // feed each visible 3D layer the camera position in its own object space
    if (is3d) {
      for (const L of API.layers) {
        if (!L._mesh3) continue;
        const p = L._mesh3.worldToLocal(cam.position.clone());
        L._mat3.uniforms.uCamLocal.value.copy(p);
      }
    }

    G.renderer.clear();
    G.renderer.setViewport(0, 0, G.host.clientWidth, G.host.clientHeight);
    G.renderer.render(G.scene, cam);

    if (is3d) {                                    // orientation gizmo overlay
      const s = 74;
      G.gizmoScene.quaternion.copy(G.cam3.quaternion).invert();
      G.renderer.clearDepth();
      G.renderer.setViewport(12, 12, s, s);
      G.renderer.setScissor(12, 12, s, s);
      G.renderer.setScissorTest(true);
      G.renderer.render(G.gizmoScene, G.gizmoCam);
      G.renderer.setScissorTest(false);
      G.renderer.setViewport(0, 0, G.host.clientWidth, G.host.clientHeight);
    }
  }

  // ------------------------------------------------------------- textures
  function upload(bytes, w, h, d) {
    const THREE = G.THREE;
    const tex = new THREE.Data3DTexture(bytes, w, h, d);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    return tex;
  }

  function blendingFor(name) {
    const THREE = G.THREE;
    if (name === "additive") return THREE.AdditiveBlending;
    if (name === "opaque") return THREE.NoBlending;
    return THREE.NormalBlending;
  }

  /** Build (or rebuild) the two meshes that draw one layer. */
  function buildMeshes(L, tex, dims) {
    const THREE = G.THREE;
    const [w, h, d] = dims;
    const uniforms = {
      volTex:    { value: tex },
      lutTex:    { value: G.luts[L.colormap] || G.luts.gray },
      uCamLocal: { value: new THREE.Vector3() },
      uClim:     { value: new THREE.Vector2(0, 1) },
      uGamma:    { value: L.gamma },
      uOpacity:  { value: L.opacity },
      uIso:      { value: L.iso },
      uAtten:    { value: L.attenuation },
      uMode:     { value: Math.max(0, RENDER_MODES.indexOf(L.rendering)) },
      uSteps:    { value: clamp(Math.round(Math.max(w, h, d) * 1.6), 64, 512) },
      uSlice:    { value: 0.5 },
      uTexel:    { value: new THREE.Vector3(1 / w, 1 / h, 1 / d) },
    };

    const mat3 = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms, vertexShader: VOL_VERT, fragmentShader: VOL_FRAG,
      transparent: true, depthWrite: false, depthTest: false, side: THREE.BackSide,
      blending: blendingFor(L.blending),
    });
    const mat2 = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms, vertexShader: SLICE_VERT, fragmentShader: SLICE_FRAG,
      transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
      blending: blendingFor(L.blending),
    });

    const mesh3 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat3);
    const mesh2 = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat2);
    mesh3.frustumCulled = mesh2.frustumCulled = false;

    if (L._mesh3) { G.group3.remove(L._mesh3); L._mesh3.geometry.dispose(); }
    if (L._mesh2) { G.group2.remove(L._mesh2); L._mesh2.geometry.dispose(); }
    if (L._tex) L._tex.dispose();

    L._tex = tex; L._mesh3 = mesh3; L._mesh2 = mesh2;
    L._mat3 = mat3; L._mat2 = mat2; L._uniforms = uniforms;
    L.loaded = true;
    G.group3.add(mesh3);
    G.group2.add(mesh2);
    applyGeometryScale();
    applyLayer(L);
  }

  /** The volume box's world extents, normalised so the longest axis is 1. */
  function boxScale() {
    const [w, h, d] = API.dims;
    const m = Math.max(w, h, d) || 1;
    return [w / m, h / m, d / m];
  }

  /** Size every mesh (and the outline) to the base volume's aspect ratio. */
  function applyGeometryScale() {
    const [sx, sy, sz] = boxScale();
    for (const L of API.layers) {
      if (L._mesh3) L._mesh3.scale.set(sx, sy, sz);
      if (L._mesh2) L._mesh2.scale.set(sx, sy, 1);
    }
    if (G.box) G.box.scale.set(sx, sy, sz);
    // range markers and the implant model are authored in world units, so a
    // new box shape means laying them out again rather than simply rescaling
    buildRangeMarkers();
    if (API.implant) buildScrew(API.implant);
  }

  // ------------------------------------------------- missing-teeth range
  // Mirrors _set_missing_slice_manual in main46.py: a translucent red plane
  // on the first and last slice of the range plus a red boundary rectangle
  // ("Missing Slices" + "Missing Slice Boundaries" in the desktop viewer).
  // Bright cyan: the warm magma CBCT, red AB, yellow MC and green implant all
  // sit elsewhere on the wheel, so the boundary never blends into the data.
  // Start and end get their own colour so the two boundaries are never
  // confused. Both sit far from the CBCT magma ramp and from the red AB /
  // yellow MC / green implant layers.
  const RANGE_START = 0x2ee6ff;   // cyan
  const RANGE_END   = 0xff5ce1;   // magenta
  // Two independent thicknesses, both in world units. In-plane width is the
  // line weight you see face-on; Z depth is the cross-section that keeps the
  // frame visible when the volume is turned edge-on (a flat ring vanishes).
  const BAR_W = 0.0012;
  const BAR_D = 0.0035;

  function clearRangeMarkers() {
    for (const g of [G.rangeGroup, G.rangeGroup2]) {
      if (!g) continue;
      const seen = new Set();
      for (const child of [...g.children]) {
        g.remove(child);
        child.geometry.dispose();
        if (!seen.has(child.material)) { seen.add(child.material); child.material.dispose(); }
      }
    }
  }

  /** Local Z of a slice inside the unit box. */
  function sliceZ(z) {
    return API.sliceCount > 0 ? (z + 0.5) / API.sliceCount - 0.5 : 0;
  }

  /* A rectangle frame built from four solid bars rather than a flat outline.
     A flat ring is edge-on invisible when the volume is viewed from the side;
     bars keep a real cross-section from every angle. */
  function addFrame(target, sx, sy, zw, tw, td, opacity, vis, color) {
    const THREE = G.THREE;
    const mat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity,
      depthWrite: false, depthTest: false,
    });
    const hx = sx / 2, hy = sy / 2;
    const bars = [
      [sx + tw, tw, td, 0, hy], [sx + tw, tw, td, 0, -hy],   // top / bottom
      [tw, sy - tw, td, -hx, 0], [tw, sy - tw, td, hx, 0],   // left / right
    ];
    for (const [w, h, d, x, y] of bars) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      m.position.set(x, y, zw);
      m.renderOrder = 900;
      m.frustumCulled = false;
      m.visible = vis;
      target.add(m);
    }
  }

  function addFill(target, sx, sy, zw, opacity, vis, order, color) {
    const THREE = G.THREE;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(sx, sy),
      new THREE.MeshBasicMaterial({
        color, transparent: true, opacity,
        side: THREE.DoubleSide, depthWrite: false, depthTest: false,
      }));
    m.position.z = zw;
    m.renderOrder = order;
    m.frustumCulled = false;
    m.visible = vis;
    target.add(m);
  }

  function buildRangeMarkers() {
    const THREE = G.THREE;
    if (!THREE || !G.rangeGroup) return;
    clearRangeMarkers();
    const { start, end } = API.range;
    if (start == null || end == null) return;
    const L = API.layers.find((l) => l.kind === "range");
    const vis = !L || L.visible;
    const op = L ? L.opacity : 0.85;
    const [sx, sy, sz] = boxScale();

    // 3D — cyan frame on the first slice, magenta on the last
    const marks = start === end
      ? [[start, RANGE_START]]
      : [[start, RANGE_START], [end, RANGE_END]];
    for (const [z, color] of marks) {
      const zw = sliceZ(z) * sz;
      addFill(G.rangeGroup, sx, sy, zw, 0.08 * op, vis, 899, color);
      addFrame(G.rangeGroup, sx, sy, zw, BAR_W, BAR_D, op, vis, color);
    }

    // 2D — the boundary slice you are on gets its own colour, full strength
    const atStart = API.slice === start;
    const atEnd = API.slice === end;
    const inside = API.slice > start && API.slice < end;
    if (atStart || atEnd || inside) {
      const color = atEnd && !atStart ? RANGE_END : RANGE_START;
      const onBoundary = atStart || atEnd;
      addFrame(G.rangeGroup2, sx, sy, 0,
               onBoundary ? BAR_W : BAR_W * 0.7, BAR_D,
               (onBoundary ? 1.0 : 0.45) * op, vis, color);
      if (onBoundary) addFill(G.rangeGroup2, sx, sy, 0, 0.07 * op, vis, 899, color);
    }
    G.needsRender = true;
  }

  // ------------------------------------------------------- implant model
  /* The clinical mask in implant_only/ is a stack of rectangles — correct as a
     footprint, but it renders as a plain cylinder. This builds a true-to-scale
     screw at the same planned position: tapered core, helical thread, and a
     prosthetic platform, in titanium. Purely a rendering; the mask and the
     recommendation algorithm are untouched. */
  const SCREW = {
    core: 0.70,      // core radius, as a fraction of the outer radius
    apex: 0.52,      // core radius at the apex
    thread: 0.17,    // thread wire radius, as a fraction of the outer radius
    pitchMm: 0.8,    // real implants sit near 0.8 mm between turns
    collar: 0.10,    // platform height, as a fraction of total length
    tip: 0.14,       // conical apex, as a fraction of total length
  };

  function disposeScrew() {
    if (!G.screwGroup) return;
    for (const m of [...G.screwGroup.children]) {
      G.screwGroup.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
  }

  /** Helix from just under the platform to just above the apex, tapering into it. */
  function threadCurve(THREE, r, yTop, yBot, turns) {
    const pts = [];
    const steps = Math.max(64, Math.round(turns * 26));
    const rw = r * SCREW.thread;
    const TAPER_FROM = 0.72;                     // start narrowing near the tip
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;                       // 0 at platform, 1 at apex end
      const k = t < TAPER_FROM
        ? 1
        : 1 - 0.55 * ((t - TAPER_FROM) / (1 - TAPER_FROM));
      const rad = (r - rw) * k;
      const a = 2 * Math.PI * turns * t;
      pts.push(new THREE.Vector3(Math.cos(a) * rad,
                                 yTop + (yBot - yTop) * t,
                                 Math.sin(a) * rad));
    }
    return new THREE.CatmullRomCurve3(pts);
  }

  /** Build the implant at the planned position. `p` is state.applied + shape. */
  function buildScrew(p) {
    const THREE = G.THREE;
    if (!THREE || !G.screwGroup) return;
    disposeScrew();
    if (!p) return;

    const [W, H, Z] = [p.shape[2], p.shape[1], p.shape[0]];
    const [sx, sy, sz] = boxScale();
    const px = p.pixel_size_mm || 0.3;
    const dPx = p.diameter_mm / px;
    const hPx = p.height_mm / px;

    // voxel -> world is isotropic, because the box is scaled by the voxel dims
    const r = (dPx / W) * sx / 2;
    const len = (hPx / H) * sy;
    if (!(r > 0) || !(len > 0)) return;

    // three.js draws the opaque queue before the transparent one, and the
    // volume layers are transparent with depthTest off — an opaque screw would
    // be painted over. Flagging it transparent puts it in the same queue, where
    // renderOrder decides; depth test/write still give it correct self-occlusion.
    // There is no environment map in this scene, and a near-1 metalness has
    // nothing to reflect, so it reads almost black. Keeping metalness moderate
    // lets the diffuse term carry the titanium colour under the three lights.
    const metal = new THREE.MeshStandardMaterial({
      color: 0xe3e9ef, metalness: 0.45, roughness: 0.33,
      emissive: 0x2b3440, emissiveIntensity: 0.5,
      transparent: true, opacity: 1.0, depthTest: true, depthWrite: true,
    });
    const add = (geo, y) => {
      const m = new THREE.Mesh(geo, metal);
      m.position.y = y;
      m.renderOrder = 950;
      m.frustumCulled = false;
      G.screwGroup.add(m);
      return m;
    };

    const collarH = len * SCREW.collar;
    const tipH = len * SCREW.tip;
    const bodyH = len - collarH - tipH;
    const rc = r * SCREW.core, ra = r * SCREW.apex;

    // prosthetic platform, tapered body, conical apex
    add(new THREE.CylinderGeometry(r * 0.97, r * 0.88, collarH, 40),
        len / 2 - collarH / 2);
    add(new THREE.CylinderGeometry(rc, rc, bodyH, 40),
        len / 2 - collarH - bodyH / 2);
    add(new THREE.ConeGeometry(ra, tipH, 40),
        -len / 2 + tipH / 2).rotation.x = Math.PI;

    const turns = Math.max(4, Math.round(p.height_mm / SCREW.pitchMm));
    const yTop = len / 2 - collarH;                    // just below the platform
    const yBot = -len / 2 + tipH * 0.72;               // stop above the apex
    add(new THREE.TubeGeometry(threadCurve(THREE, r, yTop, yBot, turns),
                               Math.max(120, turns * 24), r * SCREW.thread, 10, false), 0);

    // centre of the planned footprint, converted to world space
    const cxPx = p.x_px + dPx / 2;
    const cyPx = p.y_px + hPx / 2;
    const czSl = (p.start_slice + p.end_slice) / 2;
    G.screwGroup.userData.pos = new THREE.Vector3(
      (cxPx / W - 0.5) * sx,
      (0.5 - cyPx / H) * sy,
      ((czSl + 0.5) / Z - 0.5) * sz);
    placeScrew();
  }

  /** 3D sits at the planned depth; 2D floats just in front of the slice plane. */
  function placeScrew() {
    const g = G.screwGroup;
    if (!g || !g.userData.pos) return;
    const p = g.userData.pos;
    g.position.set(p.x, p.y, API.ndisplay === 3 ? p.z : 0.02);
    const L = API.layers.find((l) => l.kind === "implant");
    const inSlice = API.ndisplay === 3 || !API.implant ||
      (API.slice >= API.implant.start_slice && API.slice <= API.implant.end_slice);
    g.visible = Boolean(API.implant) && inSlice &&
      Boolean(L && L.visible && L.rendering === "screw");
    G.needsRender = true;
  }

  /** Give the viewer the applied implant (or null to drop the model). */
  API.setImplant = function setImplant(applied, shape) {
    API.implant = applied && shape ? { ...applied, shape } : null;
    buildScrew(API.implant);
    const L = API.layers.find((l) => l.kind === "implant");
    if (L && API.implant && !L._touchedRendering) L.rendering = "screw";
    placeScrew();
    emit();
  };

  /** Show (or clear, with nulls) the missing-teeth boundary markers. */
  API.setMissingRange = function setMissingRange(start, end) {
    const has = start != null && end != null;
    API.range = has
      ? { start: clamp(Math.round(start), 0, API.sliceCount - 1),
          end: clamp(Math.round(end), 0, API.sliceCount - 1) }
      : { start: null, end: null };
    syncRangeLayerEntry();
    buildRangeMarkers();
    emit();
  };

  /** Keep a "Missing Slices" row in the layer list, like napari does. */
  function syncRangeLayerEntry() {
    const has = API.range.start != null;
    const i = API.layers.findIndex((l) => l.kind === "range");
    if (!has) {
      if (i >= 0) API.layers.splice(i, 1);
      return;
    }
    if (i < 0) {
      API.layers.unshift({
        kind: "range", name: "Missing Slices", available: true,
        visible: true, synthetic: true, opacity: 0.85, rendering: "range",
        colormap: "cyan", blending: "translucent",
      });
    }
  }

  function applyLayer(L) {
    if (L.synthetic) { buildRangeMarkers(); return; }
    if (!L._uniforms) return;
    const asScrew = L.kind === "implant" && L.rendering === "screw";
    const u = L._uniforms;
    u.lutTex.value = G.luts[L.colormap] || G.luts.gray;
    u.uGamma.value = L.gamma;
    u.uOpacity.value = L.opacity;
    u.uIso.value = L.iso;
    u.uAtten.value = L.attenuation;
    u.uMode.value = Math.max(0, RENDER_MODES.indexOf(L.rendering));
    // contrast limits live in data units; normalise into texture space
    const span = Math.max(L.range[1] - L.range[0], 1e-9);
    u.uClim.value.set(clamp((L.climLo - L.range[0]) / span, 0, 1),
                      clamp((L.climHi - L.range[0]) / span, 0, 1));
    u.uSlice.value = API.sliceCount > 1
      ? clamp((API.slice + 0.5) / API.sliceCount, 0, 1) : 0.5;
    const blend = blendingFor(L.blending);
    L._mat3.blending = blend; L._mat2.blending = blend;
    L._mat3.needsUpdate = L._mat2.needsUpdate = true;
    L._mesh3.visible = L.visible && !asScrew;
    L._mesh2.visible = L.visible && !asScrew;
    if (L.kind === "implant") placeScrew();
    const order = API.layers.indexOf(L);
    L._mesh3.renderOrder = order;
    L._mesh2.renderOrder = order;
    G.needsRender = true;
  }

  function applyAll() {
    API.layers.forEach((L) => { if (!L.synthetic) applyLayer(L); });
  }

  // ------------------------------------------------------------ layer i/o
  function defaultsFor(meta) {
    const isBase = meta.kind === "base";
    return {
      kind: meta.kind,
      name: meta.name,
      available: meta.available,
      visible: true,
      loaded: false,
      colormap: meta.colormap,
      rendering: meta.rendering,
      blending: meta.blending,
      opacity: meta.opacity ?? 1.0,   // backend supplies the napari default
      gamma: isBase ? 1.2 : 1.0,          // desktop preset: gamma 1.2, magma
      iso: meta.iso ?? 0.5,
      attenuation: 0.05,
      climLo: 0, climHi: 1,
      range: [0, 1],
      _touchedClim: false,
    };
  }

  async function fetchLayer(L) {
    const res = await fetch(`/api/session/${API.sid}/layer/${L.kind}/raw?max_side=${API.quality}`);
    if (!res.ok) throw new Error(`${L.kind}: HTTP ${res.status}`);
    const dims = (res.headers.get("X-Vol-Dims") || "").split(",").map((n) => parseInt(n, 10));
    const rng = (res.headers.get("X-Vol-Range") || "0,1").split(",").map(Number);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const [w, h, d] = [dims[0] || 1, dims[1] || 1, dims[2] || 1];
    L.range = [rng[0], rng[1]];
    if (!L._touchedClim) {
      if (L.kind === "base") {
        // desktop preset — contrast limits (0.5, 1.0) in data units, clamped
        L.climLo = clamp(0.5, L.range[0], L.range[1]);
        L.climHi = clamp(1.0, L.climLo + 1e-6, L.range[1]);
      } else {
        L.climLo = L.range[0]; L.climHi = L.range[1];
      }
    }
    if (L.kind === "base") API.dims = [w, h, d];
    buildMeshes(L, upload(bytes, w, h, d), [w, h, d]);
  }

  function dropLayer(L) {
    if (L._mesh3) { G.group3.remove(L._mesh3); L._mesh3.geometry.dispose(); L._mat3.dispose(); }
    if (L._mesh2) { G.group2.remove(L._mesh2); L._mesh2.geometry.dispose(); L._mat2.dispose(); }
    if (L._tex) L._tex.dispose();
    L._mesh3 = L._mesh2 = L._mat3 = L._mat2 = L._tex = L._uniforms = null;
    L.loaded = false;
  }

  /** Ask the backend which layers exist, then (re)load the ones that do. */
  API.sync = async function sync(sid) {
    if (sid) API.sid = sid;
    if (!API.sid) return;
    const token = ++G.token;
    const res = await fetch(`/api/session/${API.sid}/layers`);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const j = await res.json(); if (j && j.detail) detail = j.detail; } catch (_) { /* non-json */ }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    const info = await res.json();
    if (!info || !Array.isArray(info.layers)) {
      throw new Error("malformed response from /layers");
    }

    const byKind = new Map(API.layers.filter((l) => !l.synthetic).map((l) => [l.kind, l]));
    const next = info.layers.map((meta) => {
      const prev = byKind.get(meta.kind);
      if (!prev) return defaultsFor(meta);
      prev.available = meta.available;
      return prev;
    });
    if (token !== G.token) return;
    API.layers = next;
    syncRangeLayerEntry();

    for (const L of API.layers) {
      if (L.synthetic) continue;
      if (!L.available) { if (L.loaded) dropLayer(L); continue; }
      await fetchLayer(L);
      if (token !== G.token) return;
    }
    API.slice = clamp(API.slice, 0, API.sliceCount - 1);
    applyGeometryScale();
    applyAll();
    buildRangeMarkers();
    emit();
  };

  /** Point the viewer at a session — clears any previous layer state. */
  API.attach = async function attach(sid, hostEl) {
    if (!API.ready) await boot(hostEl || G.host);
    if (sid !== API.sid) {
      API.layers.forEach((L) => { if (!L.synthetic) dropLayer(L); });
      API.layers = [];
      API.range = { start: null, end: null };
      API.implant = null;
      clearRangeMarkers();
      disposeScrew();
      API.sid = sid;
    }
    await API.sync(sid);
    API.resetView();
  };

  API.reload = () => API.sync(API.sid);

  // ----------------------------------------------------------- public ops
  API.onChange = (cb) => { API._onChange = cb; };

  API.set = function set(kind, patch) {
    const L = API.layers.find((l) => l.kind === kind);
    if (!L) return;
    if ("climLo" in patch || "climHi" in patch) L._touchedClim = true;
    if ("rendering" in patch) L._touchedRendering = true;
    Object.assign(L, patch);
    applyLayer(L);
    emit();
  };

  API.setNdisplay = function setNdisplay(n) {
    API.ndisplay = n === 2 ? 2 : 3;
    if (G.controls3) {
      G.controls3.enabled = API.ndisplay === 3;
      G.controls2.enabled = API.ndisplay === 2;
      G.needsRender = true;
    }
    placeScrew();
    emit();
  };

  API.setSlice = function setSlice(z, total) {
    if (total) API.sliceCount = Math.max(1, Math.round(total));
    API.slice = clamp(Math.round(z), 0, API.sliceCount - 1);
    applyAll();
    buildRangeMarkers();
    placeScrew();
  };

  API.setQuality = async function setQuality(maxSide) {
    API.quality = clamp(maxSide, 32, 320);
    API.layers.forEach((L) => { if (!L.synthetic) L.loaded = false; });
    await API.sync(API.sid);
  };

  API.resetView = function resetView() {
    if (!G.cam3) return;
    G.cam3.position.set(1.5, 1.05, 1.75);
    G.controls3.target.set(0, 0, 0);
    G.controls3.update();
    G.cam2.position.set(0, 0, 2);
    G.controls2.target.set(0, 0, 0);
    G.cam2.zoom = 1;
    fitOrtho();
    G.controls2.update();
    G.needsRender = true;
  };

  API.screenshot = function screenshot() {
    if (!G.renderer) return null;
    render();
    return G.renderer.domElement.toDataURL("image/png");
  };

  API.boot = boot;
})();
