/* vol3d.js — free-orbit 3D CBCT rendering (GPU ray marching via three.js).
 * Exposes window.Vol3D: { ready, show(sid), hide(), refresh(sid),
 *                         setThreshold(t0_100), setOpacity(v), active }
 */
(() => {
  "use strict";

  const API = {
    ready: false,
    active: false,
    _sid: null,
    _tex: null,
    _texDims: null,
    _mat: null,
    _renderer: null,
    _scene: null,
    _cam: null,
    _controls: null,
    _container: null,
    _loadToken: 0,
  };
  window.Vol3D = API;

  async function init() {
    const THREE = await import("three");
    window.__THREE_REF = THREE;
    const { OrbitControls } = await import("./vendor/OrbitControls.js");

    const host = document.getElementById("vol3d");
    API._container = host;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    cam.position.set(1.35, 0.95, 1.6);

    const controls = new OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.9;
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.6;
    controls.maxDistance = 8;

    // ---- material: ray-march the volume texture -----------------------
    const material = new THREE.ShaderMaterial({
      uniforms: {
        volTex:     { value: null },
        uThreshold: { value: 0.28 },   // 0..1 of the normalized density
        uOpacity:   { value: 0.36 },   // per-sample density scale
      },
      vertexShader: /* glsl */`
        varying vec3 vWorld;
        void main() {
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vWorld = wp.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */`
        precision highp float;
        uniform sampler3D volTex;
        uniform float uThreshold;
        uniform float uOpacity;
        varying vec3 vWorld;

        void main() {
          vec3 ro = cameraPosition;
          vec3 rd = normalize(vWorld - ro);
          // slab test against axis-aligned box [-0.5, 0.5]^3
          vec3 inv = 1.0 / rd;
          vec3 t0 = (-0.5 - ro) * inv;
          vec3 t1 = ( 0.5 - ro) * inv;
          vec3 tmin = min(t0, t1);
          vec3 tmax = max(t0, t1);
          float tNear = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));
          float tFar  = min(tmax.x, min(tmax.y, tmax.z));
          if (tNear > tFar) discard;

          float span  = tFar - tNear;
          int steps   = int(ceil(span * 140.0));
          steps       = clamp(steps, 24, 220);
          float dt    = span / float(steps);

          float alpha = 0.0;
          vec3  acc   = vec3(0.0);
          for (int i = 0; i < 256; i++) {
            if (i >= steps) break;
            float t = tNear + (float(i) + 0.5) * dt;
            if (t > tFar) break;
            vec3 uvw = (ro + rd * t) * 0.5 + 0.5;
            if (uvw.x < 0.0 || uvw.x > 1.0 ||
                uvw.y < 0.0 || uvw.y > 1.0 ||
                uvw.z < 0.0 || uvw.z > 1.0) continue;

            float d = texture3D(volTex, uvw).r;
            float dens = smoothstep(uThreshold, uThreshold + 0.25, d) * uOpacity;
            if (dens >= 0.995) {
              vec3 bone = mix(vec3(0.02, 0.18, 0.40), vec3(0.60, 0.93, 1.0),
                              smoothstep(0.15, 1.0, d));
              acc  = bone;
              alpha = 1.0;
              break;
            }
            if (dens > 0.002) {
              float w = dens * (1.0 - alpha);
              vec3 bone = mix(vec3(0.02, 0.18, 0.40), vec3(0.60, 0.93, 1.0),
                              smoothstep(0.15, 1.0, d));
              acc   += w * bone;
              alpha += w;
            }
          }
          if (alpha < 0.004) discard;
          gl_FragColor = vec4(acc / alpha, 1.0);
        }`,
      side: THREE.FrontSide,
    });

    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    scene.add(box);

    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.001, 1.001, 1.001)),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.18 }));
    scene.add(edge);

    API._renderer = renderer;
    API._scene = scene;
    API._cam = cam;
    API._controls = controls;
    API._mat = material;
    API.ready = true;

    const resize = () => {
      const w = Math.max(1, host.clientWidth);
      const h = Math.max(1, host.clientHeight);
      renderer.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(host);
    resize();

    (function loop() {
      requestAnimationFrame(loop);
      if (API.active) {
        controls.update();
        renderer.render(scene, cam);
      }
    })();
  }

  function makeTexture(bytes, w, h, d) {
    const THREE = window.__THREE_REF;
    const rgba = new Uint8Array(w * h * d * 4);
    for (let i = 0; i < w * h * d; i++) {
      rgba[i * 4 + 0] = bytes[i];
      rgba[i * 4 + 1] = bytes[i];
      rgba[i * 4 + 2] = bytes[i];
      rgba[i * 4 + 3] = 255;
    }
    const tex = new THREE.Data3DTexture(rgba, w, h, d);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  async function load(sid) {
    const token = ++API._loadToken;
    const res = await fetch(`/api/session/${sid}/volume-raw`);
    if (!res.ok) throw new Error(`volume-raw HTTP ${res.status}`);
    const dims = (res.headers.get("X-Vol-Dims") || "").split(",");
    const w = parseInt(dims[0], 10) || 64;
    const h = parseInt(dims[1], 10) || 64;
    const d = parseInt(dims[2], 10) || 64;
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (token !== API._loadToken) return;   // superseded
    const tex = makeTexture(bytes, w, h, d);
    const old = API._tex;
    API._tex = tex;
    API._texDims = [w, h, d];
    API._mat.uniforms.volTex.value = tex;
    if (old) old.dispose();
  }

  API.show = async (sid) => {
    if (!API.ready) await init();
    if (sid && sid !== API._sid) {
      API._sid = sid;
      try { await load(sid); }
      catch (e) { toastVol3d(e.message); return; }
    }
    API.active = true;
    const host = API._container;
    host.hidden = false;
    host.clientWidth; // force layout
    API._renderer.setSize(Math.max(1, host.clientWidth), Math.max(1, host.clientHeight), false);
    API._cam.aspect = Math.max(1, host.clientWidth) / Math.max(1, host.clientHeight);
    API._cam.updateProjectionMatrix();
  };

  API.hide = () => {
    API.active = false;
    API._container.hidden = true;
  };

  API.refresh = async (sid) => {
    if (sid) API._sid = sid;
    if (API.active && API.ready) {
      try { await load(API._sid); } catch (e) { toastVol3d(e.message); }
    }
  };

  API.setThreshold = (outOf100)   => { if (API._mat) API._mat.uniforms.uThreshold.value = Math.max(0.02, Math.min(0.95, outOf100 / 100)); };
  API.setOpacity = (outOf25)     => { if (API._mat) API._mat.uniforms.uOpacity.value = Math.max(0.05, Math.min(1.0, outOf25 / 25)); };

  function toastVol3d(msg) {
    const t = document.getElementById("toast");
    if (t) { t.textContent = msg; t.className = "toast show err"; }
  }

  // defer until after DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init().catch(() => {}));
  } else {
    init().catch(() => {});
  }
})();
