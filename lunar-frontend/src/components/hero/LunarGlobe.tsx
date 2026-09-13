"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

import type { MoonPoint } from "@/lib/backend-types";

export type PayloadMode = "optical" | "iirs" | "dem";
export type LunarPhase = "crescent" | "quarter" | "gibbous" | "full" | "new";

export interface LunarFootprint {
  id: string;
  west_lon: number;
  east_lon: number;
  south_lat: number;
  north_lat: number;
}

interface Props {
  payloadMode?: PayloadMode;
  phase?: LunarPhase;
  autoRotate?: boolean;
  tiePoints?: MoonPoint[];
  footprints?: LunarFootprint[];
  activeFootprintId?: string | null;
  className?: string;
}

// ---------------------------------------------------------------------------
// Public, CORS-enabled real-Moon imagery (NASA LROC-derived, public domain).
// Tried in order; jsdelivr mirrors unpkg. Final fallback is three.js's own
// moon texture on GitHub raw (also CORS `*`). Offline / blocked network falls
// back to an improved procedural texture (never a blank globe).
// ---------------------------------------------------------------------------
const COLOR_URLS = [
  "https://unpkg.com/three-globe@2.31.0/example/img/lunar_surface.jpg",
  "https://cdn.jsdelivr.net/npm/three-globe@2.31.0/example/img/lunar_surface.jpg",
  "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/moon_1024.jpg",
];

const BUMP_URLS = [
  "https://unpkg.com/three-globe@2.31.0/example/img/lunar_bumpmap.jpg",
  "https://cdn.jsdelivr.net/npm/three-globe@2.31.0/example/img/lunar_bumpmap.jpg",
];

const MOON_RADIUS = 2.2;
// Lift the globe slightly above canvas centre so the bottom of the panel keeps
// generous breathing room (the old y=-0.65 cropped the limb at 480px).
const MOON_Y = 0.42;

function normalizeLon360(lon: number): number {
  return ((lon % 360) + 360) % 360;
}

// Convert planetocentric lat / lon (deg, lon in 0..360 or -180..180) to a
// position on the sphere. Matches THREE.SphereGeometry UV mapping so markers
// and footprint quads sit exactly on the equirectangular texture.
function latLonToVec3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const lon360 = normalizeLon360(lon);
  const phi = (lon360 / 360) * Math.PI * 2;
  const polar = ((90 - lat) * Math.PI) / 180;
  const x = -radius * Math.cos(phi) * Math.sin(polar);
  const y = radius * Math.cos(polar);
  const z = radius * Math.sin(phi) * Math.sin(polar);
  return new THREE.Vector3(x, y, z);
}

function loadTextureFirstAvailable(urls: string[]): Promise<THREE.Texture | null> {
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin("anonymous");
  return (async () => {
    for (const url of urls) {
      try {
        const tex = await loader.loadAsync(url);
        return tex;
      } catch {
        // try next mirror
      }
    }
    return null;
  })();
}

// ---------------------------------------------------------------------------
// Improved procedural fallback (offline only): pole-safe, maria-aware.
// Never draws near the poles (|lat| > ~78deg) so no UV pinch artifact.
// ---------------------------------------------------------------------------
function createFallbackTextures(mode: PayloadMode): {
  colorMap: THREE.CanvasTexture;
  bumpMap: THREE.CanvasTexture;
} {
  const width = 2048;
  const height = 1024;

  const colorCanvas = document.createElement("canvas");
  colorCanvas.width = width;
  colorCanvas.height = height;
  const ctx = colorCanvas.getContext("2d")!;

  const bumpCanvas = document.createElement("canvas");
  bumpCanvas.width = width;
  bumpCanvas.height = height;
  const bCtx = bumpCanvas.getContext("2d")!;

  ctx.fillStyle = mode === "optical" ? "#8f949c" : mode === "iirs" ? "#223349" : "#1d4a40";
  ctx.fillRect(0, 0, width, height);
  bCtx.fillStyle = "#808080";
  bCtx.fillRect(0, 0, width, height);

  // Nearside maria in equirectangular UV space (lon -180..180 -> u, lat -> v).
  // u = (lon+180)/360 * W ; v = (90-lat)/180 * H
  const toXY = (lon: number, lat: number) => ({
    x: ((lon + 180) / 360) * width,
    y: ((90 - lat) / 180) * height,
  });
  const maria = [
    { lon: -56, lat: 18, rx: 260, ry: 150, tone: "#4a4d53" }, // Oceanus Procellarum
    { lon: -15, lat: 33, rx: 150, ry: 115, tone: "#43464c" }, // Mare Imbrium
    { lon: 17, lat: 28, rx: 120, ry: 95, tone: "#3f4247" }, // Mare Serenitatis
    { lon: 31, lat: 8, rx: 115, ry: 85, tone: "#3a3d43" }, // Mare Tranquillitatis
    { lon: 59, lat: 17, rx: 70, ry: 55, tone: "#363940" }, // Mare Crisium
    { lon: 54, lat: -3, rx: 95, ry: 70, tone: "#3a3d43" }, // Mare Fecunditatis
    { lon: -16, lat: -21, rx: 90, ry: 70, tone: "#414449" }, // Mare Nubium
    { lon: -39, lat: -24, rx: 70, ry: 60, tone: "#43464c" }, // Mare Humorum
  ];
  maria.forEach((m) => {
    const { x, y } = toXY(m.lon, m.lat);
    const grad = ctx.createRadialGradient(x, y, 8, x, y, m.rx);
    grad.addColorStop(0, m.tone);
    grad.addColorStop(0.75, m.tone);
    grad.addColorStop(1, "transparent");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(x, y, m.rx, m.ry, 0, 0, Math.PI * 2);
    ctx.fill();

    const bGrad = bCtx.createRadialGradient(x, y, 8, x, y, m.rx);
    bGrad.addColorStop(0, "#585858");
    bGrad.addColorStop(1, "#808080");
    bCtx.fillStyle = bGrad;
    bCtx.beginPath();
    bCtx.ellipse(x, y, m.rx, m.ry, 0, 0, Math.PI * 2);
    bCtx.fill();
  });

  // Named craters at true selenographic coords (pole-safe by construction).
  const toXYr = (lon: number, lat: number, km: number) => {
    const { x, y } = toXY(lon, lat);
    return { x, y, r: Math.max(10, km / 6) };
  };
  const craters = [
    { ...toXYr(-11.4, -43.3, 85), rays: 16 }, // Tycho
    { ...toXYr(-20.1, 9.6, 93), rays: 10 }, // Copernicus
    { ...toXYr(-38, 8.1, 32), rays: 8 }, // Kepler
    { ...toXYr(15.5, 51.6, 109), rays: 0 }, // Plato
    { ...toXYr(5.4, -0.7, 34), rays: 0 }, // Moltke (near region_001 area)
    { ...toXYr(-23.4, -3.2, 12), rays: 0 }, // Sinus Medii marker crater
  ];
  craters.forEach((c) => {
    if (c.rays > 0) {
      ctx.save();
      ctx.translate(c.x, c.y);
      for (let i = 0; i < c.rays; i++) {
        const ang = (Math.PI * 2 * i) / c.rays + Math.sin(i * 3) * 0.15;
        const rayLen = c.r * 5;
        const rayGrad = ctx.createLinearGradient(0, 0, Math.cos(ang) * rayLen, Math.sin(ang) * rayLen);
        rayGrad.addColorStop(0, "rgba(235,240,248,0.4)");
        rayGrad.addColorStop(1, "rgba(235,240,248,0)");
        ctx.strokeStyle = rayGrad;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(ang) * rayLen, Math.sin(ang) * rayLen);
        ctx.stroke();
      }
      ctx.restore();
    }
    const cGrad = ctx.createRadialGradient(c.x, c.y, c.r * 0.2, c.x, c.y, c.r);
    cGrad.addColorStop(0, "#2e3138");
    cGrad.addColorStop(0.7, "#54595f");
    cGrad.addColorStop(0.86, "#c9cfd8");
    cGrad.addColorStop(1, "transparent");
    ctx.fillStyle = cGrad;
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.fill();

    const bCrater = bCtx.createRadialGradient(c.x, c.y, c.r * 0.1, c.x, c.y, c.r);
    bCrater.addColorStop(0, "#2a2a2a");
    bCrater.addColorStop(0.7, "#4c4c4c");
    bCrater.addColorStop(0.88, "#e8e8e8");
    bCrater.addColorStop(1, "#808080");
    bCtx.fillStyle = bCrater;
    bCtx.beginPath();
    bCtx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    bCtx.fill();
  });

  // Fine regolith grain, kept away from the poles to avoid pinch streaks.
  let seed = 1337;
  const rand = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
  for (let i = 0; i < 900; i++) {
    const rx = rand() * width;
    const v = rand();
    if (v < 0.06 || v > 0.94) continue; // pole guard
    const ry = v * height;
    const r = rand() * 5 + 1;
    const g = ctx.createRadialGradient(rx, ry, r * 0.2, rx, ry, r);
    g.addColorStop(0, "rgba(45,49,55,0.45)");
    g.addColorStop(0.8, "rgba(200,208,220,0.25)");
    g.addColorStop(1, "transparent");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(rx, ry, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const colorMap = new THREE.CanvasTexture(colorCanvas);
  colorMap.colorSpace = THREE.SRGBColorSpace;
  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  return { colorMap, bumpMap };
}

function makeLabelSprite(title: string, sub: string, active: boolean): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const g = canvas.getContext("2d")!;
  const accent = active ? "#fbbf24" : "#67e8f9";
  g.fillStyle = "rgba(2,6,23,0.88)";
  g.strokeStyle = accent;
  g.lineWidth = 3;
  const r = 22;
  g.beginPath();
  if (typeof (g as CanvasRenderingContext2D & { roundRect?: Function }).roundRect === "function") {
    (g as CanvasRenderingContext2D & { roundRect: (x: number, y: number, w: number, h: number, r: number) => void }).roundRect(6, 6, 500, 116, r);
  } else {
    g.rect(6, 6, 500, 116);
  }
  g.fill();
  g.stroke();
  g.fillStyle = "#ffffff";
  g.font = "700 44px ui-monospace, monospace";
  g.textBaseline = "middle";
  g.fillText(title, 28, 48);
  g.fillStyle = accent;
  g.font = "500 30px ui-monospace, monospace";
  g.fillText(sub, 28, 92);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.05, 0.26, 1);
  return sprite;
}

export default function LunarGlobe({
  payloadMode = "optical",
  phase = "full",
  autoRotate = true,
  tiePoints,
  footprints,
  activeFootprintId,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const moonRef = useRef<THREE.Mesh | null>(null);
  const overlayRef = useRef<THREE.Group | null>(null);
  const sunLightRef = useRef<THREE.DirectionalLight | null>(null);
  const animFrameId = useRef<number | null>(null);
  const autoRotateRef = useRef(autoRotate);
  autoRotateRef.current = autoRotate;

  const isDragging = useRef(false);
  const prevPointer = useRef({ x: 0, y: 0 });
  const rotVel = useRef({ x: 0, y: 0.0012 });
  const [ready, setReady] = useState(false);
  const [texSource, setTexSource] = useState<"loading" | "lroc" | "fallback">("loading");
  const [globeKey, setGlobeKey] = useState(0);

  // Tint the real LROC texture per payload mode (hero modes keep working
  // without needing separate spectral textures).
  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    if (payloadMode === "optical") mat.color.set("#ffffff");
    else if (payloadMode === "iirs") mat.color.set("#b9ccff");
    else mat.color.set("#c2e8cf");
  }, [payloadMode, globeKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 1000);
    camera.position.set(0, MOON_Y, 7.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Fully front-lit sun by default so maria + region overlays stay visible.
    const sunLight = new THREE.DirectionalLight(0xfff6e8, 2.6);
    sunLight.position.set(1.0, 0.9, 7.0);
    scene.add(sunLight);
    sunLightRef.current = sunLight;

    const hemi = new THREE.HemisphereLight(0xbfd4ff, 0x201812, 0.5);
    scene.add(hemi);
    const earthShine = new THREE.DirectionalLight(0x3a5a8a, 0.35);
    earthShine.position.set(-6, -1.5, -2);
    scene.add(earthShine);
    scene.add(new THREE.AmbientLight(0xffffff, 0.32));

    const geo = new THREE.SphereGeometry(MOON_RADIUS, 96, 96);
    const mat = new THREE.MeshStandardMaterial({
      color: payloadMode === "optical" ? "#ffffff" : payloadMode === "iirs" ? "#b9ccff" : "#c2e8cf",
      roughness: 1.0,
      metalness: 0.0,
      bumpScale: 0.05,
    });
    materialRef.current = mat;
    const moon = new THREE.Mesh(geo, mat);
    moon.position.set(0, MOON_Y, 0);
    moon.rotation.y = 0.85;
    scene.add(moon);
    moonRef.current = moon;

    const overlay = new THREE.Group();
    moon.add(overlay);
    overlayRef.current = overlay;

    // Subtle celestial rim (thin, additive) — no longer a thick cartoon ring.
    const glowGeo = new THREE.SphereGeometry(MOON_RADIUS * 1.022, 64, 64);
    const glowMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float f = clamp(1.0 - dot(vNormal, viewDir), 0.0, 1.0);
          float intensity = pow(f, 3.6);
          gl_FragColor = vec4(vec3(0.36, 0.6, 0.95), intensity * 0.55);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });
    const glowMesh = new THREE.Mesh(glowGeo, glowMat);
    glowMesh.position.copy(moon.position);
    scene.add(glowMesh);

    // Starfield shell for depth.
    const starGeo = new THREE.BufferGeometry();
    const starCount = 1200;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 30 + Math.random() * 40;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      starPos[i * 3] = r * Math.sin(p) * Math.cos(t);
      starPos[i * 3 + 1] = r * Math.cos(p);
      starPos[i * 3 + 2] = r * Math.sin(p) * Math.sin(t);
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const stars = new THREE.Points(
      starGeo,
      new THREE.PointsMaterial({ color: 0xffffff, size: 0.14, transparent: true, opacity: 0.75 })
    );
    scene.add(stars);

    // --- Texture pipeline: public real-Moon APIs first, procedural fallback.
    (async () => {
      const colorTex = await loadTextureFirstAvailable(COLOR_URLS);
      if (disposed) return;
      if (colorTex) {
        colorTex.colorSpace = THREE.SRGBColorSpace;
        colorTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
        mat.map = colorTex;
        const bumpTex = await loadTextureFirstAvailable(BUMP_URLS);
        if (disposed) return;
        if (bumpTex) {
          mat.bumpMap = bumpTex;
        } else {
          mat.bumpMap = colorTex;
        }
        mat.bumpScale = 0.05;
        mat.needsUpdate = true;
        setTexSource("lroc");
      } else {
        const { colorMap, bumpMap } = createFallbackTextures(payloadMode);
        mat.map = colorMap;
        mat.bumpMap = bumpMap;
        mat.bumpScale = 0.05;
        mat.needsUpdate = true;
        setTexSource("fallback");
      }
      setGlobeKey((k) => k + 1);
    })();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    window.addEventListener("resize", handleResize);

    const onPointerDown = (e: MouseEvent | TouchEvent) => {
      isDragging.current = true;
      const clientX = "touches" in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : (e as MouseEvent).clientY;
      prevPointer.current = { x: clientX, y: clientY };
      rotVel.current = { x: 0, y: 0 };
    };
    const onPointerMove = (e: MouseEvent | TouchEvent) => {
      if (!isDragging.current || !moonRef.current) return;
      const clientX = "touches" in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : (e as MouseEvent).clientY;
      const dx = clientX - prevPointer.current.x;
      const dy = clientY - prevPointer.current.y;
      prevPointer.current = { x: clientX, y: clientY };
      const factor = 0.0055;
      moonRef.current.rotation.y += dx * factor;
      moonRef.current.rotation.x = THREE.MathUtils.clamp(
        moonRef.current.rotation.x + dy * factor,
        -0.9,
        0.9
      );
      rotVel.current = { x: dy * factor * 0.7, y: dx * factor * 0.7 };
    };
    const onPointerUp = () => {
      isDragging.current = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.position.z = THREE.MathUtils.clamp(camera.position.z + e.deltaY * 0.004, 5.0, 9.5);
    };

    const dom = renderer.domElement;
    dom.addEventListener("mousedown", onPointerDown);
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);
    dom.addEventListener("touchstart", onPointerDown, { passive: true });
    window.addEventListener("touchmove", onPointerMove, { passive: true });
    window.addEventListener("touchend", onPointerUp);
    dom.addEventListener("wheel", onWheel, { passive: false });

    const animate = () => {
      if (moonRef.current && !isDragging.current) {
        moonRef.current.rotation.y += rotVel.current.y;
        moonRef.current.rotation.x += rotVel.current.x;
        rotVel.current.x *= 0.94;
        rotVel.current.y = THREE.MathUtils.lerp(
          rotVel.current.y,
          autoRotateRef.current ? 0.0012 : 0,
          0.04
        );
      }
      if (moonRef.current) glowMesh.rotation.copy(moonRef.current.rotation);
      renderer.render(scene, camera);
      animFrameId.current = requestAnimationFrame(animate);
    };
    animFrameId.current = requestAnimationFrame(animate);
    setReady(true);

    return () => {
      disposed = true;
      setReady(false);
      resizeObserver.disconnect();
      window.removeEventListener("resize", handleResize);
      dom.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("mousemove", onPointerMove);
      window.removeEventListener("mouseup", onPointerUp);
      dom.removeEventListener("touchstart", onPointerDown);
      window.removeEventListener("touchmove", onPointerMove);
      window.removeEventListener("touchend", onPointerUp);
      dom.removeEventListener("wheel", onWheel);
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
      renderer.dispose();
      geo.dispose();
      mat.dispose();
      glowGeo.dispose();
      glowMat.dispose();
      starGeo.dispose();
      (stars.material as THREE.Material).dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sun phase control (kept for the hero toolbar; console forces "full").
  useEffect(() => {
    if (!sunLightRef.current) return;
    const light = sunLightRef.current;
    switch (phase) {
      case "new":
        light.position.set(0, 0.5, -6.5);
        light.intensity = 3.0;
        break;
      case "crescent":
        light.position.set(5.2, 0.8, 2.2);
        light.intensity = 2.8;
        break;
      case "quarter":
        light.position.set(6.5, 0.2, 0.2);
        light.intensity = 2.5;
        break;
      case "gibbous":
        light.position.set(4.0, 0.5, 4.8);
        light.intensity = 2.4;
        break;
      case "full":
      default:
        light.position.set(1.0, 0.9, 7.0);
        light.intensity = 2.6;
        break;
    }
  }, [phase, ready]);

  // --- Overlays: graticule + region footprints (true coords) + tie-points.
  useEffect(() => {
    const moon = moonRef.current;
    const overlay = overlayRef.current;
    if (!moon || !overlay || !ready) return;

    // Clear previous overlay children.
    while (overlay.children.length > 0) {
      const child = overlay.children.pop()!;
      child.traverse?.((o: THREE.Object3D) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const m = (mesh as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(m)) m.forEach((x) => x.dispose());
        else if (m) m.dispose();
      });
    }

    const R = MOON_RADIUS;

    // 1. Faint lat/lon graticule every 30deg — sells "true coordinate mapping".
    const gratMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
    });
    const gratPts: THREE.Vector3[] = [];
    for (let lat = -60; lat <= 60; lat += 30) {
      let prev: THREE.Vector3 | null = null;
      for (let lon = 0; lon <= 360; lon += 6) {
        const p = latLonToVec3(lat, lon, R * 1.002);
        if (prev) gratPts.push(prev.clone(), p.clone());
        prev = p;
      }
    }
    for (let lon = 0; lon < 360; lon += 30) {
      let prev: THREE.Vector3 | null = null;
      for (let lat = -90; lat <= 90; lat += 4) {
        const p = latLonToVec3(lat, lon, R * 1.002);
        if (prev) gratPts.push(prev.clone(), p.clone());
        prev = p;
      }
    }
    const gratGeo = new THREE.BufferGeometry().setFromPoints(gratPts);
    overlay.add(new THREE.LineSegments(gratGeo, gratMat));

    // 2. Region footprint quads from real selenographic bounds.
    const faceTargets: THREE.Vector3[] = [];
    (footprints ?? []).forEach((fp) => {
      const active = fp.id === activeFootprintId;
      const colorHex = active ? 0xfbbf24 : 0x22d3ee;
      let w = normalizeLon360(fp.west_lon);
      let e = normalizeLon360(fp.east_lon);
      if (e <= w) e += 360;
      const s = fp.south_lat;
      const n = fp.north_lat;
      const SEG = 20;
      const ring: THREE.Vector3[] = [];
      for (let i = 0; i <= SEG; i++) ring.push(latLonToVec3(n, w + ((e - w) * i) / SEG, R * 1.008));
      for (let i = 1; i <= SEG; i++) ring.push(latLonToVec3(n + ((s - n) * i) / SEG, e, R * 1.008));
      for (let i = 1; i <= SEG; i++) ring.push(latLonToVec3(s, e + ((w - e) * i) / SEG, R * 1.008));
      for (let i = 1; i < SEG; i++) ring.push(latLonToVec3(s + ((n - s) * i) / SEG, w, R * 1.008));

      const lineGeo = new THREE.BufferGeometry().setFromPoints(ring);
      overlay.add(
        new THREE.LineLoop(
          lineGeo,
          new THREE.LineBasicMaterial({
            color: colorHex,
            transparent: true,
            opacity: active ? 1.0 : 0.75,
            depthTest: true,
          })
        )
      );

      // Translucent fill via triangle fan around the footprint centre.
      const center = latLonToVec3((s + n) / 2, (w + e) / 2, R * 1.006);
      const fillVerts: number[] = [];
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % ring.length];
        fillVerts.push(center.x, center.y, center.z, a.x, a.y, a.z, b.x, b.y, b.z);
      }
      const fillGeo = new THREE.BufferGeometry();
      fillGeo.setAttribute("position", new THREE.Float32BufferAttribute(fillVerts, 3));
      overlay.add(
        new THREE.Mesh(
          fillGeo,
          new THREE.MeshBasicMaterial({
            color: colorHex,
            transparent: true,
            opacity: active ? 0.22 : 0.1,
            side: THREE.DoubleSide,
            depthWrite: false,
          })
        )
      );

      // Label sprite at the footprint centre, floating just above the limb.
      const cLon = normalizeLon360((w + e) / 2);
      const cLon180 = cLon > 180 ? cLon - 360 : cLon;
      const label = makeLabelSprite(
        fp.id,
        `${((s + n) / 2).toFixed(2)}°, ${cLon180.toFixed(2)}°`,
        active
      );
      label.position.copy(latLonToVec3((s + n) / 2, (w + e) / 2, R * 1.14));
      overlay.add(label);

      if (active) faceTargets.push(center.clone());
    });

    // 3. Tie-point markers (true lat/lon from /moon-points).
    let sx = 0;
    let sy = 0;
    let sz = 0;
    let nValid = 0;
    (tiePoints ?? []).forEach((pt) => {
      if (pt.latitude === null || pt.latitude === undefined) return;
      if (pt.longitude === null || pt.longitude === undefined) return;
      const p = latLonToVec3(pt.latitude, pt.longitude, R * 1.015);
      sx += p.x;
      sy += p.y;
      sz += p.z;
      nValid++;
      const conf = pt.confidence ?? 0.9;
      const colorHex = conf >= 0.8 ? 0x10b981 : conf >= 0.5 ? 0xf59e0b : 0xf43f5e;
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.032, 16, 16),
        new THREE.MeshBasicMaterial({ color: colorHex })
      );
      marker.position.copy(p);
      overlay.add(marker);
      const ringMesh = new THREE.Mesh(
        new THREE.RingGeometry(0.05, 0.075, 24),
        new THREE.MeshBasicMaterial({
          color: colorHex,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
          depthWrite: false,
        })
      );
      ringMesh.position.copy(p);
      ringMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), p.clone().normalize());
      overlay.add(ringMesh);
    });
    if (nValid > 0) faceTargets.push(new THREE.Vector3(sx, sy, sz));

    // Face the active region / tie-point centroid toward the camera.
    if (faceTargets.length > 0 && moonRef.current) {
      const c = faceTargets.reduce((acc, v) => acc.add(v), new THREE.Vector3());
      const ang = Math.atan2(c.x, c.z);
      moonRef.current.rotation.y = -ang;
      moonRef.current.rotation.x = THREE.MathUtils.clamp(
        -Math.asin(THREE.MathUtils.clamp(c.clone().normalize().y, -1, 1)) * 0.6,
        -0.5,
        0.5
      );
      rotVel.current = { x: 0, y: autoRotateRef.current ? 0.0006 : 0 };
    }
  }, [tiePoints, footprints, activeFootprintId, ready]);

  return (
    <div className="relative h-full w-full">
      <div
        ref={containerRef}
        className={className ?? "absolute inset-0 z-10 h-full w-full cursor-grab active:cursor-grabbing"}
      />
      <div className="pointer-events-none absolute right-3 top-3 z-20 rounded-full border border-white/10 bg-slate-950/80 px-2.5 py-1 font-mono text-[10px] text-slate-300 backdrop-blur">
        {texSource === "lroc" && <span>● Real Moon · NASA LROC (public CDN)</span>}
        {texSource === "loading" && <span className="animate-pulse">○ Loading real Moon texture…</span>}
        {texSource === "fallback" && <span>● Procedural fallback (offline)</span>}
      </div>
    </div>
  );
}
