"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export type PayloadMode = "optical" | "iirs" | "dem";
export type LunarPhase = "crescent" | "quarter" | "gibbous" | "full" | "new";

interface Props {
  payloadMode?: PayloadMode;
  phase?: LunarPhase;
  autoRotate?: boolean;
}

// Generate realistic procedural lunar textures via HTML5 Canvas
function createLunarTextures(mode: PayloadMode): {
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

  // 1. Base Regolith tone
  if (mode === "optical") {
    ctx.fillStyle = "#8a8f98";
  } else if (mode === "iirs") {
    ctx.fillStyle = "#1e2a3a";
  } else {
    ctx.fillStyle = "#18453b";
  }
  ctx.fillRect(0, 0, width, height);

  bCtx.fillStyle = "#808080";
  bCtx.fillRect(0, 0, width, height);

  // 2. Maria (dark basaltic volcanic plains)
  const maria = [
    { x: width * 0.42, y: height * 0.38, rx: 280, ry: 190, tone: "#42454b", bTone: "#555" }, // Oceanus Procellarum
    { x: width * 0.56, y: height * 0.35, rx: 160, ry: 120, tone: "#3d4046", bTone: "#505050" }, // Mare Imbrium
    { x: width * 0.68, y: height * 0.42, rx: 130, ry: 100, tone: "#3b3e44", bTone: "#4d4d4d" }, // Mare Serenitatis
    { x: width * 0.74, y: height * 0.48, rx: 120, ry: 90, tone: "#36393e", bTone: "#484848" }, // Mare Tranquillitatis
    { x: width * 0.82, y: height * 0.43, rx: 75, ry: 60, tone: "#33353b", bTone: "#454545" }, // Mare Crisium
    { x: width * 0.65, y: height * 0.76, rx: 220, ry: 140, tone: "#2e3137", bTone: "#3e3e3e" }, // South Pole-Aitken Basin
  ];

  maria.forEach((m) => {
    // Color
    const grad = ctx.createRadialGradient(m.x, m.y, 10, m.x, m.y, m.rx);
    if (mode === "optical") {
      grad.addColorStop(0, m.tone);
      grad.addColorStop(0.7, m.tone);
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
    } else if (mode === "iirs") {
      // Hyperspectral mineral absorption signatures
      grad.addColorStop(0, "#2c4060");
      grad.addColorStop(0.5, "#d68a2d");
      grad.addColorStop(0.8, "#502868");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
    } else {
      // DEM elevation hypsometric low
      grad.addColorStop(0, "#082042");
      grad.addColorStop(0.6, "#144565");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
    }

    ctx.beginPath();
    ctx.ellipse(m.x, m.y, m.rx, m.ry, 0, 0, Math.PI * 2);
    ctx.fill();

    // Bump
    const bGrad = bCtx.createRadialGradient(m.x, m.y, 10, m.x, m.y, m.rx);
    bGrad.addColorStop(0, m.bTone);
    bGrad.addColorStop(1, "#808080");
    bCtx.fillStyle = bGrad;
    bCtx.beginPath();
    bCtx.ellipse(m.x, m.y, m.rx, m.ry, 0, 0, Math.PI * 2);
    bCtx.fill();
  });

  // 3. Impact craters & radiating ejecta rays
  const prominentCraters = [
    { x: width * 0.52, y: height * 0.74, r: 45, rays: 18 }, // Tycho
    { x: width * 0.50, y: height * 0.45, r: 40, rays: 12 }, // Copernicus
    { x: width * 0.44, y: height * 0.48, r: 25, rays: 8 },  // Kepler
    { x: width * 0.62, y: height * 0.30, r: 35, rays: 6 },  // Plato
    { x: width * 0.32, y: height * 0.35, r: 30, rays: 4 },
    { x: width * 0.78, y: height * 0.65, r: 38, rays: 8 },
    { x: width * 0.22, y: height * 0.62, r: 28, rays: 5 },
    { x: width * 0.15, y: height * 0.40, r: 32, rays: 6 },
    { x: width * 0.88, y: height * 0.28, r: 26, rays: 4 },
  ];

  prominentCraters.forEach((c) => {
    // Ejecta rays
    if (c.rays > 0) {
      ctx.save();
      ctx.translate(c.x, c.y);
      for (let i = 0; i < c.rays; i++) {
        const ang = (Math.PI * 2 * i) / c.rays + Math.sin(i * 3) * 0.2;
        const rayLen = c.r * (4 + Math.cos(i) * 2.5);
        const rayGrad = ctx.createLinearGradient(0, 0, Math.cos(ang) * rayLen, Math.sin(ang) * rayLen);
        if (mode === "optical") {
          rayGrad.addColorStop(0, "rgba(235, 240, 248, 0.5)");
          rayGrad.addColorStop(1, "rgba(235, 240, 248, 0)");
        } else if (mode === "iirs") {
          rayGrad.addColorStop(0, "rgba(80, 240, 220, 0.4)");
          rayGrad.addColorStop(1, "rgba(80, 240, 220, 0)");
        } else {
          rayGrad.addColorStop(0, "rgba(255, 220, 100, 0.4)");
          rayGrad.addColorStop(1, "rgba(255, 220, 100, 0)");
        }
        ctx.strokeStyle = rayGrad;
        ctx.lineWidth = 2 + Math.sin(i) * 1.5;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(ang) * rayLen, Math.sin(ang) * rayLen);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Subtle highland micro-craters
    const cGrad = ctx.createRadialGradient(c.x, c.y, c.r * 0.2, c.x, c.y, c.r);
    if (mode === "optical") {
      cGrad.addColorStop(0, "#2c2f35");
      cGrad.addColorStop(0.7, "#4a4f56");
      cGrad.addColorStop(0.85, "#c5cbd4"); // bright rim crest
      cGrad.addColorStop(1, "transparent");
    } else if (mode === "iirs") {
      cGrad.addColorStop(0, "#162030");
      cGrad.addColorStop(0.7, "#4287f5");
      cGrad.addColorStop(0.85, "#f0a53a");
      cGrad.addColorStop(1, "transparent");
    } else {
      cGrad.addColorStop(0, "#1c4a5e");
      cGrad.addColorStop(0.8, "#56ab2f");
      cGrad.addColorStop(0.9, "#e74c3c");
      cGrad.addColorStop(1, "transparent");
    }
    ctx.fillStyle = cGrad;
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.fill();

    // Central peak
    ctx.fillStyle = mode === "optical" ? "#f4f7fb" : "#fff";
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r * 0.12, 0, Math.PI * 2);
    ctx.fill();

    // Bump crater depression & rim crest
    const bCrater = bCtx.createRadialGradient(c.x, c.y, c.r * 0.1, c.x, c.y, c.r);
    bCrater.addColorStop(0, "#202020");
    bCrater.addColorStop(0.7, "#484848");
    bCrater.addColorStop(0.88, "#ffffff"); // raised rim crest
    bCrater.addColorStop(1, "#808080");
    bCtx.fillStyle = bCrater;
    bCtx.beginPath();
    bCtx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    bCtx.fill();
  });

  // 4. Subtle regolith noise & natural micro-craters
  const prng = (seed: number) => {
    let s = seed;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  };
  const rand = prng(1337);

  // Micro-crater depressions
  for (let i = 0; i < 350; i++) {
    const rx = rand() * width;
    const ry = rand() * height;
    const r = rand() * 8 + 1.5;

    const microGrad = ctx.createRadialGradient(rx, ry, r * 0.2, rx, ry, r);
    microGrad.addColorStop(0, mode === "optical" ? "rgba(40,44,50,0.5)" : "rgba(20,30,45,0.5)");
    microGrad.addColorStop(0.7, mode === "optical" ? "rgba(60,65,72,0.3)" : "rgba(30,45,65,0.3)");
    microGrad.addColorStop(0.9, mode === "optical" ? "rgba(210,218,230,0.4)" : "rgba(240,180,80,0.4)");
    microGrad.addColorStop(1, "transparent");

    ctx.fillStyle = microGrad;
    ctx.beginPath();
    ctx.arc(rx, ry, r, 0, Math.PI * 2);
    ctx.fill();

    const bMicro = bCtx.createRadialGradient(rx, ry, r * 0.15, rx, ry, r);
    bMicro.addColorStop(0, "#303030");
    bMicro.addColorStop(0.8, "#606060");
    bMicro.addColorStop(0.92, "#d0d0d0");
    bMicro.addColorStop(1, "#808080");
    bCtx.fillStyle = bMicro;
    bCtx.beginPath();
    bCtx.arc(rx, ry, r, 0, Math.PI * 2);
    bCtx.fill();
  }

  // Textures
  const colorMap = new THREE.CanvasTexture(colorCanvas);
  colorMap.wrapS = THREE.RepeatWrapping;
  colorMap.wrapT = THREE.ClampToEdgeWrapping;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = THREE.RepeatWrapping;
  bumpMap.wrapT = THREE.ClampToEdgeWrapping;

  return { colorMap, bumpMap };
}

export default function LunarGlobe({
  payloadMode = "optical",
  phase = "crescent",
  autoRotate = true,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const moonRef = useRef<THREE.Mesh | null>(null);
  const sunLightRef = useRef<THREE.DirectionalLight | null>(null);
  const animFrameId = useRef<number | null>(null);

  // Rotation state
  const isDragging = useRef(false);
  const prevPointer = useRef({ x: 0, y: 0 });
  const rotVel = useRef({ x: 0, y: 0.0012 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 1000);
    camera.position.set(0, -0.2, 7.2);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting (Directional Sun)
    const sunLight = new THREE.DirectionalLight(0xfff7ed, 2.6);
    sunLight.position.set(5.5, 1.2, 3.5);
    scene.add(sunLight);
    sunLightRef.current = sunLight;

    // Soft celestial fill light from Earth-shine
    const earthShine = new THREE.DirectionalLight(0x28486e, 0.45);
    earthShine.position.set(-6, -2, -2);
    scene.add(earthShine);

    // Ambient space light
    const ambient = new THREE.AmbientLight(0x0a1120, 0.28);
    scene.add(ambient);

    // Create Moon Mesh
    const geo = new THREE.SphereGeometry(2.35, 96, 96);
    const { colorMap, bumpMap } = createLunarTextures(payloadMode);

    const mat = new THREE.MeshStandardMaterial({
      map: colorMap,
      bumpMap: bumpMap,
      bumpScale: 0.038,
      roughness: 0.92,
      metalness: 0.05,
    });

    const moon = new THREE.Mesh(geo, mat);
    moon.position.set(0, -0.65, 0); // slightly lowered like in the Pinterest pin
    moon.rotation.y = 0.85;
    scene.add(moon);
    moonRef.current = moon;

    // Atmosphere / Lunar Rim Glow (Ethereal blue Fresnel shader)
    const glowGeo = new THREE.SphereGeometry(2.39, 64, 64);
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
          float fresnel = dot(vNormal, viewDir);
          fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
          float intensity = pow(fresnel, 3.2);
          vec3 glowColor = vec3(0.32, 0.58, 0.95); // Deep celestial blue
          gl_FragColor = vec4(glowColor, intensity * 0.72);
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

    // Resize handler
    const handleResize = () => {
      if (!container || !renderer) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Mouse / Pointer Drag Interactions
    const onPointerDown = (e: MouseEvent | TouchEvent) => {
      isDragging.current = true;
      const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
      prevPointer.current = { x: clientX, y: clientY };
      rotVel.current = { x: 0, y: 0 };
    };

    const onPointerMove = (e: MouseEvent | TouchEvent) => {
      if (!isDragging.current || !moonRef.current) return;
      const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

      const dx = clientX - prevPointer.current.x;
      const dy = clientY - prevPointer.current.y;

      prevPointer.current = { x: clientX, y: clientY };

      const factor = 0.0055;
      moonRef.current.rotation.y += dx * factor;
      moonRef.current.rotation.x += dy * factor;

      rotVel.current = { x: dy * factor * 0.7, y: dx * factor * 0.7 };
    };

    const onPointerUp = () => {
      isDragging.current = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.position.z = THREE.MathUtils.clamp(
        camera.position.z + e.deltaY * 0.004,
        5.2,
        9.0
      );
    };

    const dom = renderer.domElement;
    dom.addEventListener("mousedown", onPointerDown);
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);

    dom.addEventListener("touchstart", onPointerDown, { passive: true });
    window.addEventListener("touchmove", onPointerMove, { passive: true });
    window.addEventListener("touchend", onPointerUp);
    dom.addEventListener("wheel", onWheel, { passive: false });

    // Animation Loop
    const animate = () => {
      if (moonRef.current) {
        if (!isDragging.current) {
          // Apply residual momentum & damping
          moonRef.current.rotation.y += rotVel.current.y;
          moonRef.current.rotation.x += rotVel.current.x;

          rotVel.current.x *= 0.94;
          rotVel.current.y = THREE.MathUtils.lerp(
            rotVel.current.y,
            autoRotate ? 0.0015 : 0,
            0.04
          );
        }
        glowMesh.rotation.copy(moonRef.current.rotation);
      }

      renderer.render(scene, camera);
      animFrameId.current = requestAnimationFrame(animate);
    };

    animFrameId.current = requestAnimationFrame(animate);

    return () => {
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
    };
  }, [payloadMode, autoRotate]);

  // Handle phase changes (Sun angle lighting direction)
  useEffect(() => {
    if (!sunLightRef.current) return;
    const light = sunLightRef.current;
    switch (phase) {
      case "new":
        light.position.set(0, 0.5, -6.5);
        light.intensity = 3.2;
        break;
      case "crescent":
        // Classic crescent with dramatic rim shadow (matching Pinterest pin)
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
        light.position.set(0.5, 0.5, 7.0);
        light.intensity = 2.2;
        break;
    }
  }, [phase]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-10 h-full w-full cursor-grab active:cursor-grabbing"
    />
  );
}
