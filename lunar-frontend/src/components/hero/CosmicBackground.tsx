"use client";

import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  radius: number;
  alpha: number;
  baseAlpha: number;
  twinkleSpeed: number;
  color: string;
}

interface ShootingStar {
  x: number;
  y: number;
  length: number;
  speed: number;
  angle: number;
  alpha: number;
  fadeSpeed: number;
  thickness: number;
}

export default function CosmicBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      initStars();
    };

    window.addEventListener("resize", handleResize);

    // Color palette for stars
    const starColors = ["#ffffff", "#e0e8ff", "#b9d1ff", "#ffeacc", "#a3cbff"];

    let stars: Star[] = [];
    const initStars = () => {
      stars = [];
      const count = Math.floor((width * height) / 2400);
      for (let i = 0; i < count; i++) {
        const baseAlpha = 0.2 + Math.random() * 0.7;
        stars.push({
          x: Math.random() * width,
          y: Math.random() * height,
          radius: Math.random() < 0.85 ? Math.random() * 1.1 + 0.3 : Math.random() * 1.8 + 1.2,
          alpha: baseAlpha,
          baseAlpha,
          twinkleSpeed: 0.008 + Math.random() * 0.025,
          color: starColors[Math.floor(Math.random() * starColors.length)],
        });
      }
    };
    initStars();

    // Shooting stars
    const shootingStars: ShootingStar[] = [];
    let lastSpawn = Date.now();
    const spawnShootingStar = () => {
      const angle = (Math.PI / 180) * (30 + Math.random() * 25); // ~35-55 deg downward right
      shootingStars.push({
        x: Math.random() * width * 0.8,
        y: Math.random() * height * 0.4,
        length: 80 + Math.random() * 140,
        speed: 12 + Math.random() * 16,
        angle,
        alpha: 1.0,
        fadeSpeed: 0.015 + Math.random() * 0.02,
        thickness: 1.2 + Math.random() * 1.5,
      });
    };

    let tick = 0;

    const render = () => {
      tick++;

      // Background gradient
      const bgGrad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.45,
        100,
        width * 0.5,
        height * 0.5,
        Math.max(width, height)
      );
      bgGrad.addColorStop(0, "#080e1c");
      bgGrad.addColorStop(0.35, "#040710");
      bgGrad.addColorStop(0.7, "#020308");
      bgGrad.addColorStop(1, "#010205");

      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // Subtle cyan/blue celestial dust cloud
      const dustGrad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.65,
        50,
        width * 0.5,
        height * 0.65,
        width * 0.55
      );
      dustGrad.addColorStop(0, "rgba(28, 64, 110, 0.12)");
      dustGrad.addColorStop(0.4, "rgba(14, 38, 70, 0.06)");
      dustGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = dustGrad;
      ctx.fillRect(0, 0, width, height);

      // Draw and twinkle stars
      for (const s of stars) {
        s.alpha = s.baseAlpha + Math.sin(tick * s.twinkleSpeed) * 0.25;
        ctx.fillStyle = s.color;
        ctx.globalAlpha = Math.max(0.1, Math.min(1, s.alpha));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
        ctx.fill();

        // Extra glow for brighter stars
        if (s.radius > 1.4) {
          ctx.globalAlpha = s.alpha * 0.3;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.radius * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Spawn shooting stars periodically
      const now = Date.now();
      if (now - lastSpawn > 3200 && Math.random() < 0.035) {
        spawnShootingStar();
        lastSpawn = now;
      }

      // Draw shooting stars
      for (let i = shootingStars.length - 1; i >= 0; i--) {
        const ss = shootingStars[i];
        ss.x += Math.cos(ss.angle) * ss.speed;
        ss.y += Math.sin(ss.angle) * ss.speed;
        ss.alpha -= ss.fadeSpeed;

        if (ss.alpha <= 0 || ss.x > width + 200 || ss.y > height + 200) {
          shootingStars.splice(i, 1);
          continue;
        }

        const tailX = ss.x - Math.cos(ss.angle) * ss.length;
        const tailY = ss.y - Math.sin(ss.angle) * ss.length;

        const grad = ctx.createLinearGradient(tailX, tailY, ss.x, ss.y);
        grad.addColorStop(0, "rgba(255, 255, 255, 0)");
        grad.addColorStop(0.7, `rgba(180, 220, 255, ${ss.alpha * 0.6})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${ss.alpha})`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = ss.thickness;
        ctx.lineCap = "round";
        ctx.globalAlpha = 1;

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(ss.x, ss.y);
        ctx.stroke();

        // Glowing head
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = ss.alpha;
        ctx.beginPath();
        ctx.arc(ss.x, ss.y, ss.thickness * 1.2, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = 1;
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-0 h-full w-full"
    />
  );
}
