// src/primitives.tsx — reusable UI primitives
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { PricePoint, Tone } from "./types";
import { fmtPrice } from "./format";

interface SparklineProps { data: PricePoint[]; up: boolean; width?: number | string; height?: number; fillOpacity?: number; }
export const Sparkline = ({ data, up, width = 80, height = 24, fillOpacity = 0.18 }: SparklineProps) => {
  if (!data || data.length === 0) return null;
  const W = typeof width === "number" ? width : 100;
  const vs = data.map(d => d.value);
  const min = Math.min(...vs), max = Math.max(...vs);
  const range = max - min || 1;
  const stepX = W / (data.length - 1);
  const pts = data.map((d, i) => [i * stepX, height - ((d.value - min) / range) * (height - 2) - 1]);
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const fill = `${path} L${W},${height} L0,${height} Z`;
  const color = up ? "var(--green)" : "var(--red)";
  const gid = `spk-${Math.random().toString(36).slice(2, 9)}`;
  return (
    <svg width={width as any} height={height} viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" style={{ display: "block" }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fill} fill={`url(#${gid})`} />
      <path d={path} stroke={color} strokeWidth="1.25" fill="none" />
    </svg>
  );
};

interface LineChartProps { data: PricePoint[]; up: boolean; height?: number; showGrid?: boolean; fillOpacity?: number; showVolume?: boolean; }
export const LineChart = ({ data, up, height = 280, showGrid = true, fillOpacity = 0.22, showVolume = false }: LineChartProps) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [hover, setHover] = useState<{ i: number; x: number; y: number; v: number } | null>(null);

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setW(e.contentRect.width);
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  if (!data || data.length === 0) return null;
  const padL = 56, padR = 12, padT = 10, padB = showVolume ? 60 : 24;
  const chartH = height - padT - padB;
  const chartW = Math.max(1, w - padL - padR);
  const vs = data.map(d => d.value);
  const min = Math.min(...vs), max = Math.max(...vs);
  const range = (max - min) || 1;
  const stepX = chartW / (data.length - 1);
  const xy = data.map((d, i) => [padL + i * stepX, padT + chartH - ((d.value - min) / range) * chartH] as [number, number]);
  const path = xy.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const fill = `${path} L${padL + chartW},${padT + chartH} L${padL},${padT + chartH} Z`;
  const color = up ? "var(--green)" : "var(--red)";
  const gid = `chart-${Math.random().toString(36).slice(2, 9)}`;

  const yTicks = 5;
  const ticks = Array.from({ length: yTicks }, (_, i) => {
    const v = min + (range * (yTicks - 1 - i)) / (yTicks - 1);
    const y = padT + (chartH * i) / (yTicks - 1);
    return { v, y };
  });

  const xTicks = 6;
  const xTicksArr = Array.from({ length: xTicks }, (_, i) => {
    const idx = Math.floor((data.length - 1) * (i / (xTicks - 1)));
    return { idx, x: padL + idx * stepX, label: data[idx].t };
  });

  const handleMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const i = Math.round((x - padL) / stepX);
    if (i >= 0 && i < data.length) setHover({ i, x: xy[i][0], y: xy[i][1], v: data[i].value });
  };

  const vols = data.map((d, i) => {
    const seed = (d.value * 13 + i * 7) % 100;
    return 0.3 + (seed / 100) * 0.7;
  });

  return (
    <div ref={wrapRef} style={{ width: "100%", position: "relative" }}>
      <svg width={w} height={height} onMouseMove={handleMove} onMouseLeave={() => setHover(null)} style={{ display: "block", cursor: "crosshair" }}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        {showGrid && ticks.map((t, i) => (
          <line key={i} x1={padL} x2={padL + chartW} y1={t.y} y2={t.y} stroke="var(--border)" strokeDasharray="2 4" strokeWidth="1" />
        ))}
        {showGrid && ticks.map((t, i) => (
          <text key={`yl-${i}`} x={padL - 8} y={t.y + 3} fill="var(--text-muted)" fontSize="10" textAnchor="end" fontFamily="var(--mono)">{fmtPrice(t.v)}</text>
        ))}
        {xTicksArr.map((t, i) => (
          <text key={`xl-${i}`} x={t.x} y={padT + chartH + 14} fill="var(--text-muted)" fontSize="10" textAnchor="middle" fontFamily="var(--mono)">{t.label}</text>
        ))}
        <path d={fill} fill={`url(#${gid})`} />
        <path d={path} stroke={color} strokeWidth="1.5" fill="none" />
        {showVolume && data.map((_d, i) => {
          const h = vols[i] * 28;
          return (<rect key={i} x={padL + i * stepX - stepX * 0.35} y={height - padB + 24 - h} width={stepX * 0.7} height={h} fill={color} opacity="0.35" />);
        })}
        {hover && (
          <g>
            <line x1={hover.x} x2={hover.x} y1={padT} y2={padT + chartH} stroke="var(--text-secondary)" strokeDasharray="3 3" strokeWidth="1" />
            <line x1={padL} x2={padL + chartW} y1={hover.y} y2={hover.y} stroke="var(--text-secondary)" strokeDasharray="3 3" strokeWidth="1" />
            <circle cx={hover.x} cy={hover.y} r="4" fill={color} stroke="var(--bg2)" strokeWidth="2" />
          </g>
        )}
      </svg>
      {hover && (
        <div style={{
          position: "absolute", left: Math.min(hover.x + 12, w - 140), top: Math.max(hover.y - 36, 8),
          background: "var(--bg3)", border: "1px solid var(--border-bright)", padding: "6px 10px",
          fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-primary)", pointerEvents: "none", whiteSpace: "nowrap",
        }}>
          <div style={{ color: "var(--text-muted)", fontSize: 9, marginBottom: 2 }}>{data[hover.i].t}</div>
          ${fmtPrice(hover.v)}
        </div>
      )}
    </div>
  );
};

interface DonutSlice { label: string; value: number; color: string; }
export const Donut = ({ slices, size = 200, thickness = 26 }: { slices: DonutSlice[]; size?: number; thickness?: number }) => {
  const total = slices.reduce((a, s) => a + s.value, 0);
  const R = size / 2 - 2;
  const r = R - thickness;
  const cx = size / 2, cy = size / 2;
  let angle = -Math.PI / 2;
  const arcs = slices.map((s, i) => {
    const a = (s.value / total) * Math.PI * 2;
    const a0 = angle, a1 = angle + a;
    angle = a1;
    const large = a > Math.PI ? 1 : 0;
    const x0 = cx + R * Math.cos(a0), y0 = cy + R * Math.sin(a0);
    const x1 = cx + R * Math.cos(a1), y1 = cy + R * Math.sin(a1);
    const x2 = cx + r * Math.cos(a1), y2 = cy + r * Math.sin(a1);
    const x3 = cx + r * Math.cos(a0), y3 = cy + r * Math.sin(a0);
    const d = `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3} Z`;
    return { d, color: s.color, key: i };
  });
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {arcs.map(a => <path key={a.key} d={a.d} fill={a.color} />)}
    </svg>
  );
};

export const Pill = ({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) => {
  const tones: Record<Tone, { bg: string; bd: string; fg: string }> = {
    up:      { bg: "rgba(0,212,106,0.13)",  bd: "rgba(0,212,106,0.4)",  fg: "var(--green)" },
    down:    { bg: "rgba(255,68,102,0.13)", bd: "rgba(255,68,102,0.4)", fg: "var(--red)" },
    neutral: { bg: "rgba(255,255,255,0.04)", bd: "var(--border)",       fg: "var(--text-secondary)" },
    warn:    { bg: "rgba(255,204,68,0.10)",  bd: "rgba(255,204,68,0.4)", fg: "var(--yellow)" },
  };
  const t = tones[tone];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: t.bg, border: `1px solid ${t.bd}`, color: t.fg,
      padding: "2px 8px", fontSize: 10, fontFamily: "var(--mono)",
      letterSpacing: 0.5, textTransform: "uppercase",
    }}>{children}</span>
  );
};
