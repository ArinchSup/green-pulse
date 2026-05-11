// src/primitives.tsx — reusable UI primitives
import React, { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { PricePoint, Tone } from "./types";
import { fmtPrice } from "./format";

interface LineChartProps { 
  data: PricePoint[]; 
  up: boolean; 
  height?: number; 
  showGrid?: boolean; 
  fillOpacity?: number; 
  showVolume?: boolean;
  fibLevels?: { label: string; value: number; color?: string }[];
  srLevels?: { label: string; value: number; color: string }[];
  demandZone?: [number, number] | null;
  focusLength?: number; 
  focusStartT?: string | number;
}

export const LineChart = ({ 
  data, up, height = 200, showGrid = true, fillOpacity = 0.2, showVolume = false,
  fibLevels, srLevels, demandZone, focusLength, focusStartT
}: LineChartProps) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [hover, setHover] = useState<{ i: number; x: number; y: number; v: number; t: string } | null>(null);

  const [viewX, setViewX] = useState({ s: 0, e: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef({ x: 0 });

  useEffect(() => {
    if (!data || !Array.isArray(data) || data.length < 2) return;

    const len = data.length;
    let startIdx = 0;
    
    if (focusStartT) {
      // 🌟 1. แปลงเวลาเป็น Timestamp เพื่อความแม่นยำระดับวินาที (แก้บัค 1M, 1Y ซูมผิดจุด)
      const targetTime = new Date(String(focusStartT).replace(" ", "T")).getTime();
      if (!isNaN(targetTime)) {
        const foundIdx = data.findIndex(d => new Date(String(d.t).replace(" ", "T")).getTime() >= targetTime);
        if (foundIdx !== -1) startIdx = foundIdx;
      }
    } else if (focusLength) {
      startIdx = Math.max(0, len - focusLength);
    }

    startIdx = Math.max(0, Math.min(startIdx, len - 2));
    setViewX({ s: startIdx, e: len - 1 + (len - startIdx) * 0.05 });
  }, [data, focusStartT, focusLength]);

  useEffect(() => {
    if (!wrapRef.current) return;
    const obs = new ResizeObserver(es => { if (es[0]) setW(es[0].contentRect.width); });
    obs.observe(wrapRef.current);
    const el = wrapRef.current;
    const preventScroll = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", preventScroll, { passive: false });
    return () => { obs.disconnect(); el.removeEventListener("wheel", preventScroll); };
  }, []);

  if (!data || !Array.isArray(data) || data.length < 2) return null;

  const padL = 10, padR = 60, padT = 20, padB = showVolume ? 40 : 20;
  const chartW = Math.max(1, w - padL - padR);
  const chartH = Math.max(1, height - padT - padB);

  const spanX = Math.max(0.0001, viewX.e - viewX.s);
  const stepX = chartW / spanX;

  const sIdx = Math.max(0, Math.floor(viewX.s));
  const eIdx = Math.min(data.length, Math.ceil(viewX.e) + 1);
  const visibleData = data.slice(sIdx, eIdx);

  const vs = visibleData.map(d => d.value || 0).filter(v => v > 0);
  let dataMin = 0, dataMax = 100;
  if (vs.length > 0) {
    dataMin = Math.min(...vs);
    dataMax = Math.max(...vs);
    if (dataMin === dataMax) { dataMin -= 1; dataMax += 1; } 
  }
  
  const paddingY = (dataMax - dataMin) * 0.1 || 1; 
  const viewYMin = dataMin - paddingY;
  const viewYMax = dataMax + paddingY;
  const rangeY = Math.max(0.0001, viewYMax - viewYMin);

  const getY = (val: number) => padT + chartH - (((val || 0) - viewYMin) / rangeY) * chartH;

  const pts = visibleData.map((d, i) => [padL + (sIdx + i - viewX.s) * stepX, getY(d.value)]);
  const path = pts.length > 0 ? pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") : "";
  const fill = pts.length > 0 ? `${path} L${pts[pts.length - 1][0]},${height - padB} L${pts[0][0]},${height - padB} Z` : "";
  const color = up ? "var(--green)" : "var(--red)";
  const gid = `grad-${color.replace(/[^\w]/g, "")}`;

  const yTicks = 5;
  const ticks = Array.from({ length: yTicks }, (_, i) => ({
    v: viewYMin + (rangeY * (yTicks - 1 - i)) / (yTicks - 1),
    y: padT + (chartH * i) / (yTicks - 1)
  }));

  const xTicks = 6;
  const xTicksArr = Array.from({ length: xTicks }, (_, i) => {
    const exactIndex = viewX.s + spanX * (i / (xTicks - 1));
    const idx = Math.floor(exactIndex);
    if (idx < 0 || idx >= data.length || !data[idx]) return null;

    let labelStr = String(data[idx].t || "");
    if (labelStr.includes("-")) {
      const parts = labelStr.split(" ")[0].split("-");
      if (parts.length === 3) {
        // 🌟 2. จัดวันที่เป็นรูปแบบ MM-DD-YY (ใส่เลขปี 2 หลักท้าย)
        labelStr = `${parts[1]}-${parts[2]}-${parts[0].substring(2)}`;
      }
    }
    return { x: padL + (exactIndex - viewX.s) * stepX, label: labelStr };
  }).filter(Boolean) as { x: number; label: string | number }[];

  const handleWheel = (e: React.WheelEvent) => {
    const zoomFactor = e.deltaY > 0 ? 1.15 : 0.85;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = Math.max(0, Math.min(1, (e.clientX - rect.left - padL) / chartW));
    
    let newSpanX = Math.max(5, spanX * zoomFactor); 
    newSpanX = Math.min(data.length - 1, newSpanX); // 🌟 ล็อกไม่ให้ Zoom Out ทะลุข้อมูลอดีต
    
    let newS = viewX.s + (spanX - newSpanX) * mouseX;
    let newE = newS + newSpanX;

    if (newS < 0) { newS = 0; newE = Math.min(data.length - 1, newSpanX); }
    if (newE > data.length - 1) { newE = data.length - 1; newS = Math.max(0, newE - newSpanX); }

    setViewX({ s: newS, e: newE });
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isDragging) {
      const dx = e.clientX - dragRef.current.x;
      const shiftX = -dx * (spanX / chartW);
      
      setViewX(prev => {
        let nS = prev.s + shiftX;
        let nE = prev.e + shiftX;
        if (nS < 0) { nS = 0; nE = Math.min(data.length - 1, spanX); }
        if (nE > data.length - 1) { nE = data.length - 1; nS = Math.max(0, nE - spanX); }
        return { s: nS, e: nE };
      });
      dragRef.current = { x: e.clientX }; 
    } else {
      const rect = e.currentTarget.getBoundingClientRect();
      const exactIndex = (e.clientX - rect.left - padL) / stepX + viewX.s;
      const i = Math.round(exactIndex);
      if (i >= 0 && i < data.length && data[i]) {
        // 🌟 จัดฟอร์แมตเวลาให้สวยงามก่อนแสดง
        let timeStr = String(data[i].t || "");
        if (timeStr.includes("-")) {
          const parts = timeStr.split(" ")[0].split("-");
          if (parts.length === 3) {
            const dStr = `${parts[1]}-${parts[2]}-${parts[0].substring(2)}`;
            const tStr = timeStr.includes(" ") ? " " + timeStr.split(" ")[1].substring(0, 5) : "";
            timeStr = dStr + tStr;
          }
        }
        setHover({ i, x: padL + (i - viewX.s) * stepX, y: getY(data[i].value), v: data[i].value, t: timeStr });
      } else setHover(null);
    }
  };

  const handleDoubleClick = () => {
    const len = data.length;
    let startIdx = 0;
    if (focusStartT) {
      const targetTime = new Date(String(focusStartT).replace(" ", "T")).getTime();
      if (!isNaN(targetTime)) {
        const foundIdx = data.findIndex(d => new Date(String(d.t).replace(" ", "T")).getTime() >= targetTime);
        if (foundIdx !== -1) startIdx = foundIdx;
      }
    }
    startIdx = Math.max(0, Math.min(startIdx, len - 2));
    setViewX({ s: startIdx, e: len - 1 + (len - startIdx) * 0.05 });
  };

  return (
    <div ref={wrapRef} style={{ width: "100%", position: "relative" }}>
      <svg 
        width={w} height={height} 
        onWheel={handleWheel} 
        onMouseDown={(e) => { setIsDragging(true); dragRef.current = { x: e.clientX }; }} 
        onMouseMove={handleMouseMove} 
        onMouseUp={() => setIsDragging(false)} 
        onMouseLeave={() => { setIsDragging(false); setHover(null); }}
        onDoubleClick={handleDoubleClick}
        style={{ display: "block", cursor: isDragging ? "grabbing" : "crosshair", touchAction: "none" }}
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>

        {showGrid && ticks.map((t, i) => (
          <g key={`y-${i}`}>
            <line x1={padL} x2={padL + chartW} y1={t.y} y2={t.y} stroke="var(--border)" strokeDasharray="2 4" strokeWidth="1" />
            <text x={padL + chartW + 8} y={t.y + 3} fill="var(--text-muted)" fontSize="10" textAnchor="start" fontFamily="var(--mono)">{fmtPrice(t.v)}</text>
          </g>
        ))}
        {showGrid && xTicksArr.map((t, i) => (
          <text key={`x-${i}`} x={t.x} y={padT + chartH + 14} fill="var(--text-muted)" fontSize="10" textAnchor="middle" fontFamily="var(--mono)">{t.label}</text>
        ))}

        {demandZone && demandZone.length === 2 && (() => {
          const yTop = getY(demandZone[1]), yBot = getY(demandZone[0]);
          const h = Math.abs(yBot - yTop) || 1;
          return (
            <g>
              <rect x={padL} y={Math.min(yTop, yBot)} width={chartW} height={h} fill="#b088f5" opacity={0.15} stroke="#b088f5" strokeWidth={1} />
              <text x={padL + chartW / 2} y={Math.min(yTop, yBot) + h / 2 + 4} fill="#b088f5" fontSize={12} textAnchor="middle" fontFamily="var(--mono)" opacity={0.8} fontWeight="bold">DMZ</text>
            </g>
          );
        })()}

        {fibLevels && fibLevels.map((fib, i) => {
          const y = getY(fib.value);
          if (y < padT - 5 || y > height - padB + 5) return null;
          
          // 🌟 ดึงสีมาใช้ ถ้าไม่มีให้ใช้สีส้ม และทำไฮไลท์ให้หนาขึ้น
          const fColor = fib.color || "#ff9800";
          const isHighlight = fColor === "#00e5ff"; 

          return (
            <g key={`fib-${i}`}>
              <line x1={padL} x2={padL + chartW} y1={y} y2={y} stroke={fColor} strokeWidth={isHighlight ? 1.5 : 1} strokeDasharray="4 4" opacity={isHighlight ? 1 : 0.7} />
              <text x={padL + 4} y={y - 4} fill={fColor} fontSize={9} fontFamily="var(--mono)" opacity={isHighlight ? 1 : 0.9} fontWeight={isHighlight ? "bold" : "normal"}>{fib.label}</text>
              <text x={padL + chartW + 4} y={y + 3} fill={fColor} fontSize={9} fontFamily="var(--mono)" fontWeight={isHighlight ? "bold" : "normal"}>{fmtPrice(fib.value)}</text>
            </g>
          );
        })}

        {srLevels && srLevels.map((sr, i) => {
          const y = getY(sr.value);
          if (y < padT - 5 || y > height - padB + 5) return null;
          return (
            <g key={`sr-${i}`}>
              <line x1={padL} x2={padL + chartW} y1={y} y2={y} stroke={sr.color} strokeWidth={1.5} opacity={0.8} />
              <text x={padL + 4} y={y - 4} fill={sr.color} fontSize={9} fontFamily="var(--mono)" opacity={0.9} fontWeight="bold">{sr.label}</text>
              <text x={padL + chartW + 4} y={y + 3} fill={sr.color} fontSize={9} fontFamily="var(--mono)" fontWeight="bold">{fmtPrice(sr.value)}</text>
            </g>
          );
        })}

        {pts.length > 0 && <path d={fill} fill={`url(#${gid})`} />}
        {pts.length > 0 && <path d={path} stroke={color} strokeWidth="1.5" fill="none" />}
        
        {showVolume && visibleData.map((d, i) => {
          if (!d) return null;
          const actualIndex = sIdx + i;
          const seed = ((d.value || 0) * 13 + actualIndex * 7) % 100;
          const h = (0.3 + (seed / 100) * 0.7) * 12;
          const rectW = Math.min(stepX * 0.7, 10);
          return (<rect key={`vol-${actualIndex}`} x={padL + (actualIndex - viewX.s) * stepX - rectW / 2} y={height - 15 - h} width={rectW} height={h} fill={color} opacity="0.35" />);
        })}

        {hover && (
          <g>
            <line x1={hover.x} x2={hover.x} y1={padT} y2={height - padB} stroke="var(--border-bright)" strokeWidth="1" strokeDasharray="2 2" />
            <line x1={padL} x2={padL + chartW} y1={hover.y} y2={hover.y} stroke="var(--border-bright)" strokeWidth="1" strokeDasharray="2 2" />
            <circle cx={hover.x} cy={hover.y} r="4" fill="var(--bg0)" stroke={color} strokeWidth="2" />
            
            {/* กล่องราคาเลื่อนตามแนวตั้ง (Y) */}
            <rect x={padL + chartW + 2} y={hover.y - 9} width={padR - 4} height="18" fill="var(--bg2)" stroke={color} strokeWidth="1" rx="2" />
            <text x={padL + chartW + padR / 2} y={hover.y + 4} fill={color} fontSize="10" fontFamily="var(--mono)" textAnchor="middle">{fmtPrice(hover.v)}</text>
            
            {/* 🌟 กล่องวันที่ ย้ายขึ้นไปข้างบน (padT) */}
            <rect x={hover.x - 45} y={padT - 18} width="90" height="18" fill="var(--bg2)" stroke="var(--border)" strokeWidth="1" rx="2" />
            <text x={hover.x} y={padT - 5} fill="var(--text-primary)" fontSize="10" fontFamily="var(--mono)" textAnchor="middle">{hover.t}</text>
          </g>
        )}
      </svg>
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

// 🌟 นำ Sparkline กลับมา (กราฟจิ๋วสำหรับ Watchlist และ Top Movers)
export const Sparkline = ({ data, up, width = "100%", height = 28 }: { data: PricePoint[]; up: boolean; width?: number | string; height?: number }) => {
  if (!data || data.length === 0) return <div style={{ width, height }} />;
  
  const vs = data.map(d => d.value || 0);
  const min = Math.min(...vs);
  const max = Math.max(...vs);
  const range = max - min || 1;
  
  const stepX = 100 / Math.max(1, data.length - 1);
  const pts = data.map((d, i) => [i * stepX, 100 - ((d.value - min) / range) * 100]);
  
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const fill = `${path} L100,100 L0,100 Z`;
  const color = up ? "var(--green)" : "var(--red)";
  
  return (
    <svg width={width} height={height} viewBox="0 0 100 100" preserveAspectRatio="none" style={{ display: "block" }}>
      <defs>
        <linearGradient id={`spark-${up ? 'up' : 'down'}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fill} fill={`url(#spark-${up ? 'up' : 'down'})`} />
      <path d={path} stroke={color} strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
};
