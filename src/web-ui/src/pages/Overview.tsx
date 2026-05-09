// src/pages/Overview.tsx
import { useState } from "react";
import type { Market, NewsItem, RangeKey } from "../types";
import { Sparkline, LineChart, Pill } from "../primitives";
import { fmtPrice, fmtPct } from "../format";
import { RANGE_KEYS } from "../variable";

const Stat = ({ label, value, tone }: { label: string; value: string; tone?: "up" | "down" }) => (
  <div className="stat">
    <div className="stat-label">{label}</div>
    <div className={`stat-value ${tone || ""}`}>{value}</div>
  </div>
);

export const Overview = ({ markets, selectedId, onSelect, range, setRange, news, isChartUpdating }: {
  markets: Market[]; selectedId: string; onSelect: (id: string) => void;
  range: RangeKey; setRange: (r: RangeKey) => void; news: NewsItem[];
  isChartUpdating?: boolean; 
}) => {
  const m = markets.find(x => x.id === selectedId) || markets[0];

  const [showFib, setShowFib] = useState(false);
  const [showSR, setShowSR] = useState(false);
  const [showDMZ, setShowDMZ] = useState(false);
  
  const calcSeries = m.data[range] || []; 
  const history = (m as any).chartHistory;
  
  // 🌟 ใช้ประวัติความละเอียดสูงวาดกราฟ (1D, 1W จะกลับมาละเอียดเหมือนเดิมและซูมได้)
  const drawSeries = history && history.length > 0 ? history : calcSeries;
  
  // 🌟 บังคับใช้ข้อมูล 1Y สำหรับคำนวณ Fib/SR/DMZ เสมอ (แยกส่วนกับตัววาดกราฟ)
  const refSeries = m.data["1Y"] && m.data["1Y"].length > 0 ? m.data["1Y"] : drawSeries; 
  
  let focusStartT = calcSeries.length > 0 ? calcSeries[0].t : undefined;

  // สำหรับ 1D ให้หากราฟวันล่าสุดที่ตลาดเปิด
  if (range === "1D" && drawSeries.length > 0) {
    const lastDate = String(drawSeries[drawSeries.length - 1].t).split(" ")[0];
    const firstCandleOfDay = drawSeries.find((d: any) => String(d.t).startsWith(lastDate));
    if (firstCandleOfDay) focusStartT = firstCandleOfDay.t;
  }

  // 🌟 High/Low สำหรับตีกรอบ Fib และ SR ใช้ refSeries (กรอบ 1Y แน่นอน)
  const high = refSeries.length > 0 ? Math.max(...refSeries.map((d: any) => d.value)) : 0;
  const low  = refSeries.length > 0 ? Math.min(...refSeries.map((d: any) => d.value)) : 0;
  
  // 🌟 1. ใช้ refSeries (1Y) สำหรับคำนวณ Fib และแนวรับแนวต้าน
  const fibHigh = refSeries.length > 0 ? Math.max(...refSeries.map((d: any) => d.value)) : 0;
  const fibLow  = refSeries.length > 0 ? Math.min(...refSeries.map((d: any) => d.value)) : 0;
  const fibRng  = fibHigh - fibLow;
  const refLast = refSeries.length > 0 ? refSeries[refSeries.length - 1].value : 1;

  // 🌟 2. ใช้ calcSeries (ข้อมูลปัจจุบัน) สำหรับแสดง Stats ด้านล่าง (แก้กราฟ 1W แบนราบ)
  const statHigh = calcSeries.length > 0 ? Math.max(...calcSeries.map((d: any) => d.value)) : 0;
  const statLow  = calcSeries.length > 0 ? Math.min(...calcSeries.map((d: any) => d.value)) : 0;
  const statRng  = statHigh - statLow;

  const open = calcSeries.length > 0 ? calcSeries[0].value : 1;
  const last = calcSeries.length > 0 ? calcSeries[calcSeries.length - 1].value : 1;
  const periodChg = ((last - open) / open) * 100;
  
  const sortedVals = [...refSeries.map((d: any) => d.value)].sort((a,b) => a-b);
  const bottom10Count = Math.max(1, Math.floor(sortedVals.length * 0.10));
  const dmzTop = sortedVals[bottom10Count - 1]; 
  
  const demandZone = showDMZ ? [fibLow, dmzTop] as [number, number] : undefined;

  // 🌟 ลอจิก AI หาเส้น Fib ที่ใกล้ราคาปัจจุบันที่สุด (ตามสูตร Shay)
  const fib38 = fibLow + fibRng * 0.382;
  const fib61 = fibLow + fibRng * 0.618;
  const fib78 = fibLow + fibRng * 0.786;

  let closestLabel = "38.2%";
  let minDiff = Math.abs(last - fib38);
  if (Math.abs(last - fib61) < minDiff) { minDiff = Math.abs(last - fib61); closestLabel = "61.8%"; }
  if (Math.abs(last - fib78) < minDiff) { minDiff = Math.abs(last - fib78); closestLabel = "78.6%"; }

  const getFibColor = (lbl: string) => lbl === closestLabel ? "#00e5ff" : "#ff9800";

  const fibLevels = showFib ? [
    { label: "0.0%", value: fibLow, color: "#ff9800" },
    { label: "23.6%", value: fibLow + fibRng * 0.236, color: "#ff9800" },
    { label: "38.2%", value: fib38, color: getFibColor("38.2%") },
    { label: "50.0%", value: fibLow + fibRng * 0.500, color: "#ff9800" },
    { label: "61.8%", value: fib61, color: getFibColor("61.8%") },
    { label: "78.6%", value: fib78, color: getFibColor("78.6%") },
    { label: "100.0%", value: fibHigh, color: "#ff9800" }
  ] as any : undefined;

  // 🌟 แก้ S/R ให้คำนวณจาก Timeframe ปัจจุบัน (statHigh, statLow) แทน 1Y เพื่อไม่ให้เส้นหลุดจอ
  const P = (statHigh + statLow + last) / 3;
  const srLevels = showSR ? [
    { label: "RES2", value: P + statRng * 0.618, color: "#ff4466" },
    { label: "RES1", value: P + statRng * 0.382, color: "#ff7788" },
    { label: "SUP1", value: P - statRng * 0.382, color: "#00ee77" },
    { label: "SUP2", value: P - statRng * 0.618, color: "#00d46a" }
  ] as any : undefined;

  const otherStocks = markets.filter(x => x.id !== selectedId && x.sector === "Equity");
  const movers  = [...markets].sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 4);

  return (
    <div className="page">
      <div className="ticker-strip">
        {markets.map(t => (
          <button key={t.id} onClick={() => onSelect(t.id)} className={`ticker-cell ${t.id === selectedId ? "active" : ""}`}>
            <div className="tk-label">{t.ticker}</div>
            <div className="tk-price">{fmtPrice(t.price)}</div>
            <div className={t.up ? "tk-chg up" : "tk-chg down"}>{t.up ? "▲" : "▼"} {fmtPct(t.change)}</div>
          </button>
        ))}
      </div>
      <div className="grid-main">
        <div className="card chart-card" style={{ minWidth: 0 }}>
          
          {/* === 1. ส่วนหัว (ชื่อหุ้น, ราคา, ปุ่มเวลา) === */}
          <div className="chart-header">
            <div>
              <div className="chart-meta">
                <span className="chart-ticker">{m.ticker}</span>
                <span className="chart-name">{m.label}</span>
                <Pill tone="neutral">{m.sector}</Pill>
              </div>
              <div className="chart-price-row">
                <div className="chart-price">${fmtPrice(m.price)}</div>
                <div className={m.up ? "chart-change-up" : "chart-change-down"}>
                  {m.up ? "▲" : "▼"} {fmtPct(m.change)} <span className="chg-sub">today</span>
                </div>
              </div>
            <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
              
              {/* กลุ่มปุ่ม Technical Indicators */}
              <div className="time-buttons" style={{ background: "var(--bg1)", border: "1px solid var(--border)" }}>
                <button onClick={() => setShowFib(!showFib)} className="time-btn" style={{ color: showFib ? "#ff9800" : "var(--text-muted)" }}>FIB</button>
                <button onClick={() => setShowSR(!showSR)} className="time-btn" style={{ color: showSR ? "#00e5ff" : "var(--text-muted)" }}>R/S</button>
                <button onClick={() => setShowDMZ(!showDMZ)} className="time-btn" style={{ color: showDMZ ? "#b088f5" : "var(--text-muted)" }}>DMZ</button>
              </div>

              {/* กลุ่มปุ่ม Timeframe เดิม */}
              <div className="time-buttons">
                {RANGE_KEYS.map(t => (
                  <button key={t} onClick={() => setRange(t)} className={`time-btn ${t === range ? "active" : ""}`}>{t}</button>
                ))}
              </div>
            </div>
          </div>
          </div>

          {/* === 2. กล่องแสดงกราฟ (จะตกลงมาอยู่ด้านล่างอย่างสวยงาม) === */}
          <div className="chart-container" style={{ position: "relative", width: "100%", minWidth: 0, overflow: "hidden" }}>
            
            {/* Overlay ตอนโหลด */}
            {isChartUpdating && (
              <div style={{
                position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
                background: "rgba(10, 14, 10, 0.6)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 10, color: "var(--green)", fontFamily: "var(--mono)", fontSize: "12px", letterSpacing: "1px"
              }}>
                <span className="dot live" style={{ marginRight: "8px" }}></span> UPDATING CHART...
              </div>
            )}

            {/* ถ้ามีข้อมูลให้วาดกราฟ */}
            {drawSeries.length > 0 ? (
              <>
                 <LineChart 
                   key={range} // 🌟 บังคับกราฟรีเซ็ตจอ 100% เมื่อเปลี่ยน Timeframe
                   data={drawSeries} 
                   focusStartT={focusStartT}
                   focusLength={calcSeries.length}
                   up={periodChg >= 0} 
                   height={260} 
                   showVolume 
                   fibLevels={fibLevels} 
                   srLevels={srLevels} 
                   demandZone={demandZone}
                 />
                 
                 <div className="chart-stats" style={{ marginTop: "16px" }}>
                   <Stat label="Open" value={`$${fmtPrice(open)}`} />
                   <Stat label="High" value={`$${fmtPrice(statHigh)}`} />
                   <Stat label="Low"  value={`$${fmtPrice(statLow)}`} />
                   <Stat label={`${range} Δ`} value={fmtPct(periodChg)} tone={periodChg >= 0 ? "up" : "down"} />
                   <Stat label="Range" value={`$${fmtPrice(statRng)}`} />
                   <Stat label="Volatility" value={`${((statRng / open) * 100).toFixed(2)}%`} />
                 </div>
              </>
            ) : (
              <div style={{ height: 260, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontFamily: "var(--mono)" }}>
                No Data Available
              </div>
            )}
          </div>
        </div>
        <div className="right-col">
          <div className="card">
            <div className="card-title">Top Movers</div>
            <div className="movers">
              {movers.map(t => (
                <div key={t.id} className="mover-row" onClick={() => onSelect(t.id)}>
                  <div>
                    <div className="mover-tk">{t.ticker}</div>
                    <div className="mover-name">{t.label}</div>
                  </div>
                  <Sparkline data={t.data["1D"]} up={t.up} width={64} height={22} />
                  <div className="mover-right">
                    <div className="mover-price">${fmtPrice(t.price)}</div>
                    <div className={t.up ? "mover-chg up" : "mover-chg down"}>{fmtPct(t.change)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-title">Market Pulse</div>
            <div className="pulse-grid">
              {/* 🌟 เปลี่ยนจาก indices เป็น otherStocks */}
              {otherStocks.slice(0, 4).map(t => (
                <div key={t.id} className="pulse-cell" onClick={() => onSelect(t.id)}>
                  <div className="pulse-label">{t.ticker}</div>
                  <div className="pulse-price">${fmtPrice(t.price)}</div>
                  <Sparkline data={t.data["1W"]} up={t.up} width="100%" height={28} />
                  <div className={t.up ? "pulse-chg up" : "pulse-chg down"}>{fmtPct(t.change)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="card news-card">
        <div className="card-title">
          <span>Newswire</span>
          <span className="card-sub">Live · {news.length} items</span>
        </div>
        <div className="news-list">
          {news.map((n, i) => (
            <div key={i} className="news-item">
              <div className="news-time">{n.time}</div>
              <div className="news-tag">{n.tag}</div>
              <div className="news-headline">{n.headline}</div>
              <div className="news-source">{n.source}</div>
              <div className={`news-impact ${n.impact}`}>
                {n.impact === "up" ? "▲" : n.impact === "down" ? "▼" : "—"}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

