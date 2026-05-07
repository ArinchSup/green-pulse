// src/pages/Watchlist.tsx
import React, { useMemo, useState, useEffect } from "react"; // 🌟 เพิ่ม useEffect ตรงนี้
import { Sparkline } from "../primitives";
import { fmtPrice, fmtPct } from "../format";
import { LineChart } from "../primitives";
import { RANGE_KEYS } from "../variable";
import type { Market, RangeKey, PricePoint } from "../types";

const WatchlistChart = ({ market }: { market: Market }) => {
  const [range, setRange] = useState<RangeKey>("1W");
  const [localSeries, setLocalSeries] = useState<PricePoint[]>(market.data["1W"] || []);
  const [historySeries, setHistorySeries] = useState<PricePoint[]>([]); // 🌟 เก็บประวัติ
  const [showFib, setShowFib] = useState(false);
  const [showSR, setShowSR] = useState(false);
  const [showDMZ, setShowDMZ] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchLocalData = async () => {
      setIsLoading(true);
      try {
        const res = await fetch("http://localhost:8000/chart", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: market.ticker, period: range })
        });
        
        // 🌟 2. แก้ให้ตรงกันกับหน้า App.tsx
        let hPeriod = "5Y";
        if (range === "1D") hPeriod = "1W";
        else if (range === "1W") hPeriod = "3M";
        else if (range === "1M") hPeriod = "1Y";
        else if (range === "3M") hPeriod = "5Y";

        const resH = await fetch("http://localhost:8000/chart", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: market.ticker, period: hPeriod })
        });

        if (res.ok) {
          const json = await res.json();
          if (json.data && json.data.length > 0) setLocalSeries(json.data);
        }
        if (resH.ok) {
          const jsonH = await resH.json();
          if (jsonH.data && jsonH.data.length > 0) setHistorySeries(jsonH.data);
        }
      } catch (e) { console.error("Watchlist chart fetch error", e); } 
      finally { setIsLoading(false); }
    };
    fetchLocalData();
  }, [range, market.ticker]);

  if (localSeries.length === 0) return <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>Loading chart...</div>;

  // 🌟 2. ล้างระบบเย็บกราฟออก ใช้ของใครของมัน
  const calcSeries = localSeries;
  const drawSeries = calcSeries;
  const focusStartT = calcSeries.length > 0 ? calcSeries[0].t : undefined;

  // คำนวณค่าต่างๆ อิงจาก localSeries เท่านั้น
  const high = Math.max(...localSeries.map(d => d.value));
  const low  = Math.min(...localSeries.map(d => d.value));
  const open = localSeries[0].value;
  const last = localSeries[localSeries.length - 1].value;
  const up = last >= open;
  const rng = high - low;

  const fibLevels = showFib ? [
    { label: "161.8%", value: low + rng * 1.618 }, { label: "100.0%", value: high },
    { label: "61.8%", value: low + rng * 0.618 }, { label: "50.0%", value: low + rng * 0.5 },
    { label: "38.2%", value: low + rng * 0.382 }, { label: "0.0%", value: low }
  ] : undefined;

  const srLevels = showSR ? [
    { label: "R2", value: high - rng * 0.05, color: "#00e5ff" }, { label: "R1", value: high - rng * 0.2, color: "#00e5ff" },
    { label: "S1", value: low + rng * 0.25, color: "#00e5ff" }, { label: "S2", value: low + rng * 0.05, color: "#00e5ff" }
  ] : undefined;

  const sortedVals = [...localSeries.map(d => d.value)].sort((a,b) => a-b);
  const dmzTop = sortedVals[Math.max(1, Math.floor(sortedVals.length * 0.10)) - 1];
  const demandZone = showDMZ ? [low, dmzTop] as [number, number] : undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", position: "relative" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "8px" }}>
        <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--mono)", letterSpacing: "1px" }}>{range} CHART TREND</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", transform: "scale(0.85)", transformOrigin: "left center" }}>
          <div className="time-buttons" style={{ background: "var(--bg1)", border: "1px solid var(--border)" }}>
            <button onClick={() => setShowFib(!showFib)} className="time-btn" style={{ color: showFib ? "#ff9800" : "var(--text-muted)" }}>FIB</button>
            <button onClick={() => setShowSR(!showSR)} className="time-btn" style={{ color: showSR ? "#00e5ff" : "var(--text-muted)" }}>R/S</button>
            <button onClick={() => setShowDMZ(!showDMZ)} className="time-btn" style={{ color: showDMZ ? "#b088f5" : "var(--text-muted)" }}>DMZ</button>
          </div>
          <div className="time-buttons">
            {RANGE_KEYS.map(t => <button key={t} onClick={() => setRange(t)} className={`time-btn ${t === range ? "active" : ""}`}>{t}</button>)}
          </div>
        </div>
      </div>
      <div style={{ position: "relative", height: "180px", width: "100%" }}>
        {isLoading && <div style={{ position: "absolute", inset: 0, background: "rgba(10,14,10,0.6)", zIndex: 10, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--green)", fontSize: "10px", fontFamily: "var(--mono)" }}>UPDATING...</div>}
        {/* 🌟 วาดกราฟโดยใช้ drawSeries และสั่งโฟกัสที่ localSeries */}
        <LineChart 
           data={drawSeries} 
           focusStartT={focusStartT} // 🌟 โยนวันที่ลงไป
           focusLength={localSeries.length} 
           up={up} 
           height={180} 
           showVolume={true} 
           fibLevels={fibLevels} 
           srLevels={srLevels} 
           demandZone={demandZone} 
        />
      </div>
    </div>
  );
};

export const Watchlist = ({ 
  markets, onSelect, onTrade, watched, toggleWatch, analyses = {}, 
  horizons = {}, onHorizonChange 
}: {
  markets: Market[];
  onSelect: (id: string) => void;
  onTrade: (ticker: string) => void;
  watched: Set<string>;
  toggleWatch: (id: string) => void;
  analyses?: Record<string, any>;
  horizons?: Record<string, string>;
  onHorizonChange?: (ticker: string, h: string) => void;
}) => {
  type SortCol = "ticker" | "sector" | "price" | "change";
  const [sortBy, setSortBy] = useState<SortCol>("change");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<string>("All");
  const [expandedTickers, setExpandedTickers] = useState<Set<string>>(new Set());

  const sectors = ["All", ...new Set(markets.map(m => m.sector))];

  const sorted = useMemo(() => {
    let list = markets.filter(m =>
      (filter === "All" || m.sector === filter) &&
      (q === "" || m.ticker.toLowerCase().includes(q.toLowerCase()) || m.label.toLowerCase().includes(q.toLowerCase()))
    );
    list = [...list].sort((a, b) => {
      const av = a[sortBy] as any, bv = b[sortBy] as any;
      const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
      return dir === "asc" ? cmp : -cmp;
    });
    return list;
  }, [markets, sortBy, dir, q, filter]);

  const click = (col: SortCol) => {
    if (sortBy === col) setDir(dir === "asc" ? "desc" : "asc");
    else { setSortBy(col); setDir("desc"); }
  };

  const Header = ({ col, children, align = "left" }: { col: SortCol; children: React.ReactNode; align?: "left" | "right" }) => (
    <th onClick={() => click(col)} style={{ textAlign: align, cursor: "pointer" }}>
      {children}
      <span className="sort-arrow">{sortBy === col ? (dir === "asc" ? "▲" : "▼") : "↕"}</span>
    </th>
  );

  const toggleExpand = (ticker: string) => {
    setExpandedTickers(prev => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker); 
      else next.add(ticker);                     
      return next;
    });
  };

  return (
    <div className="page">
      <div className="watchlist-toolbar">
        <input className="search-input" placeholder="Search ticker or name…" value={q} onChange={e => setQ(e.target.value)} />
        <div className="filter-chips">
          {sectors.map(s => (
            <button key={s} className={`chip ${filter === s ? "active" : ""}`} onClick={() => setFilter(s)}>{s}</button>
          ))}
        </div>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 40 }}></th>
              <Header col="ticker">Symbol</Header>
              <Header col="sector">Sector</Header>
              <Header col="price" align="right">Price</Header>
              <Header col="change" align="right">Day Δ</Header>
              <th style={{ textAlign: "center" }}>Horizon</th>
              <th style={{ textAlign: "center" }}>AI Signal</th>
              <th style={{ textAlign: "center" }}>Trend (1W)</th>
              <th style={{ textAlign: "right", width: 80 }}></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(m => {
              const aiData = analyses[m.ticker];
              const signal = aiData ? (aiData.action || aiData.ai_analysis?.action || "HOLD") : "LOADING...";
              const isExpanded = expandedTickers.has(m.ticker);
              
              let signalColor = "var(--text-muted)";
              if (signal === "BUY") signalColor = "var(--up)";
              if (signal === "SELL") signalColor = "var(--down)";

              return (
                <React.Fragment key={m.id}>
                  {/* Main Row */}
                  <tr 
                    onClick={() => toggleExpand(m.ticker)} 
                    style={{ cursor: "pointer", background: isExpanded ? "rgba(255,255,255,0.02)" : "transparent" }}
                  >
                    <td>
                      <button className="watch-star" onClick={(e) => { e.stopPropagation(); toggleWatch(m.id); }} aria-label="Toggle watch">
                        {watched.has(m.id) ? "★" : "☆"}
                      </button>
                    </td>
                    <td>
                      <div className="sym-cell">
                        <div className="sym-tk">{m.ticker}</div>
                        <div className="sym-name">{m.label}</div>
                      </div>
                    </td>
                    <td className="dim">{m.sector}</td>
                    <td className="num">${fmtPrice(m.price)}</td>
                    <td className={m.up ? "num up" : "num down"}>{m.up ? "▲" : "▼"} {fmtPct(m.change)}</td>
                    
                    <td style={{ textAlign: "center" }}>
                      <select
                        value={horizons[m.ticker] || "Mid-term"}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => onHorizonChange && onHorizonChange(m.ticker, e.target.value)}
                        style={{
                          background: "var(--bg3)", color: "var(--text-secondary)", border: "1px solid var(--border)",
                          padding: "4px 6px", fontFamily: "var(--mono)", fontSize: "10px", outline: "none", cursor: "pointer", borderRadius: "4px"
                        }}
                      >
                        <option value="Short-term">Short</option>
                        <option value="Mid-term">Mid</option>
                        <option value="Long-term">Long</option>
                      </select>
                    </td>

                    <td style={{ textAlign: "center", fontFamily: "var(--mono)", fontWeight: "bold", color: signalColor, fontSize: "11px" }}>
                      {signal}
                    </td>
                    
                    <td style={{ display: "flex", justifyContent: "center" }}>
                      <Sparkline data={m.data["1W"]} up={m.up} width={120} height={28} />
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button className="mini-btn" onClick={(e) => { e.stopPropagation(); onTrade(m.ticker); }}>Trade</button>
                    </td>
                  </tr>

                  {/* Expanded Row */}
                  {isExpanded && (
                    <tr style={{ background: "var(--bg1)", borderBottom: "1px solid var(--border)" }}>
                      <td colSpan={9} style={{ padding: "16px 24px" }}>
                        
                        <div style={{ 
                          display: "grid", 
                          // 🌟 2. ขยายช่องกราฟเป็น 2.5 ส่วน และลดช่อง AI เหลือ 2 ส่วน
                          gridTemplateColumns: "2.5fr 1fr 2fr", 
                          gap: "24px", 
                          background: "var(--bg2)", 
                          padding: "20px",
                          border: "1px solid var(--border-bright)", 
                          borderRadius: "6px",
                          minWidth: 0 // ดักไม่ให้ Flex ทะลุจอ
                        }}>
                          
                          {/* Sec 1: Chart */}
                          <div style={{ borderRight: "1px dashed var(--border)", paddingRight: "24px", display: "flex", flexDirection: "column", minWidth: 0 }}>
                            <WatchlistChart market={m} />
                          </div>

                          {/* Sec 2: Setup (Entry, Target, Stop) */}
                          <div style={{ borderRight: "1px dashed var(--border)", paddingRight: "24px", display: "flex", flexDirection: "column", gap: "12px", justifyContent: "flex-start" }}>
                            <div style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-muted)", marginBottom: "4px" }}>TRADE SETUP</div>
                            
                            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "13px" }}>
                              <span style={{ color: "var(--text-secondary)" }}>RECOMMENDED ENTRY</span>
                              <span style={{ color: "var(--text-primary)" }}>{aiData?.ai_analysis?.recommended_entry_price || "-"}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "13px" }}>
                              <span style={{ color: "var(--text-secondary)" }}>TARGET</span>
                              <span style={{ color: "var(--up)" }}>{aiData?.ai_analysis?.target_price || "-"}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: "13px" }}>
                              <span style={{ color: "var(--text-secondary)" }}>STOP LOSS</span>
                              <span style={{ color: "var(--down)" }}>{aiData?.ai_analysis?.stop_loss || "-"}</span>
                            </div>
                          </div>

                          {/* Sec 3: Analysis Rationale */}
                          <div>
                            <div style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-muted)", marginBottom: "8px" }}>QUANT AI RATIONALE</div>
                            {/* Set scrollable container */}
                            <div className="scroll" style={{ 
                              fontSize: "13px", 
                              color: "var(--text-primary)", 
                              lineHeight: "1.6", 
                              height: "130px", 
                              overflowY: "auto",
                              paddingRight: "12px",
                              whiteSpace: "pre-wrap" 
                            }}>
                              {aiData?.ai_analysis?.analysis?.rationale || "AI Analysis pending or no rationale provided for this stock."}
                              
                              {aiData?.ai_analysis?.analysis?.technical_view && (
                                <div style={{ marginTop: "12px" }}>
                                  <div style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-secondary)", marginBottom: "4px" }}>TECHNICAL VIEW:</div>
                                  <div style={{ color: "var(--text-primary)" }}>
                                    {aiData.ai_analysis.analysis.technical_view}
                                  </div>
                                </div>
                              )}

                              {aiData?.ai_analysis?.analysis?.fundamental_view && (
                                <div style={{ marginTop: "12px" }}>
                                  <div style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-secondary)", marginBottom: "4px" }}>FUNDAMENTAL VIEW:</div>
                                  <div style={{ color: "var(--text-primary)" }}>
                                    {aiData.ai_analysis.analysis.fundamental_view}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>

                        </div>

                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
            {sorted.length === 0 && (<tr><td colSpan={9} className="empty">No matches.</td></tr>)}
          </tbody>
        </table>
      </div>
    </div>
  );
};