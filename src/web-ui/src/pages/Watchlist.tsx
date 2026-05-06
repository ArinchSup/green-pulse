// src/pages/Watchlist.tsx
import React, { useMemo, useState } from "react";
import type { Market } from "../types";
import { Sparkline } from "../primitives";
import { fmtPrice, fmtPct } from "../format";

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
                          gridTemplateColumns: "1fr 1fr 3fr", 
                          gap: "24px", 
                          background: "var(--bg2)", 
                          padding: "20px",
                          border: "1px solid var(--border-bright)", 
                          borderRadius: "6px" 
                        }}>
                          
                          {/* Sec 1: Chart */}
                          <div style={{ borderRight: "1px dashed var(--border)", paddingRight: "24px" }}>
                            <div style={{ fontFamily: "var(--mono)", fontSize: "10px", color: "var(--text-muted)", marginBottom: "12px" }}>1W CHART TREND</div>
                            <Sparkline data={m.data["1W"]} up={m.up} width={200} height={80} />
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