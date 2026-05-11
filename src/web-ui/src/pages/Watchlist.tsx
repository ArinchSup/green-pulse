// src/pages/Watchlist.tsx
import React, { useEffect, useMemo, useState } from "react";
import { LineChart, Sparkline } from "../primitives";
import { fmtPrice, fmtPct } from "../format";
import { RANGE_KEYS } from "../variable";
import type { Market, PricePoint, RangeKey, SectorKey } from "../types";

// ── AI Analysis panel (self-fetching) ───────────────────────────────────────

const AIAnalysisPanel = ({ ticker, horizon }: { ticker: string; horizon: string }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true); setError(false); setData(null);
    fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, horizon }),
    })
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(json => { setData(json); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [ticker, horizon]);

  const mono: React.CSSProperties = { fontFamily: "var(--mono)", fontSize: "11px" };
  const label: React.CSSProperties = { ...mono, color: "var(--text-secondary)" };
  const muted: React.CSSProperties = { ...mono, color: "var(--text-muted)", fontSize: "10px", letterSpacing: "1px", textTransform: "uppercase" as const, marginBottom: "8px" };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", ...mono }}>
      <span style={{ color: "var(--green)", animation: "pulse 1.4s ease-in-out infinite" }}>●</span> ANALYZING...
    </div>
  );
  if (error) return (
    <div style={{ color: "var(--down)", ...mono }}>AI service unavailable — make sure port 8000 is running</div>
  );

  const ai = data?.ai_analysis;
  const action: string = ai?.action ?? "—";
  const actionColor = action === "BUY" ? "var(--up)" : action === "SELL" ? "var(--down)" : "var(--text-secondary)";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
      {/* Trade setup */}
      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        <div style={muted}>Trade Setup</div>
        <div style={{ display: "flex", justifyContent: "space-between", ...mono }}>
          <span style={label}>Signal</span>
          <span style={{ color: actionColor, fontWeight: "bold" }}>{action}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", ...mono }}>
          <span style={label}>Confidence</span>
          <span style={{ color: "var(--text-primary)" }}>{ai?.confidence_score != null ? `${ai.confidence_score}%` : "—"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", ...mono }}>
          <span style={label}>Entry</span>
          <span style={{ color: "var(--text-primary)" }}>{ai?.recommended_entry_price ?? "—"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", ...mono }}>
          <span style={label}>Target</span>
          <span style={{ color: "var(--up)" }}>{ai?.target_price ?? "—"}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", ...mono }}>
          <span style={label}>Stop Loss</span>
          <span style={{ color: "var(--down)" }}>{ai?.stop_loss ?? "—"}</span>
        </div>
      </div>

      {/* Rationale */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        <div style={muted}>Quant AI Rationale</div>
        <div className="scroll" style={{ fontSize: "12px", color: "var(--text-primary)", lineHeight: "1.6", maxHeight: "130px", overflowY: "auto", paddingRight: "8px", whiteSpace: "pre-wrap" }}>
          {ai?.analysis?.rationale ?? "No rationale provided."}
          {ai?.analysis?.technical_view && (
            <div style={{ marginTop: "10px" }}>
              <div style={{ ...mono, color: "var(--text-secondary)", marginBottom: "4px" }}>TECHNICAL:</div>
              {ai.analysis.technical_view}
            </div>
          )}
          {ai?.analysis?.fundamental_view && (
            <div style={{ marginTop: "10px" }}>
              <div style={{ ...mono, color: "var(--text-secondary)", marginBottom: "4px" }}>FUNDAMENTAL:</div>
              {ai.analysis.fundamental_view}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Watchlist chart (self-fetching) ─────────────────────────────────────────

const WatchlistChart = ({ market }: { market: Market }) => {
  const [range, setRange] = useState<RangeKey>("1W");
  const [localSeries, setLocalSeries] = useState<PricePoint[]>(market.data["1W"] || []);
  const [historySeries, setHistorySeries] = useState<PricePoint[]>([]);
  const [showFib, setShowFib] = useState(false);
  const [showSR, setShowSR] = useState(false);
  const [showDMZ, setShowDMZ] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [refSeries1Y, setRefSeries1Y] = useState<PricePoint[]>([]);

  useEffect(() => {
    setIsLoading(true);
    const histPeriod = range === "1D" || range === "1W" ? "1W" : range === "1M" ? "1Y" : "5Y";
    Promise.all([
      fetch("http://localhost:8000/chart", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticker: market.ticker, period: range }) }),
      fetch("http://localhost:8000/chart", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticker: market.ticker, period: histPeriod }) }),
      fetch("http://localhost:8000/chart", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ticker: market.ticker, period: "1Y" }) }),
    ])
      .then(([r, rH, r1Y]) => Promise.all([r.json(), rH.json(), r1Y.json()]))
      .then(([j, jH, j1Y]) => {
        if (j.data?.length)   setLocalSeries(j.data);
        if (jH.data?.length)  setHistorySeries(jH.data);
        if (j1Y.data?.length) setRefSeries1Y(j1Y.data);
      })
      .catch(e => console.error("Watchlist chart fetch error", e))
      .finally(() => setIsLoading(false));
  }, [range, market.ticker]);

  if (localSeries.length === 0) return <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>Loading chart...</div>;

  const calcSeries = localSeries;
  const drawSeries = historySeries.length > 0 ? historySeries : calcSeries;
  const refSeries  = refSeries1Y.length > 0 ? refSeries1Y : drawSeries;

  let focusStartT = calcSeries[0]?.t;
  if (range === "1D" && drawSeries.length > 0) {
    const lastDate = String(drawSeries[drawSeries.length - 1].t).split(" ")[0];
    const first = drawSeries.find((d: any) => String(d.t).startsWith(lastDate));
    if (first) focusStartT = first.t;
  }

  const fibHigh = Math.max(...refSeries.map((d: any) => d.value));
  const fibLow  = Math.min(...refSeries.map((d: any) => d.value));
  const fibRng  = fibHigh - fibLow;
  const open    = calcSeries[0]?.value ?? 1;
  const last    = calcSeries[calcSeries.length - 1]?.value ?? 1;
  const up      = last >= open;

  const sortedVals = [...refSeries.map((d: any) => d.value)].sort((a, b) => a - b);
  const dmzTop = sortedVals[Math.max(1, Math.floor(sortedVals.length * 0.10)) - 1];
  const demandZone = showDMZ ? [fibLow, dmzTop] as [number, number] : undefined;

  const statHigh = Math.max(...calcSeries.map((d: any) => d.value));
  const statLow  = Math.min(...calcSeries.map((d: any) => d.value));
  const statRng  = statHigh - statLow;
  const fib38 = fibLow + fibRng * 0.382;
  const fib61 = fibLow + fibRng * 0.618;
  const fib78 = fibLow + fibRng * 0.786;

  let closestLabel = "38.2%";
  let minDiff = Math.abs(last - fib38);
  if (Math.abs(last - fib61) < minDiff) { minDiff = Math.abs(last - fib61); closestLabel = "61.8%"; }
  if (Math.abs(last - fib78) < minDiff) { closestLabel = "78.6%"; }

  const getFibColor = (lbl: string) => lbl === closestLabel ? "#00e5ff" : "#ff9800";
  const fibLevels = showFib ? [
    { label: "0.0%",   value: fibLow,                  color: "#ff9800" },
    { label: "23.6%",  value: fibLow + fibRng * 0.236, color: "#ff9800" },
    { label: "38.2%",  value: fib38,                   color: getFibColor("38.2%") },
    { label: "50.0%",  value: fibLow + fibRng * 0.500, color: "#ff9800" },
    { label: "61.8%",  value: fib61,                   color: getFibColor("61.8%") },
    { label: "78.6%",  value: fib78,                   color: getFibColor("78.6%") },
    { label: "100.0%", value: fibHigh,                  color: "#ff9800" },
  ] as any : undefined;

  const P = (statHigh + statLow + last) / 3;
  const srLevels = showSR ? [
    { label: "RES2", value: P + statRng * 0.618, color: "#ff4466" },
    { label: "RES1", value: P + statRng * 0.382, color: "#ff7788" },
    { label: "SUP1", value: P - statRng * 0.382, color: "#00ee77" },
    { label: "SUP2", value: P - statRng * 0.618, color: "#00d46a" },
  ] as any : undefined;

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", position: "relative" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "8px" }}>
        <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--mono)", letterSpacing: "1px" }}>{range} CHART TREND</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", transform: "scale(0.85)", transformOrigin: "left center" }}>
          <div className="time-buttons" style={{ background: "var(--bg1)", border: "1px solid var(--border)" }}>
            <button onClick={() => setShowFib(!showFib)} className="time-btn" style={{ color: showFib ? "#ff9800" : "var(--text-muted)" }}>FIB</button>
            <button onClick={() => setShowSR(!showSR)}   className="time-btn" style={{ color: showSR  ? "#00e5ff" : "var(--text-muted)" }}>R/S</button>
            <button onClick={() => setShowDMZ(!showDMZ)} className="time-btn" style={{ color: showDMZ ? "#b088f5" : "var(--text-muted)" }}>DMZ</button>
          </div>
          <div className="time-buttons">
            {RANGE_KEYS.map(t => <button key={t} onClick={() => setRange(t)} className={`time-btn ${t === range ? "active" : ""}`}>{t}</button>)}
          </div>
        </div>
      </div>
      <div style={{ position: "relative", height: "180px", width: "100%" }}>
        {isLoading && (
          <div style={{ position: "absolute", inset: 0, background: "rgba(10,14,10,0.6)", zIndex: 10, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--green)", fontSize: "10px", fontFamily: "var(--mono)" }}>
            UPDATING...
          </div>
        )}
        <LineChart key={range} data={drawSeries} focusStartT={focusStartT} focusLength={calcSeries.length}
                   up={up} height={180} showVolume={true} fibLevels={fibLevels} srLevels={srLevels} demandZone={demandZone} />
      </div>
    </div>
  );
};

// ── Main Watchlist component ─────────────────────────────────────────────────

const createStub = (ticker: string): Market => ({
  id: ticker.toLowerCase(),
  label: ticker,
  ticker: ticker.toUpperCase(),
  sector: "Equity" as SectorKey,
  base: 0, price: 0, change: 0, up: false,
  data: { "1D": [], "1W": [], "1M": [], "3M": [], "1Y": [], "5Y": [] } as Record<RangeKey, PricePoint[]>,
});

export const Watchlist = ({
  markets,
  favTickers, addFavorite, removeFavorite,
  topicWatchlists, toggleTopic,
}: {
  markets: Market[];
  favTickers: string[];
  addFavorite: (ticker: string) => void;
  removeFavorite: (ticker: string) => void;
  topicWatchlists: Record<string, string[]>;
  toggleTopic: (id: string, listName: string) => void;
}) => {
  type SortCol = "ticker" | "sector" | "price" | "change";
  const [sortBy, setSortBy]     = useState<SortCol>("change");
  const [dir, setDir]           = useState<"asc" | "desc">("desc");
  const [searchInput, setSearchInput] = useState("");
  const [expandedTickers, setExpandedTickers] = useState<Set<string>>(new Set());
  const [liveQuotes, setLiveQuotes] = useState<Record<string, { price: number; change: number; up: boolean; sparkline: PricePoint[] }>>({});

  const lists = ["Favorites", ...Object.keys(topicWatchlists)];
  const [activeList, setActiveList] = useState("Favorites");

  const favoriteMarkets = useMemo(() =>
    favTickers.map(t => markets.find(m => m.ticker.toUpperCase() === t) ?? createStub(t)),
    [favTickers, markets]
  );

  const listMarkets = activeList === "Favorites"
    ? favoriteMarkets
    : markets.filter(m => (topicWatchlists[activeList] || []).includes(m.id));

  // Fetch live 1D quotes for all favorites
  useEffect(() => {
    favTickers.forEach(ticker => {
      fetch("http://localhost:8000/chart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, period: "1D" }),
      })
        .then(r => r.json())
        .then((j: { data: PricePoint[] }) => {
          if (!j.data?.length) return;
          const open  = j.data[0].value;
          const price = j.data[j.data.length - 1].value;
          const change = ((price - open) / open) * 100;
          setLiveQuotes(prev => ({ ...prev, [ticker]: { price, change, up: change >= 0, sparkline: j.data } }));
        })
        .catch(() => {});
    });
  }, [favTickers.join(",")]);

  const sorted = useMemo(() => {
    return [...listMarkets].sort((a, b) => {
      const qa = liveQuotes[a.ticker.toUpperCase()];
      const qb = liveQuotes[b.ticker.toUpperCase()];
      let av: any = sortBy === "price"  ? (qa?.price  ?? a.price)
                  : sortBy === "change" ? (qa?.change ?? a.change)
                  : (a as any)[sortBy];
      let bv: any = sortBy === "price"  ? (qb?.price  ?? b.price)
                  : sortBy === "change" ? (qb?.change ?? b.change)
                  : (b as any)[sortBy];
      const cmp = typeof av === "string" ? av.localeCompare(bv) : av - bv;
      return dir === "asc" ? cmp : -cmp;
    });
  }, [listMarkets, liveQuotes, sortBy, dir]);

  const clickHeader = (col: SortCol) => {
    if (sortBy === col) setDir(dir === "asc" ? "desc" : "asc");
    else { setSortBy(col); setDir("desc"); }
  };

  const toggleExpand = (ticker: string) => {
    setExpandedTickers(prev => {
      const next = new Set(prev);
      next.has(ticker) ? next.delete(ticker) : next.add(ticker);
      return next;
    });
  };

  const handleAdd = () => {
    const t = searchInput.trim().toUpperCase();
    if (t) { addFavorite(t); setSearchInput(""); }
  };

  const Header = ({ col, children, align = "left" }: { col: SortCol; children: React.ReactNode; align?: "left" | "right" }) => (
    <th onClick={() => clickHeader(col)} style={{ textAlign: align, cursor: "pointer" }}>
      {children}<span className="sort-arrow">{sortBy === col ? (dir === "asc" ? "▲" : "▼") : "↕"}</span>
    </th>
  );

  return (
    <div className="page">
      {/* Topic tabs */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", borderBottom: "1px solid var(--border)", paddingBottom: "12px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--mono)", fontWeight: "bold" }}>TOPICS:</span>
          <div className="filter-chips" style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1 }}>
            {lists.map(l => (
              <button key={l} className={`chip ${activeList === l ? "active" : ""}`} onClick={() => setActiveList(l)}>{l}</button>
            ))}

          </div>
        </div>

        {activeList === "Favorites" && (
          <div className="watchlist-toolbar" style={{ margin: 0, display: "flex", gap: "8px" }}>
            <input
              className="search-input"
              style={{ maxWidth: "260px" }}
              placeholder="Ticker symbol…"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && handleAdd()}
            />
            <button
              onClick={handleAdd}
              style={{ padding: "0 16px", background: "rgba(0,212,106,0.15)", color: "var(--green)", border: "1px solid var(--green)", borderRadius: "4px", cursor: "pointer", fontFamily: "var(--mono)", fontSize: "11px", fontWeight: "bold" }}
            >
              ADD
            </button>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 40 }}></th>
              <Header col="ticker">Symbol</Header>
              <Header col="sector">Sector</Header>
              <Header col="price" align="right">Price</Header>
              <Header col="change" align="right">Day Δ</Header>
              <th style={{ textAlign: "center" }}>Trend (1D)</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(m => {
              const isExpanded = expandedTickers.has(m.ticker);
              const qt = liveQuotes[m.ticker.toUpperCase()];
              const displayPrice    = qt ? qt.price    : m.price;
              const displayChange   = qt ? qt.change   : m.change;
              const displayUp       = qt ? qt.up       : m.up;
              const displaySparkline = qt ? qt.sparkline : m.data["1W"];
              return (
                <React.Fragment key={m.id}>
                  <tr onClick={() => toggleExpand(m.ticker)} style={{ cursor: "pointer", background: isExpanded ? "rgba(255,255,255,0.02)" : "transparent" }}>
                    <td>
                      <button
                        className="watch-star"
                        onClick={e => {
                          e.stopPropagation();
                          if (activeList === "Favorites") removeFavorite(m.ticker);
                          else toggleTopic(m.id, activeList);
                        }}
                        aria-label="Remove"
                        style={activeList === "Favorites" ? { color: "var(--red)", fontSize: "16px" } : undefined}
                      >
                        {activeList === "Favorites" ? "−" : (topicWatchlists[activeList] || []).includes(m.id) ? "★" : "☆"}
                      </button>
                    </td>
                    <td>
                      <div className="sym-cell">
                        <div className="sym-tk">{m.ticker}</div>
                        <div className="sym-name">{m.label}</div>
                      </div>
                    </td>
                    <td className="dim">{m.sector}</td>
                    <td className="num">${fmtPrice(displayPrice)}</td>
                    <td className={displayUp ? "num up" : "num down"}>{displayUp ? "▲" : "▼"} {fmtPct(displayChange)}</td>
                    <td style={{ display: "flex", justifyContent: "center" }}>
                      <Sparkline data={displaySparkline} up={displayUp} width={120} height={28} />
                    </td>
                  </tr>

                  {isExpanded && (
                    <tr style={{ background: "var(--bg1)", borderBottom: "1px solid var(--border)" }}>
                      <td colSpan={6} style={{ padding: "16px 24px" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "2fr 3fr", gap: "24px", background: "var(--bg2)", padding: "20px", border: "1px solid var(--border-bright)", borderRadius: "6px" }}>
                          <div style={{ borderRight: "1px dashed var(--border)", paddingRight: "24px", display: "flex", flexDirection: "column", minWidth: 0 }}>
                            <WatchlistChart market={m} />
                          </div>
                          <div style={{ minWidth: 0 }}>
                            <AIAnalysisPanel ticker={m.ticker} horizon="Mid-term" />
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
            {sorted.length === 0 && (
              <tr><td colSpan={6} className="empty">No tickers added — type a symbol above and press ADD.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
