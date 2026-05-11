// src/pages/Overview.tsx
import { useState, useEffect } from "react";
import type { Market, NewsItem, PricePoint, RangeKey } from "../types";
import { Sparkline, LineChart, Pill } from "../primitives";
import { fmtPrice, fmtPct } from "../format";
import { RANGE_KEYS } from "../variable";

const Stat = ({ label, value, tone }: { label: string; value: string; tone?: "up" | "down" }) => (
  <div className="stat">
    <div className="stat-label">{label}</div>
    <div className={`stat-value ${tone || ""}`}>{value}</div>
  </div>
);

export const Overview = ({
  markets, selectedId, onSelect, range, setRange, news, watchlists = {}
}: {
  markets: Market[]; selectedId: string; onSelect: (id: string) => void;
  range: RangeKey; setRange: (r: RangeKey) => void; news: NewsItem[];
  watchlists?: Record<string, string[]>;
}) => {
  const m = markets.find(x => x.id === selectedId) || markets[0];

  const [liveData, setLiveData] = useState<PricePoint[] | null>(null);
  const [fetching, setFetching] = useState(false);
  const [liveNews, setLiveNews] = useState<NewsItem[] | null>(null);

  useEffect(() => {
    setLiveData(null);
    setFetching(true);
    fetch("http://localhost:8000/chart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: m.ticker, period: range }),
    })
      .then(r => r.json())
      .then((j: { data: PricePoint[] }) => setLiveData(j.data))
      .catch(() => {})
      .finally(() => setFetching(false));
  }, [m.ticker, range]);

  useEffect(() => {
    fetch("http://localhost:8000/news", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: m.ticker }),
    })
      .then(r => r.json())
      .then((j: { news: NewsItem[] }) => setLiveNews(j.news))
      .catch(() => {});
  }, [m.ticker]);

  const [favQuotes, setFavQuotes] = useState<Record<string, { price: number; change: number; up: boolean; sparkline: PricePoint[] }>>({});

  const favTickerList = (watchlists["Favorites"] ?? []) as string[];
  useEffect(() => {
    favTickerList.forEach(ticker => {
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
          setFavQuotes(prev => ({ ...prev, [ticker.toUpperCase()]: { price, change, up: change >= 0, sparkline: j.data } }));
        })
        .catch(() => {});
    });
  }, [favTickerList.join(",")]);

  const [showFib, setShowFib] = useState(false);
  const [showSR, setShowSR] = useState(false);
  const [showDMZ, setShowDMZ] = useState(false);
  
  const calcSeries = liveData ?? m.data[range] ?? [];
  const refSeries  = liveData ? calcSeries : (m.data["1Y"]?.length > 0 ? m.data["1Y"] : calcSeries);

  let focusStartT = calcSeries.length > 0 ? calcSeries[0].t : undefined;

  if (range === "1D" && calcSeries.length > 0) {
    const lastDate = String(calcSeries[calcSeries.length - 1].t).split(" ")[0];
    const firstCandleOfDay = calcSeries.find((d: any) => String(d.t).startsWith(lastDate));
    if (firstCandleOfDay) focusStartT = firstCandleOfDay.t;
  }

  const fibHigh = refSeries.length > 0 ? Math.max(...refSeries.map((d: any) => d.value)) : 0;
  const fibLow  = refSeries.length > 0 ? Math.min(...refSeries.map((d: any) => d.value)) : 0;
  const fibRng  = fibHigh - fibLow;

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

  const P = (statHigh + statLow + last) / 3;
  const srLevels = showSR ? [
    { label: "RES2", value: P + statRng * 0.618, color: "#ff4466" },
    { label: "RES1", value: P + statRng * 0.382, color: "#ff7788" },
    { label: "SUP1", value: P - statRng * 0.382, color: "#00ee77" },
    { label: "SUP2", value: P - statRng * 0.618, color: "#00d46a" }
  ] as any : undefined;

  const hasFavorites = favTickerList.length > 0;

  // Build display items for all favorited tickers using live quotes
  const favItems = favTickerList.map(ticker => {
    const t = ticker.toUpperCase();
    const base = markets.find(m => m.ticker.toUpperCase() === t);
    const qt   = favQuotes[t];
    return {
      id:        base?.id ?? t.toLowerCase(),
      ticker:    t,
      label:     base?.label ?? t,
      price:     qt?.price  ?? base?.price  ?? 0,
      change:    qt?.change ?? base?.change ?? 0,
      up:        qt ? qt.up : (base?.up ?? false),
      sparkline: qt?.sparkline ?? base?.data["1D"] ?? [],
    };
  });

  const gainers = (() => {
    const pos = favItems.filter(x => x.change > 0).sort((a, b) => b.change - a.change);
    if (pos.length >= 4) return pos.slice(0, 4);
    const neg = favItems.filter(x => x.change <= 0).sort((a, b) => b.change - a.change); // least negative first
    return [...pos, ...neg].slice(0, 4);
  })();
  const losers = [...favItems].filter(x => x.change < 0).sort((a, b) => a.change - b.change).slice(0, 4);

  return (
    <div className="page">
      <div>
        <div className="card chart-card" style={{ minWidth: 0 }}>
          
          <div className="chart-header">
            <div>
              <div className="chart-meta">
                <span className="chart-ticker">{m.ticker}</span>
                <span className="chart-name">{m.label}</span>
                <Pill tone="neutral">{m.sector}</Pill>
              </div>
              <div className="chart-price-row">
                <div className="chart-price">${fmtPrice(last)}</div>
                <div className={periodChg >= 0 ? "chart-change-up" : "chart-change-down"}>
                  {periodChg >= 0 ? "▲" : "▼"} {fmtPct(periodChg)} <span className="chg-sub">{range}</span>
                </div>
              </div>
            </div> 

            <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
              <div className="time-buttons" style={{ background: "var(--bg1)", border: "1px solid var(--border)" }}>
                <button onClick={() => setShowFib(!showFib)} className="time-btn" style={{ color: showFib ? "#ff9800" : "var(--text-muted)" }}>FIB</button>
                <button onClick={() => setShowSR(!showSR)} className="time-btn" style={{ color: showSR ? "#00e5ff" : "var(--text-muted)" }}>R/S</button>
                <button onClick={() => setShowDMZ(!showDMZ)} className="time-btn" style={{ color: showDMZ ? "#b088f5" : "var(--text-muted)" }}>DMZ</button>
              </div>
              <div className="time-buttons">
                {RANGE_KEYS.map(t => (
                  <button key={t} onClick={() => setRange(t)} className={`time-btn ${t === range ? "active" : ""}`}>{t}</button>
                ))}
              </div>
            </div>
          </div>

          <div className="chart-container" style={{ position: "relative", width: "100%", minWidth: 0, overflow: "hidden" }}>
            {fetching && (
              <div style={{
                position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
                background: "rgba(10, 14, 10, 0.6)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 10, color: "var(--green)", fontFamily: "var(--mono)", fontSize: "12px", letterSpacing: "1px"
              }}>
                <span className="dot live" style={{ marginRight: "8px" }}></span> UPDATING CHART...
              </div>
            )}

            {calcSeries.length > 0 ? (
              <>
                 <LineChart 
                   key={range} 
                   data={calcSeries} 
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

        {hasFavorites && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "0" }}>
            <div className="card">
              <div className="card-title">Top Gainers</div>
              <div className="movers">
                {gainers.map(t => (
                  <div key={t.id} className="mover-row" onClick={() => onSelect(t.id)}>
                    <div>
                      <div className="mover-tk">{t.ticker}</div>
                      <div className="mover-name">{t.label}</div>
                    </div>
                    <Sparkline data={t.sparkline} up={t.up} width={64} height={22} />
                    <div className="mover-right">
                      <div className="mover-price">${fmtPrice(t.price)}</div>
                      <div className={t.up ? "mover-chg up" : "mover-chg down"}>{fmtPct(t.change)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-title">Top Losers</div>
              <div className="pulse-grid">
                {losers.map(t => (
                  <div key={t.id} className="pulse-cell" onClick={() => onSelect(t.id)}>
                    <div className="pulse-label">{t.ticker}</div>
                    <div className="pulse-price">${fmtPrice(t.price)}</div>
                    <Sparkline data={t.sparkline} up={t.up} width="100%" height={28} />
                    <div className={t.up ? "pulse-chg up" : "pulse-chg down"}>{fmtPct(t.change)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card news-card">
        <div className="card-title">
          <span>Newswire</span>
          <span className="card-sub">Live · {(liveNews ?? news).length} items</span>
        </div>
        <div className="news-list">
          {(liveNews ?? news).map((n, i) => (
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

