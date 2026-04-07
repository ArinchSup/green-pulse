import { useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Fonts ─────────────────────────────────────────────────────────────────────
const fontLink = document.createElement("link");
fontLink.href = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap";
fontLink.rel = "stylesheet";
document.head.appendChild(fontLink);

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  bg0: "#0a0e0a", bg1: "#0d1210", bg2: "#111a14", bg3: "#162019",
  border: "#1e3028", borderBright: "#2a4a35",
  green: "#00d46a", greenDim: "#00a050", greenFaint: "#0d2318",
  accent: "#39ff8e",
  textPrimary: "#c8e6c0", textSecondary: "#5a8a6a", textMuted: "#2e5040",
  red: "#ff4466", yellow: "#ffcc44", blue: "#44aaff", purple: "#bb88ff",
};

// ── Mock stock data ───────────────────────────────────────────────────────────
const PORTFOLIO = [
  { ticker: "NVDA", name: "NVIDIA Corp",      sector: "Tech",      shares: 12,  avgCost: 480.00, price: 875.20,  change: +2.34, changeP: +0.27 },
  { ticker: "AAPL", name: "Apple Inc",         sector: "Tech",      shares: 25,  avgCost: 152.00, price: 189.45,  change: -1.20, changeP: -0.63 },
  { ticker: "MSFT", name: "Microsoft Corp",    sector: "Tech",      shares: 8,   avgCost: 310.00, price: 415.80,  change: +3.80, changeP: +0.92 },
  { ticker: "BTC-X", name: "BTC Tracker",     sector: "Crypto",    shares: 0.5, avgCost: 42000,  price: 67340.0, change: +1240, changeP: +1.87 },
  { ticker: "SPY",  name: "S&P 500 ETF",       sector: "ETF",       shares: 10,  avgCost: 420.00, price: 528.60,  change: +0.90, changeP: +0.17 },
  { ticker: "TSLA", name: "Tesla Inc",         sector: "EV",        shares: 15,  avgCost: 220.00, price: 178.30,  change: -4.50, changeP: -2.46 },
];

const genCandles = (base, n = 60) => {
  let p = base;
  return Array.from({ length: n }, (_, i) => {
    const o = p;
    const move = (Math.random() - 0.48) * base * 0.025;
    p = Math.max(base * 0.5, p + move);
    const h = Math.max(o, p) + Math.random() * base * 0.008;
    const l = Math.min(o, p) - Math.random() * base * 0.008;
    return { i, open: o, close: p, high: h, low: l, v: Math.random() * 100 + 20 };
  });
};

const CHART_DATA = {
  NVDA:  genCandles(480),  AAPL: genCandles(152),
  MSFT:  genCandles(310),  "BTC-X": genCandles(42000),
  SPY:   genCandles(420),  TSLA: genCandles(220),
};

const NEWS = [
  { ts: "14:32", tag: "NVDA",  sentiment: "bull", headline: "Jensen Huang teases Blackwell Ultra GPU at GTC 2025" },
  { ts: "14:18", tag: "AAPL",  sentiment: "bear", headline: "iPhone shipments in China fall 9% amid competition" },
  { ts: "13:55", tag: "MSFT",  sentiment: "bull", headline: "Azure AI revenue up 31% QoQ, beats analyst estimates" },
  { ts: "13:40", tag: "BTC-X", sentiment: "bull", headline: "Spot BTC ETF inflows hit $842M in single session" },
  { ts: "13:12", tag: "TSLA",  sentiment: "bear", headline: "Q1 deliveries miss expectations by 7%, shares slide" },
  { ts: "12:58", tag: "SPY",   sentiment: "bull", headline: "Fed signals rate cuts possible in H2 2025" },
  { ts: "12:30", tag: "NVDA",  sentiment: "bull", headline: "NVDA added to Dow Jones Industrial Average" },
];

const NAV_ITEMS = [
  { id: "overview",   label: "Overview",   icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg> },
  { id: "portfolio",  label: "Portfolio",  icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> },
  { id: "watchlist",  label: "Watchlist",  icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> },
  { id: "screener",   label: "Screener",   icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg> },
  { id: "news",       label: "News Feed",  icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 0-2 2zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6z"/></svg> },
  { id: "settings",   label: "Settings",   icon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg> },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = (n, dec = 2) => n.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
const fmtUSD = (n) => "$" + fmt(n);
const pnl = (s) => (s.price - s.avgCost) * s.shares;
const pnlP = (s) => ((s.price - s.avgCost) / s.avgCost) * 100;
const mktVal = (s) => s.price * s.shares;

const totalValue = PORTFOLIO.reduce((a, s) => a + mktVal(s), 0);
const totalPnL   = PORTFOLIO.reduce((a, s) => a + pnl(s), 0);
const totalCost  = PORTFOLIO.reduce((a, s) => a + s.avgCost * s.shares, 0);
const totalPnLP  = (totalPnL / totalCost) * 100;

// ── Sub-components ────────────────────────────────────────────────────────────
const Dot = ({ up }) => (
  <span style={{
    display: "inline-block", width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
    background: up ? C.green : C.red,
    boxShadow: `0 0 5px ${up ? C.green : C.red}88`,
  }} />
);

const Badge = ({ children, color = C.green }) => (
  <span style={{
    fontFamily: "'JetBrains Mono'", fontSize: 10, fontWeight: 600,
    color, border: `1px solid ${color}33`, background: `${color}11`,
    borderRadius: 3, padding: "1px 6px", letterSpacing: "0.04em",
  }}>{children}</span>
);

const Card = ({ children, style }) => (
  <div style={{
    background: C.bg2, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16, ...style,
  }}>{children}</div>
);

const CardTitle = ({ children, extra }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, fontWeight: 700, color: C.textSecondary, letterSpacing: "0.12em", textTransform: "uppercase" }}>{children}</span>
    {extra}
  </div>
);

const ChartTip = ({ active, payload, prefix = "$" }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.bg3, border: `1px solid ${C.borderBright}`, borderRadius: 4, padding: "4px 10px", fontFamily: "'JetBrains Mono'", fontSize: 11, color: C.green }}>
      {prefix}{fmt(payload[0].value)}
    </div>
  );
};

// Sparkline from candle close prices
const Spark = ({ data, color, height = 40 }) => {
  const closes = data.map((d, i) => ({ i, v: d.close }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={closes} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id={`sg${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5}
          fill={`url(#sg${color})`} dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
};

// ── Main ──────────────────────────────────────────────────────────────────────
export default function StockDashboard() {
  const [activeNav, setActiveNav] = useState("overview");
  const [selected, setSelected] = useState("NVDA");
  const [prices, setPrices] = useState(() => Object.fromEntries(PORTFOLIO.map(s => [s.ticker, s.price])));
  const [chartData, setChartData] = useState(CHART_DATA);
  const [logs, setLogs] = useState(NEWS);
  const [time, setTime] = useState(new Date());

  // tick prices
  useEffect(() => {
    const t = setInterval(() => {
      setPrices(prev => {
        const next = { ...prev };
        PORTFOLIO.forEach(s => {
          next[s.ticker] = Math.max(s.price * 0.5, prev[s.ticker] * (1 + (Math.random() - 0.499) * 0.003));
        });
        return next;
      });
      setTime(new Date());
    }, 2000);
    return () => clearInterval(t);
  }, []);

  // tick chart
  useEffect(() => {
    const t = setInterval(() => {
      setChartData(prev => {
        const next = { ...prev };
        PORTFOLIO.forEach(s => {
          const arr = prev[s.ticker];
          const last = arr[arr.length - 1];
          const newClose = Math.max(last.close * 0.5, last.close * (1 + (Math.random() - 0.499) * 0.015));
          next[s.ticker] = [...arr.slice(1), {
            i: last.i + 1, open: last.close, close: newClose,
            high: Math.max(last.close, newClose) * (1 + Math.random() * 0.005),
            low: Math.min(last.close, newClose) * (1 - Math.random() * 0.005),
            v: Math.random() * 100 + 20,
          }];
        });
        return next;
      });
    }, 3000);
    return () => clearInterval(t);
  }, []);

  const sel = PORTFOLIO.find(s => s.ticker === selected);
  const selPrice = prices[selected];
  const selData = chartData[selected];
  const selPnL = (selPrice - sel.avgCost) * sel.shares;
  const selPnLP = ((selPrice - sel.avgCost) / sel.avgCost) * 100;
  const selUp = selPrice >= sel.avgCost;
  const selCloses = selData.map((d, i) => ({ i, v: d.close }));

  const totalVal = PORTFOLIO.reduce((a, s) => a + prices[s.ticker] * s.shares, 0);
  const totalPL  = PORTFOLIO.reduce((a, s) => a + (prices[s.ticker] - s.avgCost) * s.shares, 0);
  const totalPLP = (totalPL / totalCost) * 100;

  const ts = `${String(time.getHours()).padStart(2,"0")}:${String(time.getMinutes()).padStart(2,"0")}:${String(time.getSeconds()).padStart(2,"0")}`;

  // sector breakdown
  const sectors = PORTFOLIO.reduce((acc, s) => {
    acc[s.sector] = (acc[s.sector] || 0) + prices[s.ticker] * s.shares;
    return acc;
  }, {});
  const sectorArr = Object.entries(sectors).map(([k, v]) => ({ name: k, v, pct: (v / totalVal) * 100 }))
    .sort((a, b) => b.v - a.v);

  return (
    <div style={{ display: "flex", height: "100vh", background: C.bg0, fontFamily: "'IBM Plex Sans', sans-serif", color: C.textPrimary, overflow: "hidden" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        @keyframes tickUp { 0%{color:${C.accent};text-shadow:0 0 8px ${C.accent}} 100%{color:${C.green}} }
        @keyframes tickDn { 0%{color:#ff88aa;text-shadow:0 0 8px ${C.red}} 100%{color:${C.red}} }
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:${C.bg1}}
        ::-webkit-scrollbar-thumb{background:${C.borderBright};border-radius:2px}
        *{box-sizing:border-box;margin:0;padding:0}
        tr:hover td{background:${C.greenFaint}!important;cursor:pointer}
        button:hover{opacity:.85}
      `}</style>

      {/* ── Sidebar ── */}
      <aside style={{ width: 220, flexShrink: 0, background: C.bg1, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", paddingBottom: 16 }}>
        {/* Logo */}
        <div style={{ padding: "20px 20px 18px", borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 6, background: `linear-gradient(135deg,${C.green},${C.greenDim})`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 0 16px ${C.green}44` }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={C.bg0} strokeWidth="2.5">
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>
              </svg>
            </div>
            <div>
              <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 13, fontWeight: 700, color: C.accent, letterSpacing: "0.02em" }}>MKTSCAN</div>
              <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em" }}>PORTFOLIO v1.0</div>
            </div>
          </div>
        </div>

        {/* Market status */}
        <div style={{ margin: "12px 12px 4px", background: C.greenFaint, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: C.green, boxShadow: `0 0 6px ${C.green}`, animation: "pulse 2s ease-in-out infinite" }} />
          <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.green }}>Market Open · {ts}</span>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: "8px 0", overflowY: "auto" }}>
          {NAV_ITEMS.map(item => {
            const active = activeNav === item.id;
            return (
              <button key={item.id} onClick={() => setActiveNav(item.id)} style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "9px 20px", background: active ? `${C.green}14` : "transparent", border: "none", borderLeft: `2px solid ${active ? C.green : "transparent"}`, color: active ? C.green : C.textSecondary, cursor: "pointer", fontSize: 13, fontWeight: active ? 500 : 400, transition: "all .15s", letterSpacing: "0.02em" }}>
                <span style={{ opacity: active ? 1 : 0.6 }}>{item.icon}</span>{item.label}
              </button>
            );
          })}
        </nav>

        {/* P&L summary */}
        <div style={{ padding: "0 12px" }}>
          <div style={{ background: C.bg3, border: `1px solid ${C.border}`, borderRadius: 6, padding: "10px 12px" }}>
            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", marginBottom: 8 }}>TOTAL P&L</div>
            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 20, fontWeight: 700, color: totalPL >= 0 ? C.green : C.red, animation: "tickUp .3s ease", letterSpacing: "-0.02em" }}>
              {totalPL >= 0 ? "+" : ""}{fmtUSD(totalPL)}
            </div>
            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 11, color: totalPL >= 0 ? C.greenDim : C.red, marginTop: 2 }}>
              {totalPLP >= 0 ? "▲" : "▼"} {Math.abs(totalPLP).toFixed(2)}% all time
            </div>
            <div style={{ height: 1, background: C.border, margin: "10px 0" }} />
            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, marginBottom: 4 }}>PORTFOLIO VALUE</div>
            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 14, fontWeight: 600, color: C.textPrimary }}>{fmtUSD(totalVal)}</div>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Topbar */}
        <header style={{ height: 52, flexShrink: 0, borderBottom: `1px solid ${C.border}`, background: C.bg1, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 12, color: C.textMuted }}>/</span>
            <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 12, color: C.textPrimary, textTransform: "capitalize" }}>{activeNav}</span>
          </div>
          {/* Ticker tape */}
          <div style={{ display: "flex", gap: 20, overflow: "hidden" }}>
            {PORTFOLIO.slice(0, 4).map(s => {
              const p = prices[s.ticker];
              const up = p >= s.price;
              return (
                <div key={s.ticker} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 11, color: C.textMuted }}>{s.ticker}</span>
                  <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 11, color: up ? C.green : C.red, transition: "color .3s" }}>
                    {s.ticker === "BTC-X" ? fmtUSD(p) : fmtUSD(p)}
                  </span>
                </div>
              );
            })}
          </div>
          <div style={{ width: 30, height: 30, borderRadius: "50%", background: `linear-gradient(135deg,${C.green},${C.greenDim})`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'JetBrains Mono'", fontSize: 12, fontWeight: 700, color: C.bg0, boxShadow: `0 0 10px ${C.green}44` }}>K</div>
        </header>

        {/* Content */}
        <div style={{ flex: 1, overflow: "auto", padding: 20 }}>

          {/* ── Top stat cards ── */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
            {[
              { label: "Portfolio Value",   value: fmtUSD(totalVal),                color: C.green,  sub: `${PORTFOLIO.length} positions` },
              { label: "Total P&L",         value: (totalPL>=0?"+":"")+fmtUSD(totalPL), color: totalPL>=0?C.green:C.red, sub: `${totalPLP>=0?"▲":"▼"} ${Math.abs(totalPLP).toFixed(2)}%` },
              { label: "Day Gain",          value: "+$1,240.00",                     color: C.green,  sub: "▲ 0.82% today" },
              { label: "Buying Power",      value: "$8,420.00",                      color: C.blue,   sub: "Cash available" },
            ].map(s => (
              <Card key={s.label} style={{ position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", top: 0, right: 0, width: 60, height: 60, background: `radial-gradient(circle at 100% 0%, ${s.color}18, transparent 70%)` }} />
                <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 8 }}>{s.label}</div>
                <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 22, fontWeight: 700, color: s.color, letterSpacing: "-0.02em", lineHeight: 1 }}>{s.value}</div>
                <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.textMuted, marginTop: 6 }}>{s.sub}</div>
              </Card>
            ))}
          </div>

          {/* ── Selected stock chart + detail ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1.8fr 1fr", gap: 12, marginBottom: 16 }}>
            {/* Chart */}
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 18, fontWeight: 700, color: C.accent }}>{selected}</span>
                    <Badge color={selUp ? C.green : C.red}>{selUp ? "▲" : "▼"} {Math.abs(selPnLP).toFixed(2)}%</Badge>
                  </div>
                  <div style={{ fontFamily: "'IBM Plex Sans'", fontSize: 12, color: C.textSecondary, marginTop: 2 }}>{sel.name}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 22, fontWeight: 700, color: selUp ? C.green : C.red }}>{fmtUSD(selPrice)}</div>
                  <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.textMuted }}>avg cost {fmtUSD(sel.avgCost)}</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={selCloses} margin={{ top: 4, right: 0, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="selGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={selUp ? C.green : C.red} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={selUp ? C.green : C.red} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={["auto","auto"]} tick={{ fontFamily: "'JetBrains Mono'", fontSize: 9, fill: C.textMuted }} />
                  <Tooltip content={<ChartTip />} />
                  <ReferenceLine y={sel.avgCost} stroke={C.yellow} strokeDasharray="4 3" strokeWidth={1} />
                  <Area type="monotone" dataKey="v" stroke={selUp ? C.green : C.red} strokeWidth={1.8}
                    fill="url(#selGrad)" dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                {["1D","1W","1M","3M","1Y","ALL"].map(r => (
                  <button key={r} style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, padding: "3px 8px", background: r==="1M"?`${C.green}22`:"transparent", border: `1px solid ${r==="1M"?C.green:C.border}`, borderRadius: 3, color: r==="1M"?C.green:C.textMuted, cursor: "pointer" }}>{r}</button>
                ))}
              </div>
            </Card>

            {/* Position detail */}
            <Card>
              <CardTitle>Position Detail</CardTitle>
              <div style={{ display: "grid", gap: 0 }}>
                {[
                  ["Shares",       sel.shares],
                  ["Avg Cost",     fmtUSD(sel.avgCost)],
                  ["Mkt Value",    fmtUSD(selPrice * sel.shares)],
                  ["Cost Basis",   fmtUSD(sel.avgCost * sel.shares)],
                  ["Unrealized P&L", (selPnL>=0?"+":"")+fmtUSD(selPnL)],
                  ["Return",       (selPnLP>=0?"▲ ":"▼ ")+Math.abs(selPnLP).toFixed(2)+"%"],
                  ["Sector",       sel.sector],
                  ["Weight",       ((selPrice*sel.shares/totalVal)*100).toFixed(1)+"%"],
                ].map(([label, val], i) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${C.border}44` }}>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 11, color: C.textMuted }}>{label}</span>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 11, color: label==="Unrealized P&L" || label==="Return" ? (selPnL>=0?C.green:C.red) : C.textPrimary, fontWeight: 500 }}>{val}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* ── Holdings table + news + sectors ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 0.7fr", gap: 12 }}>

            {/* Holdings */}
            <Card>
              <CardTitle extra={<Badge>{PORTFOLIO.length} positions</Badge>}>Holdings</CardTitle>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["", "Ticker", "Price", "Change", "P&L", "Value", "60d"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "4px 6px 8px", fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", fontWeight: 400 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PORTFOLIO.map((s, i) => {
                      const p = prices[s.ticker];
                      const pl = pnl({ ...s, price: p });
                      const plp = pnlP({ ...s, price: p });
                      const up = p >= s.avgCost;
                      const isActive = selected === s.ticker;
                      return (
                        <tr key={s.ticker} onClick={() => setSelected(s.ticker)} style={{ borderBottom: `1px solid ${C.border}33`, background: isActive ? `${C.green}0d` : "transparent", borderLeft: isActive ? `2px solid ${C.green}` : "2px solid transparent", cursor: "pointer" }}>
                          <td style={{ padding: "7px 6px" }}><Dot up={up} /></td>
                          <td style={{ padding: "7px 4px" }}>
                            <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 12, fontWeight: 600, color: isActive ? C.accent : C.textPrimary }}>{s.ticker}</div>
                            <div style={{ fontFamily: "'IBM Plex Sans'", fontSize: 9, color: C.textMuted }}>{s.name.split(" ").slice(0, 2).join(" ")}</div>
                          </td>
                          <td style={{ padding: "7px 4px", fontFamily: "'JetBrains Mono'", fontSize: 11, color: C.textPrimary }}>{s.ticker === "BTC-X" ? "$"+fmt(p,0) : fmtUSD(p)}</td>
                          <td style={{ padding: "7px 4px", fontFamily: "'JetBrains Mono'", fontSize: 11, color: up ? C.green : C.red }}>{up ? "▲" : "▼"} {Math.abs(((p - s.price) / s.price) * 100).toFixed(2)}%</td>
                          <td style={{ padding: "7px 4px", fontFamily: "'JetBrains Mono'", fontSize: 11, color: pl>=0 ? C.green : C.red }}>{pl>=0?"+":""}{fmtUSD(pl)}</td>
                          <td style={{ padding: "7px 4px", fontFamily: "'JetBrains Mono'", fontSize: 11, color: C.textSecondary }}>{s.ticker==="BTC-X"?"$"+fmt(p*s.shares,0):fmtUSD(p * s.shares)}</td>
                          <td style={{ padding: "7px 4px", width: 60 }}><Spark data={chartData[s.ticker]} color={up ? C.green : C.red} height={28} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* News */}
            <Card style={{ display: "flex", flexDirection: "column" }}>
              <CardTitle extra={<span style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.green, display: "flex", alignItems: "center", gap: 4 }}><span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: C.green, animation: "pulse 2s infinite" }} />LIVE</span>}>
                News Feed
              </CardTitle>
              <div style={{ flex: 1, overflowY: "auto", maxHeight: 300 }}>
                {logs.map((l, i) => (
                  <div key={i} style={{ padding: "8px 0", borderBottom: `1px solid ${C.border}33`, opacity: Math.max(0.3, 1 - i * 0.04) }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                      <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted }}>{l.ts}</span>
                      <Badge color={l.sentiment === "bull" ? C.green : C.red}>{l.tag}</Badge>
                      <Badge color={l.sentiment === "bull" ? C.green : C.red}>{l.sentiment === "bull" ? "▲ BULL" : "▼ BEAR"}</Badge>
                    </div>
                    <div style={{ fontFamily: "'IBM Plex Sans'", fontSize: 11, color: C.textSecondary, lineHeight: 1.5 }}>{l.headline}</div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Sector weights */}
            <Card>
              <CardTitle>Sector Weights</CardTitle>
              {sectorArr.map(s => (
                <div key={s.name} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.textSecondary }}>{s.name}</span>
                    <span style={{ fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.green }}>{s.pct.toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 4, background: C.bg3, borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${s.pct}%`, background: `linear-gradient(90deg,${C.green},${C.accent})`, boxShadow: `0 0 6px ${C.green}66`, borderRadius: 2, transition: "width .6s ease" }} />
                  </div>
                </div>
              ))}
              <div style={{ height: 1, background: C.border, margin: "12px 0 10px" }} />
              <div style={{ fontFamily: "'JetBrains Mono'", fontSize: 9, color: C.textMuted, marginBottom: 8, letterSpacing: "0.1em" }}>ALLOCATION</div>
              <ResponsiveContainer width="100%" height={90}>
                <BarChart data={sectorArr} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fontFamily: "'JetBrains Mono'", fontSize: 8, fill: C.textMuted }} />
                  <YAxis hide />
                  <Tooltip content={({ active, payload }) => active && payload?.length ? (
                    <div style={{ background: C.bg3, border: `1px solid ${C.borderBright}`, borderRadius: 4, padding: "4px 8px", fontFamily: "'JetBrains Mono'", fontSize: 10, color: C.green }}>{payload[0].value.toFixed(1)}%</div>
                  ) : null} />
                  <Bar dataKey="pct" fill={C.green} opacity={0.8} radius={[2,2,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

          </div>
        </div>
      </main>
    </div>
  );
}