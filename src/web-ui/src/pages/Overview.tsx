// src/pages/Overview.tsx
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

export const Overview = ({ markets, selectedId, onSelect, range, setRange, news }: {
  markets: Market[]; selectedId: string; onSelect: (id: string) => void;
  range: RangeKey; setRange: (r: RangeKey) => void; news: NewsItem[];
}) => {
  const m = markets.find(x => x.id === selectedId) || markets[0];
  const series = m.data[range];
  const high = Math.max(...series.map(d => d.value));
  const low  = Math.min(...series.map(d => d.value));
  const open = series[0].value;
  const last = series[series.length - 1].value;
  const periodChg = ((last - open) / open) * 100;
  const indices = markets.filter(x => x.sector === "Index" || x.sector === "Currency");
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
        <div className="card chart-card">
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
            </div>
            <div className="time-buttons">
              {RANGE_KEYS.map(t => (
                <button key={t} onClick={() => setRange(t)} className={`time-btn ${t === range ? "active" : ""}`}>{t}</button>
              ))}
            </div>
          </div>
          <LineChart data={series} up={periodChg >= 0} height={300} showVolume />
          <div className="chart-stats">
            <Stat label="Open" value={`$${fmtPrice(open)}`} />
            <Stat label="High" value={`$${fmtPrice(high)}`} />
            <Stat label="Low"  value={`$${fmtPrice(low)}`} />
            <Stat label={`${range} Δ`} value={fmtPct(periodChg)} tone={periodChg >= 0 ? "up" : "down"} />
            <Stat label="Range" value={`$${fmtPrice(high - low)}`} />
            <Stat label="Volatility" value={`${(((high - low) / open) * 100).toFixed(2)}%`} />
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
              {indices.slice(0, 4).map(t => (
                <div key={t.id} className="pulse-cell" onClick={() => onSelect(t.id)}>
                  <div className="pulse-label">{t.ticker}</div>
                  <div className="pulse-price">{fmtPrice(t.price)}</div>
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
