// src/App.tsx
import { useEffect, useState } from "react";
import "./App.css";
import { Sidebar } from "./Sidebar";
import { Overview } from "./pages/Overview";
import { Portfolio } from "./pages/Portfolio";
import { Watchlist } from "./pages/Watchlist";
import { Settings } from "./pages/Settings";
import { TradeModal } from "./TradeModal";
import {
  MARKETS, HOLDINGS, TRANSACTIONS, NEWS, ALERTS, findMarket
} from "./variable";
import type { Market, RangeKey, Alert } from "./types";

function App() {
  const [activePage, setActivePage] = useState("overview");
  const [selectedId, setSelectedId] = useState("sp500");
  const [range, setRange] = useState<RangeKey>("1M");
  const [tradeTicker, setTradeTicker] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [watched, setWatched] = useState<Set<string>>(new Set(["sp500", "nvda", "btc", "eth"]));
  const [alerts, setAlerts] = useState<Alert[]>(ALERTS);
  const [markets, setMarkets] = useState<Market[]>(MARKETS);

  // Live ticking
  useEffect(() => {
    const t = setInterval(() => {
      setMarkets(prev => prev.map(m => {
        const drift = (Math.random() - 0.5) * m.base * 0.0008;
        const newPrice = parseFloat((m.price + drift).toFixed(2));
        const change = ((newPrice - m.base) / m.base) * 100;
        const lastT = m.data["1D"][m.data["1D"].length - 1].t;
        const newDay = [...m.data["1D"].slice(1), { i: m.data["1D"].length, t: lastT, value: newPrice }];
        return { ...m, price: newPrice, change: parseFloat(change.toFixed(2)), up: change >= 0,
                 data: { ...m.data, "1D": newDay } };
      }));
    }, 2200);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const toggleWatch = (id: string) => {
    setWatched(prev => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id); else n.add(id);
      return n;
    });
  };

  const select = (id: string) => { setSelectedId(id); setActivePage("overview"); };
  const watchedMarkets = markets.filter(m => watched.has(m.id));
  const watchlistMarkets = watched.size > 0 ? watchedMarkets : markets;
  const tradeMarket = tradeTicker ? findMarket(tradeTicker, markets) : null;

  return (
    <div className="app">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <div className="main">
        <div className="header">
          <div className="crumb">
            <span className="slash">/</span>
            <span className="seg">{activePage}</span>
            {activePage === "overview" && (
              <>
                <span className="slash"> / </span>
                <span>{markets.find(m => m.id === selectedId)?.ticker}</span>
              </>
            )}
          </div>
          <div className="header-right">
            <span className="clock">{now.toLocaleTimeString("en-US", { hour12: false })} UTC</span>
            <span className="market-status"><span className="dot live"></span>MARKET OPEN</span>
          </div>
        </div>
        <div className="scroll">
          {activePage === "overview" && (
            <Overview markets={markets} selectedId={selectedId} onSelect={setSelectedId}
                      range={range} setRange={setRange} news={NEWS} />
          )}
          {activePage === "portfolio" && (
            <Portfolio markets={markets} holdings={HOLDINGS} transactions={TRANSACTIONS} onTrade={setTradeTicker} />
          )}
          {activePage === "watchlist" && (
            <Watchlist markets={watchlistMarkets} onSelect={select} onTrade={setTradeTicker}
                       watched={watched} toggleWatch={toggleWatch} />
          )}
          {activePage === "settings" && (
            <Settings alerts={alerts} setAlerts={setAlerts} />
          )}
        </div>
      </div>
      {tradeMarket && <TradeModal market={tradeMarket} onClose={() => setTradeTicker(null)} />}
    </div>
  );
}

export default App;
