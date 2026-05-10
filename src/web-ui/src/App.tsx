// src/App.tsx
import { useEffect, useState, useCallback } from "react";
import "./App.css";
import { Sidebar } from "./Sidebar";
import { Overview } from "./pages/Overview";
import { Portfolio } from "./pages/Portfolio";
import { Watchlist } from "./pages/Watchlist";
import { Settings } from "./pages/Settings";
import { Login } from "./pages/Login";
import { TradeModal } from "./TradeModal";
import {
  MARKETS, HOLDINGS, TRANSACTIONS, NEWS, ALERTS, findMarket
} from "./variable";
import type { Market, RangeKey, Alert, User } from "./types";

const STORAGE_KEY = "gp_user";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [activePage, setActivePage] = useState("overview");
  const [selectedId, setSelectedId] = useState("sp500");
  const [range, setRange] = useState<RangeKey>("1M");
  const [tradeTicker, setTradeTicker] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [watched, setWatched] = useState<Set<string>>(new Set());
  const [alerts, setAlerts] = useState<Alert[]>(ALERTS);
  const [markets, setMarkets] = useState<Market[]>(MARKETS);

  // Resolve auth from OAuth redirect params or persisted session
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const email = params.get("email");
    const userId = params.get("user_id");

    if (token && email && userId) {
      const u: User = { token, email, userId };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(u));
      setUser(u);
      window.history.replaceState({}, "", window.location.pathname);
    } else {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try { setUser(JSON.parse(stored)); } catch { localStorage.removeItem(STORAGE_KEY); }
      }
    }
  }, []);

  // Load favorites from backend whenever user changes
  useEffect(() => {
    if (!user) return;
    fetch(`http://localhost:8080/favorites?user_id=${user.userId}`)
      .then(r => r.json())
      .then((data: { symbol: string }[]) => {
        const symbols = new Set(data.map(d => d.symbol.toUpperCase()));
        setWatched(new Set(
          MARKETS.filter(m => symbols.has(m.ticker.toUpperCase())).map(m => m.id)
        ));
      })
      .catch(() => {});
  }, [user]);

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

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
    setWatched(new Set());
  }, []);

  const toggleWatch = useCallback(async (id: string) => {
    if (!user) return;
    const market = markets.find(m => m.id === id);
    if (!market) return;
    const isWatched = watched.has(id);
    const method = isWatched ? "DELETE" : "POST";
    try {
      await fetch(
        `http://localhost:8080/favorites?user_id=${user.userId}&symbol=${encodeURIComponent(market.ticker)}`,
        { method }
      );
    } catch { /* optimistic update proceeds regardless */ }
    setWatched(prev => {
      const n = new Set(prev);
      isWatched ? n.delete(id) : n.add(id);
      return n;
    });
  }, [user, markets, watched]);

  if (!user) return <Login />;

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
            <Settings alerts={alerts} setAlerts={setAlerts} user={user} onLogout={logout} />
          )}
        </div>
      </div>
      {tradeMarket && <TradeModal market={tradeMarket} onClose={() => setTradeTicker(null)} />}
    </div>
  );
}

export default App;
