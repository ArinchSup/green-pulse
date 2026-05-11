// src/App.tsx
import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { Sidebar } from "./Sidebar";
import { Overview } from "./pages/Overview";
import { Portfolio } from "./pages/Portfolio";
import { Watchlist } from "./pages/Watchlist";
import { Settings } from "./pages/Settings";
import { Login } from "./pages/Login";
import { TradeModal } from "./TradeModal";
import { MARKETS, HOLDINGS, TRANSACTIONS, NEWS, ALERTS, findMarket } from "./variable";
import type { Alert, Market, RangeKey, User } from "./types";

const STORAGE_KEY = "gp_user";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [activePage, setActivePage] = useState("overview");
  const [selectedId, setSelectedId] = useState("sp500");
  const [range, setRange] = useState<RangeKey>("1M");
  const [tradeTicker, setTradeTicker] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [watched, setWatched] = useState<Set<string>>(new Set());
  const [userWatchlists, setUserWatchlists] = useState<Record<string, string[]>>({});
  const [alerts, setAlerts] = useState<Alert[]>(ALERTS);
  const [markets] = useState<Market[]>(MARKETS);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const email = params.get("email");
    const userId = params.get("user_id");

    if (token && email && userId) {
      const nextUser: User = { token, email, userId };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(nextUser));
      setUser(nextUser);
      window.history.replaceState({}, "", window.location.pathname);
      return;
    }

    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    try {
      setUser(JSON.parse(stored));
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    fetch(`http://localhost:8080/favorites?user_id=${user.userId}`)
      .then(r => r.json())
      .then((data: { symbol: string }[]) => {
        const symbols = new Set(data.map(d => d.symbol.toUpperCase()));
        setWatched(new Set(MARKETS.filter(m => symbols.has(m.ticker.toUpperCase())).map(m => m.id)));
      })
      .catch(() => {});
  }, [user]);

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
    setWatched(new Set());
    setUserWatchlists({});
  }, []);

  const toggleWatch = useCallback(async (id: string, listName: string = "Favorites") => {
    if (!user) return;

    if (listName !== "Favorites") {
      setUserWatchlists(prev => {
        const list = prev[listName] || [];
        const next = list.includes(id) ? list.filter(x => x !== id) : [...list, id];
        return { ...prev, [listName]: next };
      });
      return;
    }

    const market = markets.find(m => m.id === id);
    if (!market) return;
    const isWatched = watched.has(id);
    const method = isWatched ? "DELETE" : "POST";
    try {
      await fetch(
        `http://localhost:8080/favorites?user_id=${user.userId}&symbol=${encodeURIComponent(market.ticker)}`,
        { method }
      );
    } catch { /* optimistic */ }
    setWatched(prev => {
      const next = new Set(prev);
      isWatched ? next.delete(id) : next.add(id);
      return next;
    });
  }, [user, markets, watched]);

  const onCreateWatchlist = useCallback((name: string) => {
    setUserWatchlists(prev => ({ ...prev, [name]: [] }));
  }, []);

  const onDeleteWatchlist = useCallback((name: string) => {
    setUserWatchlists(prev => { const n = { ...prev }; delete n[name]; return n; });
  }, []);

  if (!user) return <Login />;

  const select = (id: string) => { setSelectedId(id); setActivePage("overview"); };
  const watchlists = { Favorites: Array.from(watched), ...userWatchlists };
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
            <Watchlist
              markets={markets}
              onSelect={select}
              onTrade={setTradeTicker}
              watchlists={watchlists}
              toggleWatch={toggleWatch}
              onCreateWatchlist={onCreateWatchlist}
              onDeleteWatchlist={onDeleteWatchlist}
            />
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
