// src/App.tsx
import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { Sidebar } from "./Sidebar";
import { Overview } from "./pages/Overview";
import { Watchlist } from "./pages/Watchlist";
import { Settings } from "./pages/Settings";
import { Login } from "./pages/Login";
import { TradeModal } from "./TradeModal";
import { MARKETS, NEWS, findMarket } from "./variable";
import type { Market, RangeKey, User } from "./types";

const STORAGE_KEY = "gp_user";

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [activePage, setActivePage] = useState("overview");
  const [selectedId, setSelectedId] = useState("sp500");
  const [range, setRange] = useState<RangeKey>("1M");
  const [tradeTicker, setTradeTicker] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [favTickers, setFavTickers] = useState<string[]>([]);
  const [userWatchlists, setUserWatchlists] = useState<Record<string, string[]>>({});
  const [markets] = useState<Market[]>(MARKETS);
  const [theme, setThemeState] = useState<string>(() => localStorage.getItem("gp_theme") ?? "green");

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
        setFavTickers(data.map(d => d.symbol.toUpperCase()));
      })
      .catch(() => {});
  }, [user]);

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const setTheme = (t: string) => {
    const el = document.documentElement;
    el.classList.remove("theme-terminal", "theme-blue");
    if (t !== "green") el.classList.add(`theme-${t}`);
    localStorage.setItem("gp_theme", t);
    setThemeState(t);
  };

  useEffect(() => {
    const el = document.documentElement;
    el.classList.remove("theme-terminal", "theme-blue");
    if (theme !== "green") el.classList.add(`theme-${theme}`);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
    setFavTickers([]);
    setUserWatchlists({});
  }, []);

  const addFavorite = useCallback(async (ticker: string) => {
    if (!user) return;
    const sym = ticker.toUpperCase().trim();
    if (!sym || favTickers.includes(sym)) return;
    try {
      await fetch(`http://localhost:8080/favorites?user_id=${user.userId}&symbol=${encodeURIComponent(sym)}`, { method: "POST" });
    } catch {}
    setFavTickers(prev => [...prev, sym]);
  }, [user, favTickers]);

  const removeFavorite = useCallback(async (ticker: string) => {
    if (!user) return;
    const sym = ticker.toUpperCase();
    try {
      await fetch(`http://localhost:8080/favorites?user_id=${user.userId}&symbol=${encodeURIComponent(sym)}`, { method: "DELETE" });
    } catch {}
    setFavTickers(prev => prev.filter(t => t !== sym));
  }, [user]);

  const toggleTopic = useCallback((id: string, listName: string) => {
    if (!user || listName === "Favorites") return;
    setUserWatchlists(prev => {
      const list = prev[listName] || [];
      const next = list.includes(id) ? list.filter(x => x !== id) : [...list, id];
      return { ...prev, [listName]: next };
    });
  }, [user]);


  const isMarketOpen = (() => {
    const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
    const day = et.getDay();
    if (day === 0 || day === 6) return false;
    const mins = et.getHours() * 60 + et.getMinutes();
    return mins >= 570 && mins < 960; // 9:30–16:00
  })();

  if (!user) return <Login />;

  const select = (id: string) => { setSelectedId(id); setActivePage("overview"); };
  const watchlists = { Favorites: favTickers, ...userWatchlists };
  const tradeMarket = tradeTicker ? findMarket(tradeTicker, markets) : null;

  return (
    <div className="app">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <div className="main">
        <div className="header">
          <div className="crumb">/{activePage}</div>
          <div className="header-right">
            <span className="clock">{now.toLocaleTimeString("en-US", { hour12: false })} UTC</span>
            <span className="market-status">
                <span className={`dot ${isMarketOpen ? "live" : "closed"}`}></span>
                {isMarketOpen ? "MARKET OPEN" : "MARKET CLOSED"}
              </span>
          </div>
        </div>

        <div className="scroll">
          {activePage === "overview" && (
            <Overview markets={markets} selectedId={selectedId} onSelect={setSelectedId}
                      range={range} setRange={setRange} news={NEWS}
                      watchlists={watchlists} />
          )}
          {activePage === "watchlist" && (
            <Watchlist
              markets={markets}
              favTickers={favTickers}
              addFavorite={addFavorite}
              removeFavorite={removeFavorite}
              topicWatchlists={userWatchlists}
              toggleTopic={toggleTopic}
            />
          )}
          {activePage === "settings" && (
            <Settings user={user} onLogout={logout} theme={theme} setTheme={setTheme} />
          )}
        </div>
      </div>
      {tradeMarket && <TradeModal market={tradeMarket} onClose={() => setTradeTicker(null)} />}
    </div>
  );
}

export default App;
