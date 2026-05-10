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
    if (!isDbLoaded) return;

    const fetchAllRealData = async () => {
      setIsChartUpdating(true);
      const promises = markets.map(async (m) => {
        let updatedM = { ...m, data: { ...m.data } }; 
        
        try {
          const res1D = await fetch("http://localhost:8000/chart", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: m.ticker, period: "1D" })
          }).then(r => r.json());
          
          if (res1D.data && res1D.data.length > 0) {
            const latestPrice = res1D.data[res1D.data.length - 1].value;
            const openPrice = res1D.data[0].value;
            updatedM.price = latestPrice;
            updatedM.change = ((latestPrice - openPrice) / openPrice) * 100;
            updatedM.up = updatedM.change >= 0;
            updatedM.data["1D"] = res1D.data;
          }

          const res1W = await fetch("http://localhost:8000/chart", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: m.ticker, period: "1W" })
          }).then(r => r.json());
          
          if (res1W.data && res1W.data.length > 0) {
            updatedM.data["1W"] = res1W.data;
          }

          if (m.id === selectedId) {
            const resRange = await fetch("http://localhost:8000/chart", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ticker: m.ticker, period: range })
            }).then(r => r.json());
            
            let drawHistoryPeriod = "5Y"; 
            if (range === "1D") drawHistoryPeriod = "1W";      
            else if (range === "1W") drawHistoryPeriod = "1W"; 
            else if (range === "1M") drawHistoryPeriod = "1Y"; 
            else drawHistoryPeriod = "5Y";

            const resHistory = await fetch("http://localhost:8000/chart", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ticker: m.ticker, period: drawHistoryPeriod })
            }).then(r => r.json());

            const res1Y = await fetch("http://localhost:8000/chart", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ticker: m.ticker, period: "1Y" })
            }).then(r => r.json());

            if (resRange.data && resRange.data.length > 0) {
              updatedM.data[range] = resRange.data;
              if (res1Y.data) updatedM.data["1Y"] = res1Y.data; 
              (updatedM as any).chartHistory = resHistory.data || resRange.data; 
            }
          }
        } catch (e) {
          console.error(`Error fetching real data for ${m.ticker}`, e);
        }
        return updatedM;
      });

      const newMarkets = await Promise.all(promises);
      setMarkets(newMarkets);
      setIsInitialLoad(false);
      setIsChartUpdating(false);
    };

    fetchAllRealData();
  }, [selectedId, range, isDbLoaded]);

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
  const logout = () => {
    setUser({ token: "temp-token", email: "guest@local", userId: USER_ID });
    setActivePage("overview");
  };
  const tradeMarket = tradeTicker ? findMarket(tradeTicker, markets) : null;

  
  console.log("Current Analyses:", analyses);

  const handleChatSubmit = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim() !== '') {
      const text = e.currentTarget.value.trim();
      e.currentTarget.value = ""; 
      
      setChatHistory(prev => [...prev, { role: "user", content: text }]);
      setIsChatLoading(true); 

      try {
        const response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, history: chatHistory }) 
        });

        if (response.ok) {
          const data = await response.json();
          const botReply = data.reply || data.response || data.answer || (Array.isArray(data) ? data[0] : data);
          
          setChatHistory(prev => [...prev, { 
            role: "assistant", 
            content: typeof botReply === 'string' ? botReply : JSON.stringify(botReply) 
          }]);
        } else {
          setChatHistory(prev => [...prev, { role: "assistant", content: "Error: AI Server responded with an error." }]);
        }
      } catch (error) {
        setChatHistory(prev => [...prev, { role: "assistant", content: "Error: Cannot connect to AI Server." }]);
      } finally {
        setIsChatLoading(false);
      }
    }
  };

  if (isInitialLoad) {
    return (
      <div className="app" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", color: "var(--green)", fontFamily: "var(--mono)", fontSize: "14px", letterSpacing: "1px", background: "var(--bg0)" }}>
        <span className="dot live" style={{ marginRight: "10px" }}></span> 
        FETCHING LIVE MARKET DATA...
      </div>
    );
  }

  return (
    <div className="app">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <div className="main">
        <div className="header">
          <div className="crumb">
            <span className="slash">/</span>
            <span className="seg">{activePage}</span>
            {(activePage === "overview" || activePage === "search") && (
              <>
                <span className="slash"> / </span>
                <span>{markets.find(m => m.id === selectedId)?.ticker}</span>
              </>
            )}
          </div>
          
          <div style={{ flex: 1, display: "flex", justifyContent: "center", position: "relative" }}>
            <input
               type="text"
               placeholder="Search ticker or company..."
               value={searchQ}
               onChange={e => { setSearchQ(e.target.value); setShowSearchDrop(true); }}
               onFocus={() => setShowSearchDrop(true)}
               onBlur={() => setTimeout(() => setShowSearchDrop(false), 200)}
               style={{ 
                 width: "350px", background: "var(--bg1)", border: "1px solid var(--border)", 
                 color: "var(--text-primary)", padding: "8px 16px", borderRadius: "20px", 
                 fontFamily: "var(--mono)", fontSize: "12px", outline: "none" 
               }}
            />
            {showSearchDrop && searchQ && (
               <div style={{ 
                 position: "absolute", top: "40px", width: "350px", background: "var(--bg2)", 
                 border: "1px solid var(--border-bright)", borderRadius: "8px", zIndex: 100, 
                 overflow: "hidden", boxShadow: "0 8px 24px rgba(0,0,0,0.8)" 
               }}>
                  {markets.filter(m => m.ticker.toLowerCase().includes(searchQ.toLowerCase()) || m.label.toLowerCase().includes(searchQ.toLowerCase())).map(m => (
                     <div
                       key={m.id}
                       onMouseDown={() => {
                          setSelectedId(m.id);
                          setActivePage("search"); 
                          setSearchQ("");
                          setShowSearchDrop(false);
                       }}
                       style={{ padding: "12px 16px", cursor: "pointer", display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)" }}
                       onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg3)"}
                       onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                     >
                       <span style={{ color: "var(--green)", fontWeight: "bold" }}>{m.ticker}</span>
                       <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>{m.label}</span>
                     </div>
                  ))}
                  
                  {!markets.some(m => m.ticker.toLowerCase() === searchQ.toLowerCase()) && (
                     <div
                       onMouseDown={async () => {
                          const newId = searchQ.toLowerCase();
                          const newTk = searchQ.toUpperCase();
                          
                          setSearchQ("");
                          setShowSearchDrop(false);

                          try {
                             const res = await fetch("http://localhost:8000/chart", {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ ticker: newTk, period: "1D" }) 
                             });

                             if (!res.ok) {
                                alert(`Cannot find data for "${newTk}" or the stock may have been delisted`);
                                return; 
                             }

                             setMarkets(prev => [...prev, {
                                id: newId, ticker: newTk, label: newTk, sector: "Global Equity",
                                price: 0, change: 0, up: true, data: {}
                             } as any]);
                             
                             setSelectedId(newId);
                             setActivePage("search"); 

                             setTimeout(() => {
                               handleGetAiAnalysis(newId, "Mid-term");
                             }, 100);

                          } catch (err) {
                             console.error("API Error:", err);
                             alert("Error fetching data for the ticker. Please try again later.");
                          }
                       }}
                       style={{ padding: "12px 16px", cursor: "pointer", color: "var(--yellow)", borderTop: "1px solid var(--border)", fontSize: "11px", fontWeight: "bold" }}
                       onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg3)"}
                       onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                     >
                       + SEARCH GLOBAL MARKET FOR "{searchQ.toUpperCase()}"
                     </div>
                  )}
               </div>
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
                      range={range} setRange={setRange} news={NEWS} isChartUpdating={isChartUpdating}
                      watchlists={watchlists} onToggleWatch={toggleWatch} onCreateWatchlist={createWatchlist}
                      pinnedOverview={pinnedOverview} 
              onRemovePinned={(id) => {
                setPinnedOverview(prev => {
                  const next = prev.filter(x => x !== id);
                  saveToDB(watchlists, next); 
                  return next;
                });
              }}
              onPinToOverview={() => {
                setPinnedOverview(prev => {
                  const next = prev.includes(selectedId) ? prev : [...prev, selectedId];
                  saveToDB(watchlists, next);
                  return next;
                });
              }}
            />
          )}

          {activePage === "search" && (
            <Overview markets={markets} selectedId={selectedId} onSelect={setSelectedId}
                      range={range} setRange={setRange} news={NEWS} isChartUpdating={isChartUpdating}
                      isSearchMode={true} 
                      watchlists={watchlists} onToggleWatch={toggleWatch} onCreateWatchlist={createWatchlist}
                      pinnedOverview={pinnedOverview} 
                      onPinToOverview={() => setPinnedOverview(prev => prev.includes(selectedId) ? prev : [...prev, selectedId])} 
                      onGoToOverview={() => setActivePage("overview")}
            />
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
               onCreateWatchlist={createWatchlist}   
               onDeleteWatchlist={deleteWatchlist}   
               analyses={analyses} horizons={horizons} onHorizonChange={handleHorizonChange}
            />
          )}
          
          {activePage === "settings" && (
            <Settings alerts={alerts} setAlerts={setAlerts} user={user} onLogout={logout} />
          )}
        </div>
      </div>
      {tradeMarket && <TradeModal market={tradeMarket} onClose={() => setTradeTicker(null)} />}
    {/* Chatbot Button */}
      {!isChatOpen && (
        <button className="fab-button" onClick={() => setIsChatOpen(true)}>
          ★
        </button>
      )}

      <div className={`chat-sidebar ${isChatOpen ? "open" : ""}`}>
        <div className="header" style={{ borderBottom: "1px solid var(--border)", padding: "0 20px", display: "flex", justifyContent: "space-between" }}>
          <div className="crumb">
            <span className="seg" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span className="dot live"></span> PULSE ASSISTANT
            </span>
          </div>
          <button className="x-btn" onClick={() => setIsChatOpen(false)} style={{ fontSize: "28px" }}>×</button>
        </div>

        {/* Chat Area */}
        <div className="scroll" style={{ flex: 1, padding: "20px", display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "11px", fontFamily: "var(--mono)", marginTop: "10px" }}>
            -- NEW CONVERSATION --
          </div>
          
          {/* Welcome Message */}
          {chatHistory.length === 0 && (
            <div style={{ background: "var(--bg2)", padding: "12px 16px", borderRadius: "8px", border: "1px solid var(--border)", alignSelf: "flex-start", maxWidth: "90%" }}>
              <span style={{ color: "var(--green)", fontFamily: "var(--mono)", fontSize: "10px", display: "block", marginBottom: "4px" }}>SYSTEM</span>
              <span style={{ fontSize: "13px", color: "var(--text-primary)", lineHeight: "1.5" }}>
                How can I assist you with your portfolio today?
              </span>
            </div>
          )}

          {/* Chat History */}
          {chatHistory.map((msg, idx) => (
             <div key={idx} style={{ 
               background: msg.role === "user" ? "var(--green-faint)" : "var(--bg2)", 
               padding: "12px 16px", 
               borderRadius: "8px", 
               border: `1px solid ${msg.role === "user" ? "var(--green)" : "var(--border)"}`, 
               alignSelf: msg.role === "user" ? "flex-end" : "flex-start", 
               maxWidth: "90%" 
             }}>
               <span style={{ color: msg.role === "user" ? "var(--text-primary)" : "var(--green)", fontFamily: "var(--mono)", fontSize: "10px", display: "block", marginBottom: "4px", textTransform: "uppercase" }}>
                 {msg.role}
               </span>
               <span style={{ fontSize: "13px", color: "var(--text-primary)", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                 {msg.content}
               </span>
             </div>
          ))}

          {/* waiting for AI response */}
          {isChatLoading && (
            <div style={{ alignSelf: "flex-start", color: "var(--text-muted)", fontSize: "11px", fontFamily: "var(--mono)" }}>
              AI is thinking...
            </div>
          )}
        </div>

        {/* User Input */}
        <div className="chat-input-area">
          <input 
            type="text" 
            className="chat-input-box" 
            placeholder={isChatLoading ? "Wait for AI to reply..." : "Ask about stocks, setup, or trend..."}
            disabled={isChatLoading}
            onKeyDown={handleChatSubmit}
          />
        </div>
      </div>

    </div> 
  );
}

export default App;
