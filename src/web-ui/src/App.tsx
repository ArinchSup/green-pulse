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
  const [watched, setWatched] = useState<Set<string>>(new Set(["amd", "nvda", "meta", "tsla"]));
  const [alerts, setAlerts] = useState<Alert[]>(ALERTS);
  const [markets, setMarkets] = useState<Market[]>(MARKETS);


  // Chatbot parts
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Ai parts
  const [analyses, setAnalyses] = useState<Record<string, any>>({});
  const [loadingAnalyses, setLoadingAnalyses] = useState<Set<string>>(new Set());
  const [horizons, setHorizons] = useState<Record<string, string>>({});
  const handleHorizonChange = (ticker: string, newHorizon: string) => {
    setHorizons(prev => ({ ...prev, [ticker]: newHorizon }));
    setAnalyses(prev => {
      const next = { ...prev };
      delete next[ticker]; 
      return next;
    });
  };
  
  useEffect(() => {
    const fetchAnalysis = async (ticker: string) => {
      try {
        setLoadingAnalyses(prev => new Set(prev).add(ticker));
        const realTicker = ticker === "SPX" ? "^GSPC" : ticker; 
        const currentHorizon = horizons[ticker] || "Mid-term";

        const response = await fetch("http://localhost:8000/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: realTicker, horizon: currentHorizon }) 
        });
        
        if (response.ok) {
          const data = await response.json();
          setAnalyses(prev => ({ ...prev, [ticker]: data }));
        }
      } catch (error) {
        console.error(`Failed to fetch analysis for ${ticker}:`, error);
      } finally {
        setLoadingAnalyses(prev => {
          const next = new Set(prev);
          next.delete(ticker);
          return next;
        });
      }
    };

    watched.forEach(tickerId => {
      const market = markets.find(m => m.id === tickerId);
      if (market && !analyses[market.ticker] && !loadingAnalyses.has(market.ticker)) {
        fetchAnalysis(market.ticker);
      }
    });
  }, [watched, markets, horizons]);

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
               markets={watchlistMarkets} 
               onSelect={select} 
               onTrade={setTradeTicker}
               watched={watched} 
               toggleWatch={toggleWatch} 
               analyses={analyses}
               horizons={horizons}
               onHorizonChange={handleHorizonChange}
            />
          )}
          
          {activePage === "settings" && (
            <Settings alerts={alerts} setAlerts={setAlerts} />
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
