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
  const [range, setRange] = useState<RangeKey>("1M");
  const [tradeTicker, setTradeTicker] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [selectedId, setSelectedId] = useState("nvda"); // เปลี่ยนจาก "sp500" เป็น "nvda"
  const [watched, setWatched] = useState<Set<string>>(new Set(["nvda", "tsla", "aapl", "msft"]));
  const [alerts, setAlerts] = useState<Alert[]>(ALERTS);
  const [markets, setMarkets] = useState<Market[]>(MARKETS);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [isChartUpdating, setIsChartUpdating] = useState(false);

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
  
  // 🌟 ปิดระบบ AI ชั่วคราวตรงนี้ โดยการใส่ /* ครอบไว้ด้านบน และ */ ปิดท้ายบล็อก
  /* useEffect(() => {
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
  */

  // Live ticking
  // 🌟 โค้ดดึงข้อมูลจริง (อัปเกรด: ดึงข้อมูลจริงของหุ้น "ทุกตัว" ในระบบ)
  useEffect(() => {
    const fetchAllRealData = async () => {
      setIsChartUpdating(true);
      // ใช้ Promise.all เพื่อให้ยิง API ดึงข้อมูลทุกตัวพร้อมกัน (ไม่กระตุก)
      const promises = markets.map(async (m) => {
        let updatedM = { ...m, data: { ...m.data } }; // ก๊อปปี้ข้อมูลเดิมไว้เตรียมทับด้วยของจริง
        
        try {
          // 1. ดึงข้อมูล 1D (เพื่อเอาราคาล่าสุดวันนี้ และ % เปลี่ยนแปลงรายวันแบบเป๊ะๆ)
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

          // 2. ดึงข้อมูล 1W (เพื่อวาดกราฟเส้นเล็กๆ ตรงเมนู Top Movers และ Watchlist)
          const res1W = await fetch("http://localhost:8000/chart", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: m.ticker, period: "1W" })
          }).then(r => r.json());
          
          if (res1W.data && res1W.data.length > 0) {
            updatedM.data["1W"] = res1W.data;
          }

          // 3. ดึงกราฟใหญ่ (ตาม Range ที่เลือก เช่น 1M, 3M) ให้เฉพาะหุ้นที่กำลังเปิดดูอยู่เท่านั้น
          if (m.id === selectedId) {
            const resRange = await fetch("http://localhost:8000/chart", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ticker: m.ticker, period: range })
            }).then(r => r.json());
            
            // 🌟 1. ใช้ตัวพิมพ์ใหญ่เท่านั้น และใช้ 5Y เป็นค่าสูงสุดแทน MAX เพราะ API รู้จักแค่นี้ครับ
            let historyPeriod = "5Y"; 
            if (range === "1D") historyPeriod = "1W";      
            else if (range === "1W") historyPeriod = "3M"; 
            else if (range === "1M") historyPeriod = "1Y"; 
            else if (range === "3M") historyPeriod = "5Y";

            const resHistory = await fetch("http://localhost:8000/chart", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ticker: m.ticker, period: historyPeriod })
            }).then(r => r.json());

            if (resRange.data && resRange.data.length > 0) {
              updatedM.data[range] = resRange.data;
              // 🌟 ยัดข้อมูลในอดีตเข้าไปในตัวแปรแบบเนียนๆ
              (updatedM as any).chartHistory = resHistory.data || resRange.data; 
            }
          }
        } catch (e) {
          console.error(`Error fetching real data for ${m.ticker}`, e);
        }
        return updatedM;
      });

      // รอให้โหลดครบทุกตัว แล้วอัปเดตหน้าเว็บทีเดียว
      const newMarkets = await Promise.all(promises);
      setMarkets(newMarkets);
      setIsInitialLoad(false);
      setIsChartUpdating(false);
    };

    fetchAllRealData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, range]);

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
                      range={range} setRange={setRange} news={NEWS} 
                      isChartUpdating={isChartUpdating} // 🌟 3. ส่งค่าไปให้หน้า Overview
            />
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
