import './App.css'
import { useState } from "react"
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import greenPulseLogo from './assets/greenPulseLogo.svg'
import shayLogo from './assets/shay.png'
import {Option} from './component.tsx'
import {MARKET} from './variable.ts'

function App() {
  const [activePage, setActivePage] = useState("overview")

  return(
    <div className="rootDiv">
      <div className="outterWrapper">

        {/*Side bar*/}
        <div className="sidebar">
          <div className="logoContainer">
            <img src={shayLogo} alt="Green Pulse Logo" className="logoImage"/>
          </div>
          <Option activePage={activePage} setActivePage={setActivePage} />
        </div>

        {/*Main content*/}
        <div className="main">
          <div className="header">
            <div>
              <div style={{color: "#c8e6c0", fontSize: "20px", fontWeight: "600", paddingLeft: "20px"}}>
                /{activePage}
              </div>
            </div>
            <div className="ismarket">
              market open
            </div>
          </div>
          {/* OVERVIEW */}
          <div className="overview">
            {/* BIG CHART */}
            <div className="main-chart-card">
              <div className="chart-header">
                <div>
                  <div className="chart-title">S&P 500 — SPX</div>
                  <div className="chart-price">${MARKET[0].price.toLocaleString()}</div>
                  <div className={MARKET[0].up ? "chart-change-up" : "chart-change-down"}>
                    {MARKET[0].up ? "▲" : "▼"} {MARKET[0].change}
                  </div>
                </div>
                <div className="time-buttons">
                  {["1D","1W","1M","3M","1Y"].map(t => (
                    <button key={t} className={`time-btn ${t === "1M" ? "active" : ""}`}>
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={MARKET[0].data}>
                  <defs>
                    <linearGradient id="spxGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#00d46a" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00d46a" stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="i" hide />
                  <YAxis domain={["auto","auto"]} hide />
                  <Tooltip
                    contentStyle={{ background: "#162019", border: "1px solid #2a4a35", borderRadius: 4 }}
                    labelStyle={{ display: "none" }}
                    itemStyle={{ color: "#00d46a", fontSize: 12 }}
                    formatter={(value) => {
                      const numericValue = typeof value === 'number' ? value : Number(value ?? 0)
                      return [`$${numericValue.toLocaleString()}`, ""]
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#00d46a"
                    strokeWidth={2}
                    fill="url(#spxGrad)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

          </div>
        </div>
      </div>
    </div>
  )
}

export default App