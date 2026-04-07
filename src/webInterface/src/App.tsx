import './App.css'
import {NavItems} from './variable.ts'
import { useState } from "react"

function App() {
  const [activePage, setActivePage] = useState("overview")

  return(
    <div className="rootDiv">
      <div className="outterWrapper">

        {/*Side bar*/}
        <div className="sidebar">
          <div className="stock">
            stock
          </div>
          {
            NavItems.map(item => (
              <div className={`sidebarOP ${activePage === item.id ? 'active' : ''}`} key={item.id} onClick={() => setActivePage(item.id)}>
                {item.label}
              </div>
            ))
          }
        </div>

        {/*Main content*/}
        <div className="main">
          <div className="header">
            <div>
              <div style={{color: "#c8e6c0", fontSize: "20px", fontWeight: "600"}}>
                Text test 1, 1 2 3
              </div>
              <div style={{color: "#c8e6c0", fontSize: "20px", fontWeight: "600"}}>
                Tuesday, April 07 2026
              </div>
            </div>
            <div className="ismarket">
              market open
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App