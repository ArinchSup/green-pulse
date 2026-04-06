import { useState } from 'react'
import './dashboard.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
        <div className="overall"> 
            <div className="left">
                left
            </div>
            <div className="center">
                center
            </div>
            <div className="right">
                right
            </div>    
        </div>
        <div className="footer">
            copyright: greed-pulse

        </div>
    </>
  )
}

export default App
