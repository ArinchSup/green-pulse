const genPrices = (base: number, count = 60) => {
    let p = base
    return Array.from({ length: count }, (_, i) => {
        const move = (Math.random() - 0.48) * base * 0.015
        p = Math.max(base * 0.7, p + move)
        return {i, value: parseFloat(p.toFixed(2))}
    })
}

export const NavItems = [
    { id: "overview", label: "Overview"},
    { id: "portfolio", label: "Portfolio"},
    { id: "watchlist", label: "Watchlist"},
    { id: "settings", label: "Settings"},
]

export const MARKET = [
    { id: "sp500", label: "S&P 500", ticker: "SPX", price: 4500.25, change: "+1.2%", up: true, data: genPrices(4500)},
    { id: "nasdaq", label: "NASDAQ", ticker: "IXIC", price: 15000.75, change: "-0.8%", up: false, data: genPrices(15000)},
    { id: "dxy", label: "DXY", ticker: "DXY", price: 95.50, change: "+0.5%", up: true, data: genPrices(95.5)},
    { id: "gold", label: "Gold", ticker: "XAU", price: 1800.00, change: "-0.3%", up: false, data: genPrices(1800)},
    { id: "oil", label: "Oil", ticker: "WTI", price: 70.25, change: "+2.1%", up: true, data: genPrices(70.25)},
    { id: "btc", label: "Bitcoin", ticker: "BTC", price: 30000.00, change: "+5.0%", up: true, data: genPrices(30000)},
    { id: "eth", label: "Ethereum", ticker: "ETH", price: 2000.00, change: "+3.5%", up: true, data: genPrices(2000)},
]