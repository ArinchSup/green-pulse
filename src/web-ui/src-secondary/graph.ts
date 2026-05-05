const canvas = document.getElementById('graph') as HTMLCanvasElement
const ctx = canvas.getContext('2d')!
const dpr = window.devicePixelRatio
const canvasW = 1440
const canvasH = 1200

canvas.width = canvasW * dpr
canvas.height = canvasH * dpr
ctx.scale(dpr, dpr)

const margin = { top: 20*1.2, left: 20, right: 100, bottom: 100*1.2 }

const chartX = margin.left
const chartY = margin.top
const chartW = canvasW - margin.left - margin.right
const chartH = canvasH - margin.top - margin.bottom

const tablebg = 'rgba(175, 175, 175, 0.5)'
const green = '#00ff00'
const red = '#ff0000'
const interval = chartW / 12

const timeLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const priceLabels: number[] = []

interface OHLCV {
    Date: string
    Open: number
    High: number
    Low: number
    Close: number
    Volume: number
}

let points: {x: number, y: number, data: OHLCV}[] = []
let chartData: OHLCV[] = []
let mapY: (price: number) => number

function draw(crosshair?: {x: number, y: number, data: OHLCV}) {
    ctx.clearRect(0, 0, canvasW, canvasH)
    ctx.lineWidth = 1
    ctx.strokeStyle = tablebg
    ctx.strokeRect(chartX, chartY, chartW, chartH)

    for(let x = chartX; x <= chartX + chartW; x += interval){
        ctx.beginPath()
        ctx.strokeStyle = tablebg
        ctx.moveTo(x + 0.5, chartY)
        ctx.lineTo(x + 0.5, chartY + chartH)
        ctx.stroke()
    }

    priceLabels.forEach(price => {
    const y = mapY(price)
        ctx.beginPath()
        ctx.strokeStyle = tablebg
        ctx.lineWidth = 1
        ctx.moveTo(chartX, y + 0.5)
        ctx.lineTo(chartX + chartW, y + 0.5)
        ctx.stroke()

        ctx.fillStyle = 'rgb(255, 255, 255)'
        ctx.font = '18px "IBM Plex Mono"'
        ctx.textAlign = 'left'
        ctx.fillText(price.toFixed(2), chartX + chartW + 8, y + 5)
    })

    timeLabels.forEach((label, i) => {
        const x = chartX + (i * interval) + (interval / 2)
        ctx.fillStyle = 'rgb(255, 255, 255)'
        ctx.font = '18px "IBM Plex Mono"'
        ctx.textAlign = 'center'
        ctx.fillText(label, x, margin.top + chartH + 30)
    })

    if(points.length > 0) {
        ctx.beginPath()
        ctx.strokeStyle = green
        ctx.lineWidth = 1.5
        points.forEach((p, i) => {
            if(i === 0) ctx.moveTo(p.x, p.y)
            else ctx.lineTo(p.x, p.y)
        })
        ctx.stroke()
    }

    // crosshair
    if(crosshair) {
        ctx.setLineDash([5, 5])
        ctx.lineWidth = 1

        ctx.beginPath()
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.moveTo(crosshair.x + 0.5, chartY)
        ctx.lineTo(crosshair.x + 0.5, chartY + chartH)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(chartX, crosshair.y + 0.5)
        ctx.lineTo(chartX + chartW, crosshair.y + 0.5)
        ctx.stroke()

        ctx.beginPath()
        ctx.fillStyle = red
        ctx.arc(crosshair.x, crosshair.y, 4, 0, Math.PI * 2)
        ctx.fill()
        ctx.setLineDash([])

        const ohlcv = [
            { label: 'O', value: crosshair.data.Open },
            { label: 'H', value: crosshair.data.High },
            { label: 'L', value: crosshair.data.Low },
            { label: 'C', value: crosshair.data.Close },
            { label: 'V', value: crosshair.data.Volume },
        ]

        ctx.font = '18px "IBM Plex Mono"'
        ctx.textAlign = 'left'

        ohlcv.forEach((item, i) => {
            ctx.fillStyle = 'rgb(255, 255, 255)'
            ctx.fillText(item.label, chartX + 10, chartY + 30 + (i * 24))
            ctx.fillStyle = 'rgb(255, 255, 255)'
            ctx.fillText(item.value.toFixed(2), chartX + 30, chartY + 30 + (i * 24))
        })
        const date = new Date(crosshair.data.Date)
        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

        ctx.fillStyle = 'rgba(193, 0, 0, 0.55)'
        ctx.fillRect(crosshair.x - 30, chartY + chartH + 5, 60, 22)

        ctx.fillStyle = 'rgb(255, 255, 255)'
        ctx.font = '18px "IBM Plex Mono"'
        ctx.textAlign = 'center'
        ctx.fillText(dateStr, crosshair.x, chartY + chartH + 20)
    }
}

async function main() {
    const res = await fetch('/fake_stock_2026.json')
    chartData = await res.json()

    const closes = chartData.map(d => d.Close)
    const minPrice = Math.min(...closes)
    const maxPrice = Math.max(...closes)

    function mapX(index: number): number {
        return chartX + (index / (chartData.length - 1)) * chartW
    }

    mapY = (price: number): number => {
        return chartY + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH
    }

    points = chartData.map((d, i) => ({
        x: mapX(i),
        y: mapY(d.Close),
        data: d
    }))

    const priceStep = 5
    const priceStart = Math.floor(minPrice / priceStep) * priceStep
    const priceEnd = Math.ceil(maxPrice / priceStep) * priceStep
    for(let p = priceStart; p <= priceEnd; p += priceStep) {
        priceLabels.push(p)
    }
    draw()
}

main()

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect()
    const mouseX = (e.clientX - rect.left) * (canvasW / rect.width)

    if(points.length === 0) return

    const closest = points.reduce((prev, curr) =>
        Math.abs(curr.x - mouseX) < Math.abs(prev.x - mouseX) ? curr : prev
    )

    draw({x: closest.x, y: closest.y, data: closest.data})
})

canvas.addEventListener('mouseleave', () => {
    draw()
})