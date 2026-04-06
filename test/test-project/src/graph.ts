const canvas = document.getElementById('graph') as HTMLCanvasElement
const ctx = canvas.getContext('2d')!
const dpr = window.devicePixelRatio

const canvasW = 1440
const canvasH = 1200

canvas.width = canvasW * dpr
canvas.height = canvasH * dpr
ctx.scale(dpr, dpr)

const margin = {
    top: 20*1.2,
    left: 20,
    right: 100,
    bottom: 100*1.2
}

//set default
const chartX = margin.left
const chartY = margin.top
const chartW = canvasW - margin.left - margin.right //1320
const chartH = canvasH - margin.top - margin.bottom //

const tablebg = ctx.strokeStyle = 'rgba(175, 175, 175, 0.5)'
const green = '#00ff00'

let interval = chartW/12 //110
ctx.lineWidth = 1
ctx.strokeStyle = tablebg
ctx.fillStyle = green

ctx.strokeRect(chartX, chartY, chartW, chartH)

//month
for(let x = chartX; x <= chartX + chartW; x += interval){
    ctx.beginPath()
    ctx.moveTo(x + 0.5, chartY)
    ctx.lineTo(x + 0.5, chartY + chartH)
    ctx.stroke()
}

let pricegap = 100
//price
for(let y = chartY; y <= chartY + chartH; y += pricegap){
    ctx.beginPath()
    ctx.moveTo(chartX, y + 0.5)
    ctx.lineTo(chartX + chartW, y + 0.5)
    ctx.stroke()
}

const timeLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

timeLabels.forEach((label, i) => {
  const x = chartX + (i * interval) + (interval / 2)

  ctx.fillStyle = 'rgba(175, 175, 175, 1)'
  ctx.font = '22px monospace'
  ctx.textAlign = 'center'
  ctx.fillText(label, x, margin.top + chartH + 30)
})

