import { Pie } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend)

const COLOR_MAP = {
  Happy: ['#16a34a', '#15803d'],
  Sad: ['#2563eb', '#1d4ed8'],
  Angry: ['#dc2626', '#b91c1c'],
  Calm: ['#6b7280', '#4b5563'],
  Fear: ['#7c3aed', '#6d28d9'],
  Surprised: ['#f59e0b', '#d97706'],
  Disgust: ['#059669', '#047857'],
  Positive: ['#16a34a', '#15803d'],
  Negative: ['#dc2626', '#b91c1c'],
  Neutral: ['#6b7280', '#4b5563'],
}

const FALLBACK_COLORS = [
  ['#0ea5e9', '#0284c7'],
  ['#16a34a', '#15803d'],
  ['#f59e0b', '#d97706'],
  ['#8b5cf6', '#7c3aed'],
  ['#ec4899', '#db2777'],
  ['#14b8a6', '#0f766e'],
  ['#ef4444', '#dc2626'],
]

function getSliceColors(label, index) {
  const normalized = String(label ?? '').trim().toLowerCase()
  const canonical = normalized.charAt(0).toUpperCase() + normalized.slice(1)
  return COLOR_MAP[label] || COLOR_MAP[canonical] || FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

export default function SentimentPieChart({ dataMap = {} }) {
  const labels = Object.keys(dataMap)
  const rawValues = labels.map(label => Number(dataMap[label] ?? 0))
  const maxValue = rawValues.length ? Math.max(...rawValues) : 0
  const isRatio = maxValue <= 1
  const values = rawValues.map(v => (isRatio ? Math.round(v * 100) : Math.round(v)))
  const bg = labels.map((label, index) => getSliceColors(label, index)[0])
  const border = labels.map((label, index) => getSliceColors(label, index)[1])

  const data = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: bg,
        borderColor: border,
        borderWidth: 2,
      },
    ],
  }

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'bottom', labels: { color: '#e2e8f0', font: { size: 13 } } },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.label}: ${ctx.parsed}${isRatio ? '%' : ''}`,
        },
      },
    },
  }

  // If all zeros show empty placeholder
  const total = values.reduce((acc, v) => acc + v, 0)
  if (total === 0) {
    return <div className="chart-empty">No data yet</div>
  }

  return <Pie data={data} options={options} />
}
