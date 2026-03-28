import { Pie } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend)

export default function SentimentPieChart({ positive = 0, negative = 0, neutral = 0 }) {
  const data = {
    labels: ['Positive', 'Negative', 'Neutral'],
    datasets: [
      {
        data: [
          Math.round(positive * 100),
          Math.round(negative * 100),
          Math.round(neutral * 100),
        ],
        backgroundColor: ['#16a34a', '#dc2626', '#6b7280'],
        borderColor: ['#15803d', '#b91c1c', '#4b5563'],
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
          label: ctx => ` ${ctx.label}: ${ctx.parsed}%`,
        },
      },
    },
  }

  // If all zeros show empty placeholder
  const total = positive + negative + neutral
  if (total === 0) {
    return <div className="chart-empty">No data yet</div>
  }

  return <Pie data={data} options={options} />
}
