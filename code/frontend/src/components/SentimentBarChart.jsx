import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function SentimentBarChart({ data = {} }) {
  const labels = Object.keys(data)
  const values = Object.values(data)

  if (labels.length === 0) {
    return <div className="chart-empty">No data yet</div>
  }

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Analyses',
        data: values,
        backgroundColor: ['#3b82f6', '#16a34a', '#f59e0b', '#8b5cf6', '#ec4899'],
        borderRadius: 6,
      },
    ],
  }

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} analyses` } },
    },
    scales: {
      x: { ticks: { color: '#cbd5e1' }, grid: { color: '#334155' } },
      y: { ticks: { color: '#cbd5e1', stepSize: 1 }, grid: { color: '#334155' }, beginAtZero: true },
    },
  }

  return <Bar data={chartData} options={options} />
}
