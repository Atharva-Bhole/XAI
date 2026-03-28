import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../services/api'
import SentimentBadge from '../components/SentimentBadge'
import SentimentPieChart from '../components/SentimentPieChart'
import SentimentBarChart from '../components/SentimentBarChart'
import { BarChart2, FileText, Zap, Clock } from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuth()
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/dashboard/stats')
      .then(r => setStats(r.data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-screen"><div className="spinner" /></div>

  const dist = stats?.sentiment_distribution ?? { positive: 0, negative: 0, neutral: 0 }
  const typeDist = stats?.input_type_distribution ?? {}

  return (
    <div className="dashboard-page">
      {/* Welcome */}
      <div className="dashboard-welcome">
        <div>
          <h1>Welcome back, {user?.name} 👋</h1>
          <p>Here is your sentiment intelligence overview.</p>
        </div>
        <Link to="/analyze" className="btn btn-primary">New Analysis</Link>
      </div>

      {/* Stat cards */}
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-icon stat-icon--blue"><BarChart2 size={22} /></div>
          <div>
            <div className="stat-value">{stats?.total_analyses ?? 0}</div>
            <div className="stat-label">Total Analyses</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon stat-icon--green"><Zap size={22} /></div>
          <div>
            <div className="stat-value">{stats?.positive_pct ?? 0}%</div>
            <div className="stat-label">Positive Rate</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon stat-icon--red"><FileText size={22} /></div>
          <div>
            <div className="stat-value">{stats?.negative_pct ?? 0}%</div>
            <div className="stat-label">Negative Rate</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon stat-icon--gray"><Clock size={22} /></div>
          <div>
            <div className="stat-value">{stats?.neutral_pct ?? 0}%</div>
            <div className="stat-label">Neutral Rate</div>
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="charts-row">
        <div className="chart-card">
          <h3>Sentiment Distribution</h3>
          <SentimentPieChart
            positive={dist.positive}
            negative={dist.negative}
            neutral={dist.neutral}
          />
        </div>
        <div className="chart-card">
          <h3>Analysis by Input Type</h3>
          <SentimentBarChart data={typeDist} />
        </div>
      </div>

      {/* Recent analyses */}
      <div className="recent-section">
        <div className="recent-header">
          <h3>Recent Analyses</h3>
          <Link to="/analyze" className="btn btn-sm btn-outline">View All</Link>
        </div>
        {stats?.recent_analyses?.length > 0 ? (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Input</th>
                  <th>Type</th>
                  <th>Language</th>
                  <th>Sentiment</th>
                  <th>Date</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_analyses.map(a => (
                  <tr key={a.id}>
                    <td>{a.id}</td>
                    <td className="td-truncate">{a.raw_input?.slice(0, 60) || '—'}</td>
                    <td><span className="badge badge--type">{a.input_type}</span></td>
                    <td>{a.detected_language || '—'}</td>
                    <td><SentimentBadge sentiment={a.sentiment} /></td>
                    <td>{new Date(a.created_at).toLocaleDateString()}</td>
                    <td>
                      <Link to={`/results/${a.id}`} className="btn btn-xs btn-outline">View</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <p>No analyses yet. <Link to="/analyze">Run your first analysis →</Link></p>
          </div>
        )}
      </div>
    </div>
  )
}
