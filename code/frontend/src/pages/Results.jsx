import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import api from '../services/api'
import SentimentBadge from '../components/SentimentBadge'
import ScoreBar from '../components/ScoreBar'
import SentimentPieChart from '../components/SentimentPieChart'
import XAIExplainer from '../components/XAIExplainer'
import toast from 'react-hot-toast'
import { ArrowLeft, Download } from 'lucide-react'

export default function Results() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    let cancelled = false
    const loadResult = async () => {
      try {
        const r = await api.get(`/analysis/${id}`)
        if (!cancelled) setData(r.data)
      } catch (err) {
        if (!cancelled) toast.error(err.displayMessage || 'Could not load results.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    loadResult()
    return () => { cancelled = true }
  }, [id])

  const handleDownload = async () => {
    setDownloading(true)
    try {
      const res = await api.get(`/analysis/${id}/report`, { responseType: 'blob' })
      const blob = new Blob([res.data], { type: 'application/pdf' })
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = `xsense_report_${id}.pdf`
      link.click()
      URL.revokeObjectURL(link.href)
      toast.success('Report downloaded!')
    } catch (err) {
      toast.error(err.displayMessage || 'Report generation failed.')
    } finally {
      setDownloading(false)
    }
  }

  if (loading) return <div className="loading-screen"><div className="spinner" /></div>
  if (!data) return <div className="not-found">Analysis not found. <Link to="/dashboard">Go back</Link></div>

  const scores = data.scores ?? {}
  const emotionScores = data.emotion_scores ?? {}
  const xai = data.xai ?? {}
  const displayScores = Object.keys(emotionScores).length > 0
    ? emotionScores
    : { positive: scores.positive ?? 0, negative: scores.negative ?? 0, neutral: scores.neutral ?? 0 }

  const scoreItems = Object.keys(displayScores).map(key => {
    const normalized = key.toLowerCase()
    const label = key.charAt(0).toUpperCase() + key.slice(1)
    const cls = normalized === 'happy' || normalized === 'positive'
      ? 'positive'
      : (normalized === 'sad' || normalized === 'angry' || normalized === 'fear' || normalized === 'disgust' || normalized === 'negative')
        ? 'negative'
        : 'neutral'
    return { key, label, cls, value: displayScores[key] ?? 0 }
  })

  return (
    <div className="results-page">
      {/* Header */}
      <div className="results-header">
        <Link to="/dashboard" className="btn btn-outline btn-sm">
          <ArrowLeft size={14} /> Back
        </Link>
        <div className="results-title-group">
          <h1>Sentiment Result</h1>
          <p>Analysis #{data.id} · {data.input_type?.toUpperCase()} · {new Date(data.created_at).toLocaleString()}</p>
        </div>
        <button className="btn btn-primary btn-sm" onClick={handleDownload} disabled={downloading}>
          {downloading ? <span className="spinner spinner--sm" /> : <Download size={14} />}
          {downloading ? 'Generating…' : 'Download PDF'}
        </button>
      </div>

      {/* Main sentiment */}
      <div className="result-main-card">
        <div className="result-sentiment">
          <span className="result-label">Predicted Sentiment</span>
          <SentimentBadge sentiment={data.sentiment} large />
        </div>
        <div className="result-meta">
          {data.detected_language && (
            <div className="meta-item">
              <span className="meta-key">Language Detected</span>
              <span className="meta-val">{data.detected_language}</span>
            </div>
          )}
          {data.translated_text && (
            <div className="meta-item">
              <span className="meta-key">Translated Text</span>
              <span className="meta-val meta-translated">{data.translated_text}</span>
            </div>
          )}
          {data.raw_input && (
            <div className="meta-item">
              <span className="meta-key">Input Preview</span>
              <span className="meta-val">{data.raw_input.slice(0, 300)}{data.raw_input.length > 300 ? '…' : ''}</span>
            </div>
          )}
        </div>
      </div>

      {/* Score cards */}
      <div className="score-cards">
        {scoreItems.map(({ label, key, cls, value }) => (
          <div className={`score-card score-card--${cls}`} key={key}>
            <div className="score-pct">{(value * 100).toFixed(1)}%</div>
            <div className="score-label">{label}</div>
            <ScoreBar value={value} variant={cls} />
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="charts-row">
        <div className="chart-card">
          <h3>Score Distribution</h3>
          <SentimentPieChart dataMap={displayScores} />
        </div>
        <div className="chart-card">
          <h3>Score Breakdown</h3>
          <div className="score-bars-list">
            {scoreItems.map(({ label, key, cls, value }) => (
              <ScoreBar key={key} label={label} value={value} variant={cls} showLabel />
            ))}
          </div>
        </div>
      </div>

      {/* XAI Explanation */}
      <XAIExplainer xai={xai} keyWords={data.key_words ?? []} />

      {/* Audio transcript */}
      {data.transcript && (
        <div className="result-card">
          <h3>Audio Transcript</h3>
          <p className="transcript-text">{data.transcript}</p>
        </div>
      )}

      {/* Actions */}
      <div className="result-actions">
        <Link to="/analyze" className="btn btn-outline">Analyse Another</Link>
        <button className="btn btn-primary" onClick={handleDownload} disabled={downloading}>
          {downloading ? 'Generating PDF…' : 'Download Full Report'}
        </button>
      </div>
    </div>
  )
}
