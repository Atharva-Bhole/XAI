export default function ScoreBar({ value = 0, variant = 'neutral', label = '', showLabel = false }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="score-bar-wrap">
      {showLabel && <div className="score-bar-label">{label}</div>}
      <div className="score-bar-track">
        <div
          className={`score-bar-fill score-bar-fill--${variant}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && <div className="score-bar-pct">{pct}%</div>}
    </div>
  )
}
