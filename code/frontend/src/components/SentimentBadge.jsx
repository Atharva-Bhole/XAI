const CONFIG = {
  Positive: { cls: 'badge--positive', marker: '+', label: 'Positive' },
  Negative: { cls: 'badge--negative', marker: '-', label: 'Negative' },
  Neutral:  { cls: 'badge--neutral',  marker: '=', label: 'Neutral'  },
}

export default function SentimentBadge({ sentiment, large = false }) {
  const cfg = CONFIG[sentiment] ?? { cls: 'badge--neutral', marker: '?', label: sentiment ?? 'Unknown' }
  return (
    <span className={`sentiment-badge ${cfg.cls} ${large ? 'sentiment-badge--lg' : ''}`}>
      {cfg.marker} {cfg.label}
    </span>
  )
}
