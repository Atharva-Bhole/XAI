const CONFIG = {
  Positive: { cls: 'badge--positive', marker: '+', label: 'Positive' },
  Negative: { cls: 'badge--negative', marker: '-', label: 'Negative' },
  Neutral:  { cls: 'badge--neutral',  marker: '=', label: 'Neutral'  },
  Happy:    { cls: 'badge--positive', marker: ':)', label: 'Happy' },
  Sad:      { cls: 'badge--negative', marker: ':(', label: 'Sad' },
  Angry:    { cls: 'badge--negative', marker: '!!', label: 'Angry' },
  Calm:     { cls: 'badge--neutral',  marker: '~', label: 'Calm' },
  Fear:     { cls: 'badge--negative', marker: '!?', label: 'Fear' },
  Surprised:{ cls: 'badge--neutral',  marker: ':O', label: 'Surprised' },
  Disgust:  { cls: 'badge--negative', marker: ':/', label: 'Disgust' },
}

export default function SentimentBadge({ sentiment, large = false }) {
  const cfg = CONFIG[sentiment] ?? { cls: 'badge--neutral', marker: '?', label: sentiment ?? 'Unknown' }
  return (
    <span className={`sentiment-badge ${cfg.cls} ${large ? 'sentiment-badge--lg' : ''}`}>
      {cfg.marker} {cfg.label}
    </span>
  )
}
