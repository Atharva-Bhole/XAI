export default function XAIExplainer({ xai = {}, keyWords = [] }) {
  const wordWeights = xai.word_weights ?? []
  const summary = xai.summary ?? ''
  const method = xai.method ?? ''

  const posWords = wordWeights.filter(w => w.weight > 0).sort((a, b) => b.weight - a.weight)
  const negWords = wordWeights.filter(w => w.weight < 0).sort((a, b) => a.weight - b.weight)

  const maxAbs = Math.max(...wordWeights.map(w => Math.abs(w.weight)), 0.001)

  return (
    <div className="xai-card">
      <div className="xai-header">
        <h3>Explainable AI</h3>
        {method && <span className="xai-method-badge">{method.toUpperCase()}</span>}
      </div>

      {summary && (
        <div className="xai-summary">
          <p>{summary}</p>
        </div>
      )}

      {wordWeights.length > 0 && (
        <div className="xai-words-section">
          {posWords.length > 0 && (
            <div className="xai-word-group">
              <h4 className="xai-group-label xai-group-label--pos">Positive Indicators</h4>
              <div className="xai-word-list">
                {posWords.slice(0, 8).map(({ word, weight }) => (
                  <div className="xai-word-row" key={word}>
                    <span className="xai-word">{word}</span>
                    <div className="xai-bar-track">
                      <div
                        className="xai-bar-fill xai-bar-fill--pos"
                        style={{ width: `${(Math.abs(weight) / maxAbs) * 100}%` }}
                      />
                    </div>
                    <span className="xai-weight xai-weight--pos">+{weight.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {negWords.length > 0 && (
            <div className="xai-word-group">
              <h4 className="xai-group-label xai-group-label--neg">Negative Indicators</h4>
              <div className="xai-word-list">
                {negWords.slice(0, 8).map(({ word, weight }) => (
                  <div className="xai-word-row" key={word}>
                    <span className="xai-word">{word}</span>
                    <div className="xai-bar-track">
                      <div
                        className="xai-bar-fill xai-bar-fill--neg"
                        style={{ width: `${(Math.abs(weight) / maxAbs) * 100}%` }}
                      />
                    </div>
                    <span className="xai-weight xai-weight--neg">{weight.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {keyWords.length > 0 && (
        <div className="xai-keywords">
          <span className="xai-kw-label">Key Words:</span>
          {keyWords.map(w => (
            <span className="xai-tag" key={w}>{w}</span>
          ))}
        </div>
      )}

      {wordWeights.length === 0 && !summary && (
        <p className="xai-empty">No explanation data available for this analysis.</p>
      )}
    </div>
  )
}
