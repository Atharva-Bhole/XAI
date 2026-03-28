import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Brain, Globe, BarChart2, FileText, Zap, Shield } from 'lucide-react'

const features = [
  { icon: Brain, title: 'AI-Powered Analysis', desc: 'State-of-the-art transformer model (RoBERTa) for accurate sentiment classification.' },
  { icon: Globe, title: 'Multilingual Support', desc: 'Analyse posts in English, Hindi, Marathi and 100+ other languages with auto-detection.' },
  { icon: Zap, title: 'Explainable AI (XAI)', desc: 'LIME-powered word-level explanations show exactly why a sentiment was predicted.' },
  { icon: BarChart2, title: 'Visual Reports', desc: 'Interactive pie charts, bar graphs and score distributions for instant insights.' },
  { icon: FileText, title: 'PDF Export', desc: 'Download a full branded PDF report including inputs, scores, and XAI explanations.' },
  { icon: Shield, title: 'Multimodal Input', desc: 'Analyse plain text, images, audio, video, or paste any social media URL.' },
]

export default function Home() {
  const { user } = useAuth()

  return (
    <div className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero-content">
          <div className="hero-badge">Real-Time Sentiment Intelligence</div>
          <h1 className="hero-title">
            <span className="brand-x">X</span>
            <span className="brand-sense">-Sense</span>
          </h1>
          <p className="hero-subtitle">
            Explainable multimodal sentiment analytics for multilingual social channels
          </p>
          <p className="hero-description">
            Analyze text, images, audio, and video in one workspace. Get transparent results with
            interpretable AI signals that show why each prediction was made.
          </p>
          <div className="hero-actions">
            {user ? (
              <Link to="/analyze" className="btn btn-primary btn-lg">Start Analysis</Link>
            ) : (
              <>
                <Link to="/register" className="btn btn-primary btn-lg">Create Free Account</Link>
                <Link to="/login" className="btn btn-outline btn-lg">Login</Link>
              </>
            )}
          </div>
        </div>
        <div className="hero-visual">
          <div className="sentiment-card sentiment-card--pos">
            <span className="sentiment-emoji">POS</span>
            <span>Positive — 78%</span>
          </div>
          <div className="sentiment-card sentiment-card--neg">
            <span className="sentiment-emoji">NEG</span>
            <span>Negative — 12%</span>
          </div>
          <div className="sentiment-card sentiment-card--neu">
            <span className="sentiment-emoji">NEU</span>
            <span>Neutral — 10%</span>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features-section">
        <h2 className="section-title">Everything You Need</h2>
        <p className="section-subtitle">One platform to understand sentiment across every modality and language.</p>
        <div className="features-grid">
          {features.map(({ icon: Icon, title, desc }) => (
            <div className="feature-card" key={title}>
              <div className="feature-icon"><Icon size={28} /></div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2>Ready to Understand Audience Sentiment at Scale?</h2>
        <p>Turn high-volume social chatter into clear, defensible decisions with explainable analytics.</p>
        {!user && <Link to="/register" className="btn btn-primary btn-lg">Create Free Account</Link>}
        {user && <Link to="/analyze" className="btn btn-primary btn-lg">Go to Analyzer</Link>}
      </section>
    </div>
  )
}
