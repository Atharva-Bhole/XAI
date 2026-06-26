import { useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import toast from 'react-hot-toast'
import { Mail, ArrowLeft, CheckCircle2 } from 'lucide-react'

export default function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [sent, setSent] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!email) {
      toast.error('Please enter your registered email address.')
      return
    }

    setLoading(true)
    try {
      const res = await api.post('/auth/forgot-password', { email })
      toast.success(res.message || 'Reset link sent!')
      setSent(true)
    } catch (err) {
      toast.error(err.displayMessage || 'Failed to send reset email.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Forgot Password</h2>
          <p>Enter your email address and we'll send you a secure link to reset your password.</p>
        </div>

        {sent ? (
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <CheckCircle2 size={48} color="#10b981" style={{ margin: '0 auto 16px' }} />
            <h3 style={{ marginBottom: '8px', color: '#f8fafc' }}>Check Your Inbox</h3>
            <p style={{ color: '#94a3b8', fontSize: '0.95rem', lineHeight: 1.5, marginBottom: '24px' }}>
              If an account exists for <strong>{email}</strong>, we have sent a password reset link. Please check your email and spam folder.
            </p>
            <Link to="/login" className="btn btn-primary btn-block" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <ArrowLeft size={18} /> Return to Login
            </Link>
          </div>
        ) : (
          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <div style={{ position: 'relative' }}>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
                  required
                  autoFocus
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
              {loading ? <span className="spinner spinner--sm" /> : 'Send Reset Link'}
            </button>
          </form>
        )}

        <div className="auth-footer">
          Remembered your password? <Link to="/login">Log in</Link>
        </div>
      </div>
    </div>
  )
}
