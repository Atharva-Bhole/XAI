import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import api from '../services/api'
import toast from 'react-hot-toast'
import { Lock, KeyRound, ArrowLeft } from 'lucide-react'

export default function ResetPassword() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!token) {
      toast.error('Invalid or missing reset token.')
      return
    }
    if (newPassword.length < 6) {
      toast.error('Password must be at least 6 characters.')
      return
    }
    if (newPassword !== confirmPassword) {
      toast.error('Passwords do not match.')
      return
    }

    setLoading(true)
    try {
      const res = await api.post('/auth/reset-password', {
        token,
        new_password: newPassword
      })
      toast.success(res.message || 'Password reset successfully! Please log in.')
      navigate('/login')
    } catch (err) {
      toast.error(err.displayMessage || 'Password reset failed or token expired.')
    } finally {
      setLoading(false)
    }
  }

  if (!token) {
    return (
      <div className="auth-page">
        <div className="auth-card" style={{ textAlign: 'center', padding: '40px 20px' }}>
          <KeyRound size={48} color="#ef4444" style={{ margin: '0 auto 16px' }} />
          <h2 style={{ color: '#f8fafc', marginBottom: '8px' }}>Invalid Reset Link</h2>
          <p style={{ color: '#94a3b8', marginBottom: '24px' }}>
            No reset token was found in the URL. Please request a new password reset link.
          </p>
          <Link to="/forgot-password" className="btn btn-primary btn-block" style={{ textDecoration: 'none' }}>
            Request New Link
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Set New Password</h2>
          <p>Please enter your new password below.</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="newPassword">New Password</label>
            <input
              type="password"
              id="newPassword"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="••••••••"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm New Password</label>
            <input
              type="password"
              id="confirmPassword"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>

          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? <span className="spinner spinner--sm" /> : 'Reset Password'}
          </button>
        </form>

        <div className="auth-footer">
          Remembered your password? <Link to="/login">Log in</Link>
        </div>
      </div>
    </div>
  )
}
