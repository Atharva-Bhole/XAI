import { Link, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    try {
      await logout()
      toast.success('Logged out.')
      navigate('/')
    } catch {
      toast.error('Logout failed.')
    }
  }

  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">
        <span className="brand-x">X</span>
        <span className="brand-sense">-Sense</span>
      </Link>

      <div className="nav-links">
        {user ? (
          <>
            <span className="nav-user">Signed in as {user.name}</span>
            <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'nav-link--active' : ''}`}>Dashboard</NavLink>
            <NavLink to="/analyze" className={({ isActive }) => `nav-link ${isActive ? 'nav-link--active' : ''}`}>Analyze</NavLink>
            <button className="btn btn-outline btn-sm" onClick={handleLogout}>Logout</button>
          </>
        ) : (
          <>
            <NavLink to="/login" className={({ isActive }) => `nav-link ${isActive ? 'nav-link--active' : ''}`}>Login</NavLink>
            <Link to="/register" className="btn btn-primary btn-sm">Register</Link>
          </>
        )}
      </div>
    </nav>
  )
}
