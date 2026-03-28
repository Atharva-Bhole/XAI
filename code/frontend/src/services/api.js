import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

// Strip content-type for FormData (let browser set boundary automatically)
api.interceptors.request.use(config => {
  if (config.data instanceof FormData) {
    delete config.headers['Content-Type']
  }
  return config
})

// Central error unwrapping
api.interceptors.response.use(
  res => res,
  err => {
    const message = err.response?.data?.error || err.message || 'An unexpected error occurred.'
    err.displayMessage = message
    return Promise.reject(err)
  }
)

export default api
