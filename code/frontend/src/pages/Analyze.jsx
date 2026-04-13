import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import toast from 'react-hot-toast'
import { Type, Image, Link2, Mic, Upload } from 'lucide-react'

const TABS = [
  { id: 'text',  label: 'Text',        icon: Type },
  { id: 'image', label: 'Image',       icon: Image },
  { id: 'url',   label: 'Social Link', icon: Link2 },
  { id: 'audio', label: 'Audio/Video', icon: Mic },
]

export default function Analyze() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('text')
  const [loading, setLoading] = useState(false)

  // Text tab state
  const [text, setText] = useState('')

  // URL tab state
  const [url, setUrl] = useState('')

  // File tab states
  const [imageFile, setImageFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [imagePrev, setImagePrev] = useState(null)

  const imageRef = useRef()
  const audioRef = useRef()

  const handleImageChange = e => {
    const f = e.target.files[0]
    if (!f) return
    setImageFile(f)
    setImagePrev(URL.createObjectURL(f))
  }

  const handleAudioChange = e => {
    const f = e.target.files[0]
    if (f) setAudioFile(f)
  }

  const handleSubmit = async e => {
    e.preventDefault()
    setLoading(true)

    try {
      let res
      if (activeTab === 'text') {
        if (!text.trim()) { toast.error('Please enter some text.'); setLoading(false); return }
        res = await api.post('/analysis/text', { text })
      } else if (activeTab === 'image') {
        if (!imageFile) { toast.error('Please select an image.'); setLoading(false); return }
        const fd = new FormData(); fd.append('image', imageFile)
        res = await api.post('/analysis/image', fd)
      } else if (activeTab === 'url') {
        if (!url.trim()) { toast.error('Please enter a URL.'); setLoading(false); return }
        res = await api.post('/analysis/url', { url })
      } else if (activeTab === 'audio') {
        if (!audioFile) { toast.error('Please select an audio or video file.'); setLoading(false); return }
        const fd = new FormData(); fd.append('audio', audioFile)
        res = await api.post('/analysis/audio', fd)
      }
      toast.success('Analysis complete!')
      navigate(`/results/${res.data.id}`)
    } catch (err) {
      toast.error(err.displayMessage || 'Analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="analyze-page">
      <div className="analyze-header">
        <h1>Analyse Content</h1>
        <p>Choose a method below and let X-Sense analyse the sentiment.</p>
      </div>

      {/* Tab bar */}
      <div className="tab-bar">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`tab-btn ${activeTab === id ? 'tab-btn--active' : ''}`}
            onClick={() => setActiveTab(id)}
            type="button"
          >
            <Icon size={16} />
            <span>{label}</span>
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="analyze-form">

        {/* TEXT */}
        {activeTab === 'text' && (
          <div className="input-panel">
            <label className="input-label">Paste or type your text</label>
            <textarea
              className="text-input"
              rows={8}
              placeholder={"e.g. This phone is very bad!\nor\nहा चित्रपट खूप छान आहे"}
              value={text}
              onChange={e => setText(e.target.value)}
              maxLength={10000}
            />
            <div className="char-count">{text.length} / 10,000</div>
          </div>
        )}

        {/* IMAGE */}
        {activeTab === 'image' && (
          <div className="input-panel">
            <label className="input-label">Upload an image</label>
            <div
              className="dropzone"
              onClick={() => imageRef.current.click()}
            >
              {imagePrev ? (
                <img src={imagePrev} alt="preview" className="img-preview" />
              ) : (
                <>
                  <Upload size={36} className="dropzone-icon" />
                  <p>Click to select or drag & drop</p>
                  <span className="dropzone-hint">PNG, JPG, GIF, WEBP (max 500 MB)</span>
                </>
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              ref={imageRef}
              style={{ display: 'none' }}
              onChange={handleImageChange}
            />
            {imagePrev && (
              <button
                type="button"
                className="btn btn-xs btn-outline"
                onClick={() => { setImageFile(null); setImagePrev(null) }}
              >
                Remove Image
              </button>
            )}
          </div>
        )}

        {/* URL */}
        {activeTab === 'url' && (
          <div className="input-panel">
            <label className="input-label">Paste a social media URL</label>
            <input
              type="url"
              className="url-input"
              placeholder="https://twitter.com/user/status/12345"
              value={url}
              onChange={e => setUrl(e.target.value)}
            />
            <p className="input-hint">
              Supports Twitter/X, and any public webpage. The system will extract and analyse the text content.
            </p>
          </div>
        )}

        {/* AUDIO/VIDEO */}
        {activeTab === 'audio' && (
          <div className="input-panel">
            <label className="input-label">Upload audio or video</label>
            <div
              className="dropzone"
              onClick={() => audioRef.current.click()}
            >
              <Mic size={36} className="dropzone-icon" />
              <p>{audioFile ? audioFile.name : 'Click to select a file'}</p>
              <span className="dropzone-hint">MP3, WAV, MP4, AVI, MOV, MKV (max 500 MB)</span>
            </div>
            <input
              type="file"
              accept="audio/*,video/*"
              ref={audioRef}
              style={{ display: 'none' }}
              onChange={handleAudioChange}
            />
            <p className="input-hint">
              Audio is transcribed to text, then analysed for sentiment. Video audio track is extracted automatically.
            </p>
          </div>
        )}

        <button type="submit" className="btn btn-primary btn-lg btn-submit" disabled={loading}>
          {loading ? (
            <><span className="spinner spinner--sm" /> Analysing…</>
          ) : (
            '🔍 Analyse Sentiment'
          )}
        </button>
      </form>
    </div>
  )
}
