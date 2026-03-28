# X-Sense — Explainable Multimodal Real-Time Sentiment Intelligence System

X-Sense analyses social media posts (text, images, audio, video) across multiple languages and explains *why* a sentiment was predicted.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Chart.js |
| Backend | Python / Flask (REST API) |
| ML Models | HuggingFace Transformers (RoBERTa), LIME XAI |
| Database | SQLite (dev) / MySQL or PostgreSQL (prod) |
| PDF Reports | fpdf2 |
| Language | langdetect + deep_translator |
| Speech | OpenAI Whisper / SpeechRecognition |

---

## Project Structure

```
code/
├── app.py                  # Flask application factory
├── config.py               # Configuration classes
├── requirements.txt        # Python dependencies
│
├── database/
│   └── db.py               # SQLAlchemy instance
│
├── models/
│   ├── user.py             # User model
│   └── analysis.py         # Analysis model
│
├── routes/
│   ├── auth.py             # /api/auth  (register, login, logout, me)
│   ├── analysis.py         # /api/analysis  (text, image, audio, url, history)
│   └── dashboard.py        # /api/dashboard/stats
│
├── ml/
│   ├── sentiment.py        # RoBERTa sentiment model (transformer)
│   ├── explainer.py        # LIME / keyword XAI
│   ├── language_detector.py# langdetect + Devanagari heuristic
│   ├── translator.py       # deep_translator (Google)
│   ├── image_analyzer.py   # PIL + OpenCV image sentiment
│   └── audio_analyzer.py   # Whisper transcription + ffmpeg
│
├── utils/
│   ├── social_media.py     # URL fetcher (Twitter API + BeautifulSoup)
│   └── report.py           # fpdf2 PDF report generator
│
├── uploads/                # Uploaded media files (auto-created)
├── reports/                # Generated PDF reports (auto-created)
│
└── frontend/               # React + Vite application
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── context/
        │   └── AuthContext.jsx
        ├── services/
        │   └── api.js
        ├── pages/
        │   ├── Home.jsx
        │   ├── Login.jsx
        │   ├── Register.jsx
        │   ├── Dashboard.jsx
        │   ├── Analyze.jsx
        │   └── Results.jsx
        ├── components/
        │   ├── Navbar.jsx
        │   ├── Footer.jsx
        │   ├── SentimentBadge.jsx
        │   ├── ScoreBar.jsx
        │   ├── SentimentPieChart.jsx
        │   ├── SentimentBarChart.jsx
        │   └── XAIExplainer.jsx
        └── styles/
            └── index.css
```

---

## Setup & Run

### 1. Clone and set up Python virtual environment

```bash
cd code
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables (optional)

Create a `.env` file in `code/`:

```env
SECRET_KEY=your-super-secret-key
JWT_SECRET_KEY=your-jwt-secret
TWITTER_BEARER_TOKEN=your-twitter-bearer-token
```

### 4. Start the Flask API server

```bash
python app.py
```

Flask runs on **http://localhost:5000**

### 5. Install and start the React frontend

```bash
cd frontend
npm install
npm run dev
```

React runs on **http://localhost:3000**

Open **http://localhost:3000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login |
| POST | `/api/auth/logout` | Logout |
| GET  | `/api/auth/me` | Get current user |
| POST | `/api/analysis/text` | Analyse text |
| POST | `/api/analysis/image` | Analyse image file |
| POST | `/api/analysis/audio` | Analyse audio/video file |
| POST | `/api/analysis/url` | Analyse social media URL |
| GET  | `/api/analysis/history` | Paginated history |
| GET  | `/api/analysis/:id` | Get single result |
| GET  | `/api/analysis/:id/report` | Download PDF report |
| GET  | `/api/dashboard/stats` | Dashboard statistics |
| GET  | `/api/health` | Health check |

---

## Features

- **Multilingual**: Auto-detects English, Hindi, Marathi and 100+ other languages; translates to English before analysis.
- **Explainable AI**: LIME word-level importance scores show exactly which words drove the prediction.
- **Multimodal**: Text, image (colour + face detection), audio (Whisper transcription), video (ffmpeg + Whisper).
- **Visual Dashboard**: Pie chart, bar chart, score breakdowns powered by Chart.js.
- **PDF Reports**: Branded downloadable reports including input, scores, XAI explanation and score bar charts.
- **Secure**: Session-based auth, parameterised SQL (SQLAlchemy ORM), SSRF protection on URL input, file extension validation.
