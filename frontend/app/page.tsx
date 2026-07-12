'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

interface PredictionResult {
  result: string;
  confidence: number;
  advice: string;
  all_probabilities?: Record<string, number>;
}

const CLASS_FULL_NAMES: Record<string, string> = {
  akiec: 'Actinic Keratosis',
  bcc: 'Basal Cell Carcinoma',
  bkl: 'Benign Keratosis',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Melanocytic Nevus',
  vasc: 'Vascular Lesion',
};

const SERIOUS_CLASSES = ['mel', 'bcc', 'akiec'];

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [introDone, setIntroDone] = useState(false);

  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIntroDone(true), 3200);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    if (showCamera) {
      navigator.mediaDevices
        .getUserMedia({ video: { facingMode: 'environment' } })
        .then((stream) => {
          streamRef.current = stream;
          if (videoRef.current) videoRef.current.srcObject = stream;
        })
        .catch(() => setCameraError("Couldn't access camera. Check browser permissions."));
    }
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [showCamera]);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG).');
      return;
    }
    setImageFile(file);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx?.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        handleFileSelect(file);
        closeCamera();
      }
    }, 'image/jpeg', 0.92);
  };

  const closeCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setShowCamera(false);
    setCameraError(null);
  };

  const handleAnalyze = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Analysis failed. Please check the server and try again.');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error && err.message.toLowerCase().includes('fetch')
          ? "Couldn't reach the analysis server. Make sure it's running."
          : err instanceof Error
          ? err.message
          : 'Something went wrong.'
      );
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async () => {
    const text = chatInput.trim();
    if (!text || chatLoading) return;

    const newMessages = [...chatMessages, { role: 'user', content: text }];
    setChatMessages(newMessages);
    setChatInput('');
    setChatLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: newMessages }),
      });
      const data = await res.json();
      setChatMessages([...newMessages, { role: 'assistant', content: data.reply }]);
    } catch {
      setChatMessages([...newMessages, { role: 'assistant', content: "Sorry, I couldn't connect. Please try again." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const reset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  const isSerious = result && SERIOUS_CLASSES.includes(result.result);
  const isNoDisease = result?.result === 'No disease found';

  return (
    <div className="page page-visible">
      <div className={`intro-overlay ${introDone ? 'intro-hide' : ''}`}>
        <h1 className="intro-title">DermEx</h1>
        <p className="intro-tagline">Your personalised skin care AI agent</p>
        <svg className="intro-curve" viewBox="0 0 200 50" fill="none">
          <path
            d="M15 10 Q 100 90 185 10"
            stroke="#17C990"
            strokeWidth="2.5"
            strokeLinecap="round"
            className="intro-curve-path"
          />
        </svg>
      </div>

      <div className="container">
        <div className="header">
          <p className="eyebrow">AI-Assisted Screening</p>
          <h1 className="title">Skin Lesion Analyzer</h1>
          <p className="subtitle">Upload or capture a close-up photo of a skin lesion for an AI-generated prediction.</p>
        </div>

        {!result && (
          <div className="fade-in" key="upload-stage">
            <div
              className={`lens ${dragActive ? 'lens-active' : ''} ${imagePreview ? 'lens-filled' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload skin lesion image"
              onKeyDown={(e) => { if (e.key === 'Enter') fileInputRef.current?.click(); }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
              {imagePreview ? (
                <img src={imagePreview} alt="Selected lesion preview" className="lens-img" />
              ) : (
                <div className="lens-placeholder">
                  <div className="lens-plus">+</div>
                  <p>Click or drop an image</p>
                </div>
              )}
              {loading && <div className="scan-line" />}
            </div>

            {loading && <p className="analyzing-label">Analyzing…</p>}

            {!imageFile && (
              <div className="camera-row">
                <button className="btn-ghost" onClick={() => setShowCamera(true)}>
                  Use camera instead
                </button>
              </div>
            )}

            {imageFile && (
              <div className="action-row">
                <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
                  {loading ? 'Analyzing…' : 'Analyze image'}
                </button>
                <button className="btn-ghost" onClick={reset} disabled={loading}>
                  Choose a different image
                </button>
              </div>
            )}
          </div>
        )}

        {error && <p className="error-text" role="alert">{error}</p>}

        {result && (
          <div className="fade-in" key="results-stage">
            <div className="results-header">
              {imagePreview && <img src={imagePreview} alt="Analyzed lesion" className="result-thumb" />}
              <div className={`badge ${isSerious ? 'badge-warn' : 'badge-ok'}`}>
                {isNoDisease ? 'No disease found' : `Predicted: ${CLASS_FULL_NAMES[result.result] || result.result}`}
              </div>
              <p className="confidence-text">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            </div>

            {result.all_probabilities && (
              <div className="card">
                {Object.entries(result.all_probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([cls, prob], i) => (
                    <div className="prob-row" key={cls} style={{ animationDelay: `${i * 60}ms` }}>
                      <div className="prob-labels">
                        <span>{CLASS_FULL_NAMES[cls] || cls}</span>
                        <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{ width: `${prob * 100}%`, background: cls === result.result ? '#17C990' : '#3A4145' }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            )}

            <div className="card advice">
              <ReactMarkdown>{result.advice}</ReactMarkdown>
            </div>

            <details className="info-panel">
              <summary>What is the ABCDE rule?</summary>
              <div className="info-body">
                <p><strong>A — Asymmetry:</strong> one half doesn't match the other.</p>
                <p><strong>B — Border:</strong> edges are irregular, ragged, or blurred.</p>
                <p><strong>C — Color:</strong> uneven shading or multiple colors in one spot.</p>
                <p><strong>D — Diameter:</strong> larger than about 6mm (pencil eraser size).</p>
                <p><strong>E — Evolving:</strong> changing in size, shape, or color over time.</p>
              </div>
            </details>

            <details className="info-panel">
              <summary>About this AI model's accuracy</summary>
              <div className="info-body">
                <p>This model was trained on the HAM10000 dataset of 10,015 dermatoscopic images and reaches about 80.7% overall accuracy across 7 lesion types on a held-out test set.</p>
                <p>Performance varies by condition — rarer conditions and visually similar lesions (like melanoma vs. common moles) are harder for the model to distinguish. This tool is for demonstration purposes and is not a substitute for professional diagnosis.</p>
              </div>
            </details>

            <div className="action-row">
              <button className="btn-ghost" onClick={reset}>Analyze another image</button>
            </div>
          </div>
        )}
      </div>

      {showCamera && (
        <div className="camera-modal" role="dialog" aria-label="Camera capture">
          <div className="camera-box">
            {cameraError ? (
              <p className="error-text">{cameraError}</p>
            ) : (
              <video ref={videoRef} autoPlay playsInline className="camera-video" />
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <div className="camera-controls">
              {!cameraError && (
                <button className="btn-primary" onClick={capturePhoto}>Capture</button>
              )}
              <button className="btn-ghost" onClick={closeCamera}>Cancel</button>
            </div>
          </div>
        </div>
      )}

<button className="chat-fab" onClick={() => setChatOpen(true)} aria-label="Open skincare chat">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M4 4h16a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H8l-4.5 4V6a2 2 0 0 1 2-2Z"
      stroke="#0B1210"
      strokeWidth="1.8"
      strokeLinejoin="round"
      fill="none"
    />
    <circle cx="8.5" cy="10.5" r="1" fill="#0B1210" />
    <circle cx="12" cy="10.5" r="1" fill="#0B1210" />
    <circle cx="15.5" cy="10.5" r="1" fill="#0B1210" />
  </svg>
</button>

      {chatOpen && (
        <div className="chat-panel">
          <div className="chat-header">
            <span>Ask DermEx</span>
            <button className="chat-close" onClick={() => setChatOpen(false)} aria-label="Close chat">×</button>
          </div>
          <div className="chat-body">
            {chatMessages.length === 0 && (
              <p className="chat-empty">Ask me anything about skin care — routines, ingredients, sun protection, and more.</p>
            )}
            {chatMessages.map((m, i) => (
              <div key={i} className={`chat-bubble ${m.role === 'user' ? 'chat-user' : 'chat-bot'}`}>
                {m.content}
              </div>
            ))}
            {chatLoading && <div className="chat-bubble chat-bot chat-typing">…</div>}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input-row">
            <input
              className="chat-input"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
              placeholder="Type a question…"
            />
            <button className="chat-send" onClick={sendChatMessage} disabled={chatLoading}>Send</button>
          </div>
        </div>
      )}

      <style jsx global>{`
        * { box-sizing: border-box; }
        body { margin: 0; background: #14181A; }
      `}</style>

      <style jsx>{`
        .page {
          background: #14181A;
          min-height: 100vh;
        }
        .intro-overlay {
          position: fixed;
          inset: 0;
          background: #14181A;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          z-index: 100;
          transition: opacity 0.7s ease, visibility 0.7s ease;
          opacity: 1;
        }
        .intro-hide {
          opacity: 0;
          visibility: hidden;
          pointer-events: none;
        }
        .intro-title {
          font-family: Georgia, serif;
          font-size: clamp(36px, 8vw, 56px);
          font-weight: 600;
          color: #EDF2F0;
          margin: 0;
          opacity: 0;
          animation: introFadeUp 0.8s ease 0.2s forwards;
        }
        .intro-tagline {
          font-family: Inter, sans-serif;
          font-size: 15px;
          color: #9AA6A0;
          margin: 0;
          opacity: 0;
          animation: introFadeUp 0.8s ease 0.6s forwards;
        }
        .intro-curve {
          width: 160px;
          height: 40px;
          margin-top: -4px;
        }
        .intro-curve-path {
          stroke-dasharray: 220;
          stroke-dashoffset: 220;
          animation: drawCurve 1s ease 0.6s forwards;
        }
        @keyframes introFadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes drawCurve {
          to { stroke-dashoffset: 0; }
        }
        .container {
          max-width: 720px;
          margin: 0 auto;
          padding: 48px 20px 80px;
        }
        .header { text-align: center; margin-bottom: 40px; }
        .eyebrow {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #5FBF9B;
          margin-bottom: 10px;
        }
        .title {
          font-family: Georgia, serif;
          font-size: clamp(28px, 6vw, 42px);
          font-weight: 600;
          color: #EDF2F0;
          letter-spacing: -0.02em;
          margin: 0;
        }
        .subtitle {
          font-family: Inter, sans-serif;
          font-size: 15px;
          color: #9AA6A0;
          margin-top: 10px;
          padding: 0 8px;
        }
        .fade-in {
          animation: fadeSlideIn 0.4s ease both;
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .lens {
          width: min(280px, 70vw);
          height: min(280px, 70vw);
          border-radius: 50%;
          margin: 0 auto;
          border: 3px dashed #3A4145;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          overflow: hidden;
          position: relative;
          background: #1E2326;
          transition: border-color 0.2s, transform 0.15s;
        }
        .lens:hover { border-color: #5FBF9B; }
        .lens:focus-visible { outline: 3px solid #17C990; outline-offset: 4px; }
        .lens-active { border-color: #17C990; transform: scale(1.02); }
        .lens-filled { border-style: solid; border-color: #17C990; }
        .lens-img { width: 100%; height: 100%; object-fit: cover; }
        .lens-placeholder { text-align: center; padding: 24px; font-family: Inter, sans-serif; }
        .lens-plus { font-size: 32px; margin-bottom: 8px; color: #3A4145; }
        .lens-placeholder p { font-size: 14px; color: #5FBF9B; margin: 0; }
        .scan-line {
          position: absolute;
          left: 4px;
          right: 4px;
          height: 3px;
          background: linear-gradient(90deg, transparent, #17C990, transparent);
          box-shadow: 0 0 12px 2px rgba(23, 201, 144, 0.5);
          animation: scanMove 1.6s ease-in-out infinite;
        }
        @keyframes scanMove {
          0% { top: 4%; }
          50% { top: 94%; }
          100% { top: 4%; }
        }
        .analyzing-label {
          text-align: center;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 13px;
          color: #17C990;
          margin-top: 12px;
        }
        .camera-row, .action-row {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          margin-top: 20px;
        }
        .btn-primary {
          font-family: Inter, sans-serif;
          font-weight: 600;
          font-size: 15px;
          padding: 14px 32px;
          border-radius: 8px;
          border: none;
          background: #17C990;
          color: #0B1210;
          cursor: pointer;
          min-width: 200px;
          transition: transform 0.1s, background 0.2s;
        }
        .btn-primary:hover:not(:disabled) { background: #21E6A8; }
        .btn-primary:active:not(:disabled) { transform: scale(0.98); }
        .btn-primary:disabled { opacity: 0.6; cursor: default; }
        .btn-primary:focus-visible { outline: 3px solid #5FBF9B; outline-offset: 2px; }
        .btn-ghost {
          font-family: Inter, sans-serif;
          font-size: 14px;
          padding: 10px 24px;
          border-radius: 8px;
          border: 1px solid #3A4145;
          background: transparent;
          color: #9AA6A0;
          cursor: pointer;
          transition: background 0.2s;
        }
        .btn-ghost:hover { background: #1E2326; }
        .btn-ghost:focus-visible { outline: 3px solid #5FBF9B; outline-offset: 2px; }
        .error-text {
          text-align: center;
          color: #FF6B5B;
          margin-top: 20px;
          font-family: Inter, sans-serif;
          font-size: 14px;
        }
        .results-header { text-align: center; margin-bottom: 24px; }
        .result-thumb {
          width: 120px;
          height: 120px;
          border-radius: 50%;
          object-fit: cover;
          margin: 0 auto 20px;
          display: block;
        }
        .badge {
          display: inline-block;
          padding: 6px 16px;
          border-radius: 999px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 13px;
          margin-bottom: 10px;
        }
        .badge-ok { background: rgba(23, 201, 144, 0.15); color: #17C990; }
        .badge-warn { background: rgba(255, 107, 91, 0.15); color: #FF6B5B; }
        .confidence-text {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 13px;
          color: #5FBF9B;
        }
        .card {
          background: #1E2326;
          padding: 20px;
          border-radius: 12px;
          border: 1px solid #2A3033;
          margin-bottom: 24px;
        }
        .prob-row {
          margin-bottom: 10px;
          animation: fadeSlideIn 0.35s ease both;
        }
        .prob-labels {
          display: flex;
          justify-content: space-between;
          font-family: Inter, sans-serif;
          font-size: 13px;
          color: #EDF2F0;
          margin-bottom: 4px;
        }
        .prob-value { font-family: 'IBM Plex Mono', monospace; color: #5FBF9B; }
        .prob-track { height: 6px; background: #262B2E; border-radius: 3px; overflow: hidden; }
        .prob-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.6s ease;
        }
        .advice {
          font-family: Inter, sans-serif;
          font-size: 15px;
          line-height: 1.6;
          color: #EDF2F0;
          padding: 28px;
        }
        .info-panel {
          background: #1E2326;
          border: 1px solid #2A3033;
          border-radius: 12px;
          padding: 16px 20px;
          margin-bottom: 16px;
          font-family: Inter, sans-serif;
        }
        .info-panel summary {
          cursor: pointer;
          font-weight: 600;
          font-size: 14px;
          color: #17C990;
          list-style: none;
        }
        .info-panel summary::-webkit-details-marker { display: none; }
        .info-panel summary::before {
          content: '+ ';
          color: #5FBF9B;
        }
        .info-panel[open] summary::before { content: '- '; }
        .info-body {
          margin-top: 12px;
          font-size: 14px;
          color: #9AA6A0;
          line-height: 1.6;
        }
        .info-body p { margin: 0 0 8px; }
        .camera-modal {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.75);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 50;
          padding: 20px;
          animation: fadeSlideIn 0.25s ease both;
        }
        .camera-box {
          background: #1E2326;
          border-radius: 16px;
          padding: 20px;
          max-width: 480px;
          width: 100%;
          border: 1px solid #2A3033;
        }
        .camera-video {
          width: 100%;
          border-radius: 12px;
          background: #000;
        }
        .camera-controls {
          display: flex;
          gap: 12px;
          justify-content: center;
          margin-top: 16px;
        }
        .chat-fab {
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: #17C990;
  border: none;
  cursor: pointer;
  box-shadow: 0 4px 16px rgba(23, 201, 144, 0.3);
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.15s, box-shadow 0.2s;
}
.chat-fab:hover {
  transform: scale(1.06);
  box-shadow: 0 6px 20px rgba(23, 201, 144, 0.45);
}
.chat-fab:active {
  transform: scale(0.96);
}
        .chat-panel {
          position: fixed;
          bottom: 92px;
          right: 24px;
          width: min(340px, 90vw);
          height: min(460px, 70vh);
          background: #1E2326;
          border: 1px solid #2A3033;
          border-radius: 16px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          z-index: 60;
          animation: fadeSlideIn 0.25s ease both;
        }
        .chat-header {
          background: #17C990;
          color: #0B1210;
          padding: 14px 16px;
          font-family: Inter, sans-serif;
          font-weight: 600;
          font-size: 14px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .chat-close {
          background: none;
          border: none;
          color: #0B1210;
          font-size: 20px;
          cursor: pointer;
          line-height: 1;
        }
        .chat-body {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .chat-empty {
          font-family: Inter, sans-serif;
          font-size: 13px;
          color: #5FBF9B;
          text-align: center;
          margin-top: 20px;
        }
        .chat-bubble {
          font-family: Inter, sans-serif;
          font-size: 13px;
          line-height: 1.5;
          padding: 10px 14px;
          border-radius: 12px;
          max-width: 85%;
          white-space: pre-wrap;
        }
        .chat-user {
          background: #17C990;
          color: #0B1210;
          align-self: flex-end;
        }
        .chat-bot {
          background: #262B2E;
          color: #EDF2F0;
          align-self: flex-start;
        }
        .chat-typing { opacity: 0.6; }
        .chat-input-row {
          display: flex;
          gap: 8px;
          padding: 12px;
          border-top: 1px solid #2A3033;
        }
        .chat-input {
          flex: 1;
          border: 1px solid #3A4145;
          border-radius: 8px;
          padding: 8px 12px;
          font-family: Inter, sans-serif;
          font-size: 13px;
          outline: none;
          background: #14181A;
          color: #EDF2F0;
        }
        .chat-input:focus { border-color: #17C990; }
        .chat-send {
          background: #17C990;
          color: #0B1210;
          border: none;
          border-radius: 8px;
          padding: 8px 16px;
          font-family: Inter, sans-serif;
          font-size: 13px;
          cursor: pointer;
        }
        .chat-send:disabled { opacity: 0.6; cursor: default; }
        @media (max-width: 480px) {
          .container { padding: 32px 16px 60px; }
          .advice { padding: 20px; font-size: 14px; }
        }
      `}</style>
    </div>
  );
}
