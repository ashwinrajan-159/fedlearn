import { useState, useEffect } from 'react'
import './index.css'

function App() {
  const [metrics, setMetrics] = useState({ round: 0, workers: 0, status: 'IDLE' })
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    const connect = () => {
      ws = new WebSocket('ws://localhost:8080/dashboard');

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'METRICS') {
            setMetrics({
              round: data.round,
              workers: data.workers,
              status: data.status
            });
          }
        } catch (e) {
          console.error("Failed to parse websocket message", e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setMetrics({ round: 0, workers: 0, status: 'DISCONNECTED' });
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  return (
    <div className="glass-panel">
      <header>
        <h1>FedLearn Coordinator</h1>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: 10, height: 10, borderRadius: '50%', 
            background: connected ? '#10b981' : '#ef4444', 
            marginRight: 8,
            boxShadow: `0 0 10px ${connected ? '#10b981' : '#ef4444'}`
          }} />
          <span style={{ color: '#94a3b8', fontWeight: 600 }}>
            {connected ? 'WS CONNECTED' : 'WS DISCONNECTED'}
          </span>
        </div>
      </header>

      <div>
        <p style={{ color: '#cbd5e1', fontSize: '1.1rem' }}>
          Live view of the global model aggregation server.
        </p>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Federated Round</div>
          <div className="stat-value">{metrics.round}</div>
          <div className={`status-badge status-${metrics.status}`}>
            {metrics.status}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Active Workers</div>
          <div className="stat-value">{metrics.workers}</div>
          <div style={{ color: '#94a3b8', fontSize: '0.9rem', marginTop: '0.5rem' }}>
            Browser nodes computing gradients
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
