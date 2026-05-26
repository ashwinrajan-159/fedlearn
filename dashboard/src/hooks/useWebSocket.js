import { useState, useEffect, useRef } from 'react';

/**
 * Custom React hook for robust WebSocket connection and real-time state tracking.
 * Establishes auto-reconnecting listeners for coordinator metrics.
 * 
 * @param {string} url - WebSocket URL to connect to.
 * @param {number} maxPoints - Maximum historical data points to retain for charting.
 */
export function useWebSocket(url = 'ws://localhost:8080/dashboard', maxPoints = 100) {
  const [connected, setConnected] = useState(false);
  const [metrics, setMetrics] = useState({
    workers: 0,
    shards: 1,
    loss: null,
    accuracy: null,
    round: 0,
    status: 'DISCONNECTED',
    bufferFill: 0,
    bufferK: 5,
  });
  const [history, setHistory] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!url) return;
    let reconnectTimeout = null;

    const connect = () => {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        setConnected(true);
        setMetrics(prev => ({ ...prev, status: 'CONNECTED' }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'METRICS' || data.workers !== undefined) {
            const workers = data.workers !== undefined ? data.workers : 0;
            const shards = data.shards !== undefined ? data.shards : 1;
            const loss = data.valLoss !== undefined ? data.valLoss : (data.loss !== undefined ? data.loss : null);
            const accuracy = data.valAccuracy !== undefined ? data.valAccuracy : (data.accuracy !== undefined ? data.accuracy : null);
            const round = data.round !== undefined ? data.round : 0;
            const status = data.status || 'RUNNING';
            const bufferFill = data.bufferFill !== undefined ? data.bufferFill : 0;
            const bufferK = data.bufferK !== undefined ? data.bufferK : 5;
            const timestamp = data.timestamp || Date.now();

            const newMetric = {
              workers,
              shards,
              loss,
              accuracy,
              round,
              status,
              bufferFill,
              bufferK,
              timestamp,
            };

            setMetrics(newMetric);

            // Append to chart history if we receive valid performance metrics
            if (loss !== null || accuracy !== null || history.length === 0) {
              setHistory(prev => {
                const formattedTime = new Date(timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                });
                
                const updated = [...prev, {
                  ...newMetric,
                  timeStr: formattedTime,
                  // Fallbacks for chart data to ensure safe plotting
                  displayLoss: loss !== null ? Number(loss.toFixed(4)) : null,
                  displayAccuracy: accuracy !== null ? Number((accuracy * 100).toFixed(1)) : null,
                }];

                if (updated.length > maxPoints) {
                  return updated.slice(updated.length - maxPoints);
                }
                return updated;
              });
            }
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current.onclose = () => {
        setConnected(false);
        setMetrics(prev => ({
          workers: 0,
          shards: 1,
          loss: null,
          accuracy: null,
          round: 0,
          status: 'DISCONNECTED',
          bufferFill: 0,
          bufferK: 5,
        }));
        reconnectTimeout = setTimeout(connect, 3000);
      };

      wsRef.current.onerror = (err) => {
        console.error('WebSocket Error:', err);
      };
    };

    connect();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [url, maxPoints]);

  return { connected, metrics, history };
}
