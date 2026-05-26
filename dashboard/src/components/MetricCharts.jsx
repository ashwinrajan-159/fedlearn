import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from 'recharts';

/**
 * Premium charting component rendering Loss and Accuracy trends.
 * Supports smooth interpolations, custom glass tooltips, and blank states.
 * 
 * @param {Array} history - List of historical metrics data points.
 */
export function MetricCharts({ history }) {
  const hasData = history.length > 0;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{label}</p>
          {payload.map((p, idx) => (
            <p key={idx} className="tooltip-value" style={{ color: p.color }}>
              {p.name}: {p.value} {p.name === 'Accuracy' ? '%' : ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (!hasData) {
    return (
      <div className="empty-charts-card">
        <div className="pulse-loader" />
        <p className="pulse-text">Awaiting network updates to stream graphs...</p>
      </div>
    );
  }

  return (
    <div className="charts-grid">
      {/* Loss Card */}
      <div className="glass-chart-card">
        <div className="chart-header">
          <div>
            <h3 className="chart-title">Global Convergence Loss</h3>
            <p className="chart-description">Cross-Entropy Loss (Lower is better)</p>
          </div>
          {history.length > 0 && (
            <span className="live-badge-red">LIVE</span>
          )}
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={history} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.03)" />
              <XAxis 
                dataKey="timeStr" 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false}
                dy={8}
              />
              <YAxis 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false}
                domain={['auto', 'auto']}
                dx={-8}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255, 255, 255, 0.05)', strokeWidth: 1 }} />
              <Line 
                name="Loss"
                type="monotone" 
                dataKey="displayLoss" 
                stroke="#ef4444" 
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: '#ef4444' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Accuracy Card */}
      <div className="glass-chart-card">
        <div className="chart-header">
          <div>
            <h3 className="chart-title">Validation Accuracy</h3>
            <p className="chart-description">Evaluation score on held-out dataset</p>
          </div>
          {history.length > 0 && (
            <span className="live-badge-green">LIVE</span>
          )}
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={history} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.03)" />
              <XAxis 
                dataKey="timeStr" 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false}
                dy={8}
              />
              <YAxis 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false}
                domain={[0, 100]}
                dx={-8}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255, 255, 255, 0.05)', strokeWidth: 1 }} />
              <Line 
                name="Accuracy"
                type="monotone" 
                dataKey="displayAccuracy" 
                stroke="#10b981" 
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 0, fill: '#10b981' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
