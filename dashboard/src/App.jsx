import React, { useMemo, useState, useEffect, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { MetricCharts } from './components/MetricCharts';
import { 
  Activity, 
  Cpu, 
  Layers, 
  TrendingDown, 
  Target, 
  Wifi, 
  WifiOff, 
  Database,
  RefreshCw,
  Plus,
  Link,
  Copy,
  Check,
  Terminal,
  ArrowLeft,
  AlertTriangle,
  Flame,
  Settings2,
  Users
} from 'lucide-react';

/**
 * Parses the current URL path to extract jobId if in /join/<jobId> route.
 */
const getJobIdFromPath = () => {
  const path = window.location.pathname;
  const match = path.match(/^\/join\/([^/]+)/);
  return match ? match[1] : null;
};

function App() {
  // --- Routing State ---
  const [jobId, setJobId] = useState(getJobIdFromPath());
  const [jobConfig, setJobConfig] = useState(null);
  const [fetchError, setFetchError] = useState(null);
  const [loadingJob, setLoadingJob] = useState(false);

  // --- Landing Form Hyperparameter States ---
  const [ownerId, setOwnerId] = useState('Research Lab Alpha');
  const [minWorkers, setMinWorkers] = useState(3);
  const [fedproxMu, setFedproxMu] = useState(0.05);
  const [localSteps, setLocalSteps] = useState(3);
  const [dpC, setDpC] = useState(1.0);
  const [dpSigma, setDpSigma] = useState(0.2);
  const [topK, setTopK] = useState(0.10);
  const [quantize, setQuantize] = useState('int8');
  const [submittingJob, setSubmittingJob] = useState(false);

  // --- Invite Link Copy State ---
  const [copied, setCopied] = useState(false);

  // --- Local Volunteer Worker State ---
  const [volunteerEnabled, setVolunteerEnabled] = useState(true);
  const [workerLogs, setWorkerLogs] = useState([]);
  const [workerStatus, setWorkerStatus] = useState('STANDBY');
  const workerRef = useRef(null);
  const logContainerRef = useRef(null);

  // --- WebSocket Connection for Telemetry ---
  const wsUrl = jobId ? `ws://localhost:8080/dashboard/${jobId}` : null;
  const { connected, metrics, history } = useWebSocket(wsUrl, 100);

  // Sync route on popstate (browser back/forward)
  useEffect(() => {
    const handlePopState = () => {
      setJobId(getJobIdFromPath());
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Fetch job status if jobId is present
  useEffect(() => {
    if (!jobId) {
      setJobConfig(null);
      setFetchError(null);
      return;
    }

    setLoadingJob(true);
    setFetchError(null);

    fetch(`http://localhost:8000/job/${jobId}`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Job ${jobId} not found or server error.`);
        }
        const data = await res.json();
        setJobConfig(data);
      })
      .catch((err) => {
        console.error(err);
        setFetchError(err.message);
      })
      .finally(() => {
        setLoadingJob(false);
      });
  }, [jobId]);

  // --- Spawn / Manage Compute Web Worker ---
  useEffect(() => {
    // Terminate existing worker if any
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }

    if (!jobId || !jobConfig || !volunteerEnabled) {
      setWorkerStatus('STANDBY');
      return;
    }

    setWorkerStatus('INITIALIZING');
    addWorkerLog('Spawning browser Web Worker thread...');

    try {
      workerRef.current = new Worker('/fl_worker.js');

      // Setup event receiver
      workerRef.current.onmessage = (e) => {
        const { type, msg } = e.data;
        if (type === 'STATUS') {
          setWorkerStatus('ACTIVE');
          addWorkerLog(`[System] ${msg}`);
        } else if (type === 'RESULT') {
          addWorkerLog(`[Train] ${msg}`);
        } else if (type === 'ERROR') {
          setWorkerStatus('ERROR');
          addWorkerLog(`[Error] ${msg}`);
        }
      };

      workerRef.current.onerror = (err) => {
        setWorkerStatus('ERROR');
        addWorkerLog(`[Thread Error] Web Worker crashed: ${err.message}`);
      };

      // Send initialization signal with mapped configs
      workerRef.current.postMessage({
        type: 'INIT',
        payload: {
          threads: navigator.hardwareConcurrency || 4,
          dpClipC: Number(jobConfig.modelConfig.dpC),
          dpSigma: Number(jobConfig.modelConfig.dpSigma),
          dpDelta: 1e-5,
          topKFrac: Number(jobConfig.modelConfig.topK),
          quantize: jobConfig.modelConfig.quantize,
          fedproxMu: Number(jobConfig.modelConfig.fedproxMu),
          localSteps: Number(jobConfig.modelConfig.localSteps),
          jobId: jobConfig.jobId
        }
      });

    } catch (err) {
      setWorkerStatus('ERROR');
      addWorkerLog(`[Initialization Failed] ${err.message}`);
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [jobId, jobConfig, volunteerEnabled]);

  // Autoscroll terminal logger
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [workerLogs]);

  // Helper to append log line
  const addWorkerLog = (text) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setWorkerLogs((prev) => [...prev, `[${time}] ${text}`].slice(-100));
  };

  // --- Handlers ---
  const handleCreateJob = async (e) => {
    e.preventDefault();
    setSubmittingJob(true);

    try {
      const response = await fetch('http://localhost:8000/job/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ownerId: ownerId.trim() || 'Anonymous Sponsor',
          modelConfig: {
            minWorkers: Number(minWorkers),
            fedproxMu: Number(fedproxMu),
            localSteps: Number(localSteps),
            dpC: Number(dpC),
            dpSigma: Number(dpSigma),
            topK: Number(topK),
            quantize: quantize,
          }
        })
      });

      if (!response.ok) {
        throw new Error('Failed to create training job');
      }

      const createdJob = await response.json();
      // Reset logs
      setWorkerLogs([]);
      // Set path and navigate
      window.history.pushState({}, '', `/join/${createdJob.jobId}`);
      setJobId(createdJob.jobId);
    } catch (err) {
      alert(`Error creating job: ${err.message}`);
    } finally {
      setSubmittingJob(false);
    }
  };

  const copyInviteLink = () => {
    const link = `${window.location.origin}/join/${jobId}`;
    navigator.clipboard.writeText(link).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const navigateToHome = () => {
    window.history.pushState({}, '', '/');
    setJobId(null);
    setJobConfig(null);
    setFetchError(null);
    setWorkerLogs([]);
  };

  // Compute best historical validation metrics to contextualize progress
  const lowestLoss = useMemo(() => {
    const validLosses = history
      .map(h => h.loss)
      .filter(l => l !== null && l !== undefined);
    return validLosses.length > 0 ? Math.min(...validLosses) : null;
  }, [history]);

  const highestAccuracy = useMemo(() => {
    const validAccs = history
      .map(h => h.accuracy)
      .filter(a => a !== null && a !== undefined);
    return validAccs.length > 0 ? Math.max(...validAccs) : null;
  }, [history]);

  const estimatedSpeed = useMemo(() => {
    if (history.length < 2) return 0;
    const points = history.slice(-5);
    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];
    const durationSec = (lastPoint.timestamp - firstPoint.timestamp) / 1000;
    if (durationSec <= 0) return 0;
    return Number(((points.length - 1) / durationSec).toFixed(1));
  }, [history]);

  const getPrivacyStrength = (sigma) => {
    const s = Number(sigma);
    if (s === 0) return 'Disabled (No DP)';
    if (s < 0.25) return `Low (σ = ${s})`;
    if (s < 0.5) return `Medium (σ = ${s})`;
    return `High (σ = ${s})`;
  };

  const bufferPercentage = Math.min(100, (metrics.bufferFill / (metrics.bufferK || 1)) * 100);

  // ═══════════════════════════════════════════════════════════════════════════════
  //  RENDER CASE 1: Loading
  // ═══════════════════════════════════════════════════════════════════════════════
  if (loadingJob) {
    return (
      <div className="dashboard-container loading-container">
        <div className="empty-charts-card">
          <div className="pulse-loader" />
          <p className="pulse-text">Fetching Federated Job Configuration...</p>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  RENDER CASE 2: Error (Job not found)
  // ═══════════════════════════════════════════════════════════════════════════════
  if (fetchError) {
    return (
      <div className="dashboard-container error-page-container">
        <div className="glass-card error-card">
          <div className="error-icon-container">
            <AlertTriangle className="error-icon" />
          </div>
          <h2>Session Not Found</h2>
          <p className="error-message">
            The Federated Learning session ID <code className="retro-code">"{jobId}"</code> does not exist on the coordinator or has expired.
          </p>
          <button className="primary-btn" onClick={navigateToHome}>
            <ArrowLeft size={16} /> Return to Portal Home
          </button>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  RENDER CASE 3: Active Job Dashboard & Volunteer Compute Node
  // ═══════════════════════════════════════════════════════════════════════════════
  if (jobId && jobConfig) {
    return (
      <div className="dashboard-container">
        {/* Header bar */}
        <header className="dashboard-header">
          <div className="brand">
            <Database className="brand-icon" onClick={navigateToHome} style={{ cursor: 'pointer' }} />
            <div>
              <div className="brand-top-nav">
                <button className="back-portal-btn" onClick={navigateToHome}>
                  <ArrowLeft size={12} /> Portal Home
                </button>
                <span className="job-id-pill">JOB ID: {jobId}</span>
              </div>
              <h1>Job Dashboard: {jobConfig.ownerId}'s Session</h1>
              <p className="brand-subtitle">
                Isolated Parallel Training Session on Coordinator Port 8080
              </p>
            </div>
          </div>
          
          <div className="connection-status">
            <span className={`status-badge-server status-${metrics.status}`}>
              {metrics.status}
            </span>
            <div className="ws-indicator">
              {connected ? (
                <>
                  <Wifi className="icon-connected" />
                  <span className="text-connected">TELEMETRY SECURE</span>
                </>
              ) : (
                <>
                  <WifiOff className="icon-disconnected animate-pulse" />
                  <span className="text-disconnected">RECONNECTING...</span>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Dashboard Sub-grid: Stats Cards & Volunteer Compute Panel */}
        <div className="grid-split-dashboard">
          
          {/* LEFT COLUMN: System Specs & Volunteer Console */}
          <div className="system-volunteer-panel">
            {/* Invite Widget */}
            <div className="glass-card invite-widget-premium">
              <h3>Volunteer Invite Link</h3>
              <p>Share this link. Anyone who opens it instantly contributes their browser compute power to this specific training job.</p>
              <div className="copy-input-container">
                <input 
                  type="text" 
                  readOnly 
                  value={`${window.location.origin}/join/${jobId}`} 
                  onClick={copyInviteLink}
                />
                <button className="copy-btn" onClick={copyInviteLink}>
                  {copied ? <Check size={16} className="text-emerald" /> : <Copy size={16} />}
                </button>
              </div>
            </div>

            {/* Volunteer Control Box */}
            <div className={`glass-card volunteer-control-card border-glow-${workerStatus}`}>
              <div className="volunteer-header">
                <div className="volunteer-title-container">
                  <Flame className={`flame-icon flame-${workerStatus}`} />
                  <div>
                    <h3>Local Browser Worker Node</h3>
                    <p className="volunteer-subtitle">ONNX Runtime-Web Sandbox</p>
                  </div>
                </div>
                
                <div className="toggle-switch-container">
                  <span className="toggle-label">Volunteer Compute</span>
                  <label className="toggle-switch">
                    <input 
                      type="checkbox" 
                      checked={volunteerEnabled} 
                      onChange={(e) => setVolunteerEnabled(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              {/* Status banner */}
              <div className="volunteer-status-bar">
                <div className="status-indicator-dot-container">
                  <span className={`status-indicator-dot dot-${workerStatus}`} />
                  <span className="status-indicator-text">NODE STATUS: <strong>{workerStatus}</strong></span>
                </div>
                {volunteerEnabled && (
                  <span className="thread-badge">
                    {navigator.hardwareConcurrency || 4} CPU THREADS
                  </span>
                )}
              </div>

              {/* Terminal Logs */}
              <div className="retro-terminal">
                <div className="terminal-header">
                  <Terminal size={12} className="terminal-icon" />
                  <span>JS WEB WORKER TELEMETRY CONSOLE</span>
                  <div className="terminal-dots">
                    <span className="dot-red" />
                    <span className="dot-yellow" />
                    <span className="dot-green" />
                  </div>
                </div>
                <div className="terminal-logs-body" ref={logContainerRef}>
                  {workerLogs.length === 0 ? (
                    <div className="terminal-empty-state">
                      <span className="terminal-prompt">&gt;</span> Console standby. Awaiting thread activation...
                    </div>
                  ) : (
                    workerLogs.map((log, index) => (
                      <div key={index} className="terminal-line">
                        <span className="terminal-prompt">&gt;</span> {log}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Grid of metrics & cards */}
          <div className="stats-metric-grid">
            <div className="grid-summary">
              {/* Workers Card */}
              <div className="glass-card stat-card-premium">
                <div className="card-top">
                  <span className="card-label">Active Job Workers</span>
                  <Users className="card-icon blue-glow" />
                </div>
                <div className="card-value-container">
                  <span className="card-value">{metrics.workers}</span>
                  <span className="card-trend text-blue">Devices Connected</span>
                </div>
                <p className="card-footer-text">Connected to Coordinator Port 8080 and contributing gradients</p>
              </div>

              {/* Estimated Speed Card */}
              <div className="glass-card stat-card-premium">
                <div className="card-top">
                  <span className="card-label">Estimated Speed</span>
                  <Cpu className="card-icon purple-glow" />
                </div>
                <div className="card-value-container">
                  <span className="card-value">
                    {estimatedSpeed > 0 ? estimatedSpeed : '0.0'}
                  </span>
                  <span className="card-trend text-purple">Updates / Sec</span>
                </div>
                <p className="card-footer-text">Rolling throughput computed from incoming telemetry signals</p>
              </div>

              {/* Early Stopping / Rounds Card */}
              <div className="glass-card stat-card-premium">
                <div className="card-top">
                  <span className="card-label">Federated Rounds</span>
                  <Activity className="card-icon orange-glow" />
                </div>
                <div className="card-value-container">
                  <span className="card-value">{metrics.round}</span>
                  <span className="card-trend text-orange">Aggregations</span>
                </div>
                <p className="card-footer-text">Min workers required to trigger train round: {jobConfig.modelConfig.minWorkers}</p>
              </div>

              {/* FedBuff Queue Card */}
              <div className="glass-card stat-card-premium">
                <div className="card-top">
                  <span className="card-label">FedBuff Aggregation Queue</span>
                  <RefreshCw className="card-icon emerald-glow" />
                </div>
                <div className="card-value-container">
                  <span className="card-value">
                    {metrics.bufferFill}<span className="slash">/</span>{metrics.bufferK}
                  </span>
                  <span className="card-trend text-emerald">Updates</span>
                </div>
                
                <div className="progress-bar-container">
                  <div className="progress-bar-bg">
                    <div 
                      className="progress-bar-fill" 
                      style={{ width: `${bufferPercentage}%` }}
                    />
                  </div>
                </div>
                <p className="card-footer-text">Gradients buffered before robust Geometric Median</p>
              </div>
            </div>

            {/* Numerical Performance Info */}
            <div className="metrics-bar">
              {/* Loss Info */}
              <div className="glass-card metric-sub-card">
                <div className="metric-meta">
                  <TrendingDown className="icon-red" />
                  <span>Held-Out Loss</span>
                </div>
                <div className="metric-score red-text font-mono">
                  {metrics.loss !== null ? metrics.loss.toFixed(5) : 'N/A'}
                </div>
                <div className="metric-footer">
                  Best validation loss: <span>{lowestLoss !== null ? lowestLoss.toFixed(5) : 'N/A'}</span>
                </div>
              </div>

              {/* Accuracy Info */}
              <div className="glass-card metric-sub-card">
                <div className="metric-meta">
                  <Target className="icon-green" />
                  <span>Validation Accuracy</span>
                </div>
                <div className="metric-score green-text font-mono">
                  {metrics.accuracy !== null ? `${(metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                </div>
                <div className="metric-footer">
                  Best validation accuracy: <span>{highestAccuracy !== null ? `${(highestAccuracy * 100).toFixed(2)}%` : 'N/A'}</span>
                </div>
              </div>
            </div>
            
            {/* Shard Configurations Card */}
            <div className="glass-card config-summary-card">
              <div className="config-header">
                <Settings2 size={16} className="config-header-icon" />
                <h3>Hyperparameter Config Pipeline</h3>
              </div>
              <div className="config-badges-grid">
                <div className="config-badge-item">
                  <span className="config-badge-label">FedProx Regularizer:</span>
                  <span className="config-badge-value">μ = {jobConfig.modelConfig.fedproxMu}</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">Privacy Strength:</span>
                  <span className="config-badge-value">{getPrivacyStrength(jobConfig.modelConfig.dpSigma)}</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">DP Parameters:</span>
                  <span className="config-badge-value">C = {jobConfig.modelConfig.dpC}, σ = {jobConfig.modelConfig.dpSigma}</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">Local Steps:</span>
                  <span className="config-badge-value">{jobConfig.modelConfig.localSteps} SGD Steps</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">Top-K Sparsity:</span>
                  <span className="config-badge-value">{(jobConfig.modelConfig.topK * 100).toFixed(0)}% dimensions</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">Decompression Protocol:</span>
                  <span className="config-badge-value">{jobConfig.modelConfig.quantize.toUpperCase()}</span>
                </div>
                <div className="config-badge-item">
                  <span className="config-badge-label">Minimum Workers:</span>
                  <span className="config-badge-value">{jobConfig.modelConfig.minWorkers} Nodes</span>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Charts section */}
        <div className="charts-section">
          <MetricCharts history={history} />
        </div>

        {/* System Features Footer */}
        <footer className="dashboard-footer">
          <div className="feature-badges">
            <span className="badge">FedProx Regularization</span>
            <span className="badge">Differential Privacy</span>
            <span className="badge">Top-K Compression</span>
            <span className="badge">INT8 Quantization</span>
            <span className="badge">Weiszfeld median BFT</span>
            <span className="badge">Multi-Tenant Routing</span>
          </div>
          <p className="footer-copyright">
            FedLearn Multi-User Distributed Learning Dashboard • ONNX runtime worker
          </p>
        </footer>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  RENDER CASE 4: Landing Portal / Job Creation Portal
  // ═══════════════════════════════════════════════════════════════════════════════
  return (
    <div className="landing-portal-container">
      {/* Brand Hero header */}
      <div className="landing-hero">
        <div className="logo-pulse-ring">
          <Database className="hero-logo-icon" />
        </div>
        <h1>FedLearn Shard Portal</h1>
        <p className="hero-subtitle">
          Next-Generation Federated Learning Multi-User System with Dynamic Edge Volunteering.
        </p>
      </div>

      <div className="portal-grid">
        
        {/* LEFT COLUMN: Features Explanation & Premium Wow factor */}
        <div className="glass-card features-showcase">
          <h2>Isolated Federated Learning as a Service</h2>
          <p className="showcase-desc">
            Instantly spin up independent training sessions. Share your custom session URL to pool edge computing power from multiple devices.
          </p>

          <div className="feature-points-list">
            <div className="feature-point-item">
              <div className="point-icon-box bg-blue">
                <Cpu size={16} />
              </div>
              <div>
                <h4>Zero-Install Browser Compute</h4>
                <p>Volunteers join instantly via Web Workers. Training runs securely inside standard browser threads with ONNX runtime-web.</p>
              </div>
            </div>

            <div className="feature-point-item">
              <div className="point-icon-box bg-purple">
                <Target size={16} />
              </div>
              <div>
                <h4>Non-IID FedProx Stability</h4>
                <p>Configurable proximal normalization weights ensure high convergence speeds even when client data is highly heterogeneous.</p>
              </div>
            </div>

            <div className="feature-point-item">
              <div className="point-icon-box bg-orange">
                <Wifi size={16} />
              </div>
              <div>
                <h4>Bandwidth Sparsification</h4>
                <p>Combines Top-K absolute index filtering and INT8 symmetric quantization to compress client uploads by up to 100x.</p>
              </div>
            </div>

            <div className="feature-point-item">
              <div className="point-icon-box bg-emerald">
                <Layers size={16} />
              </div>
              <div>
                <h4>Byzantine Fault Tolerance</h4>
                <p>Weiszfeld Geometric Median client-side aggregation filters out malicious uploads, tolerating up to 49% adversaries.</p>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Premium Configuration Creation Form */}
        <div className="glass-card config-form-card">
          <div className="form-header">
            <Plus size={18} className="text-indigo" />
            <h2>Initialize Training Job</h2>
          </div>
          
          <form onSubmit={handleCreateJob}>
            
            {/* General details */}
            <div className="form-section">
              <label>Sponsor / Laboratory Name</label>
              <input 
                type="text" 
                required 
                placeholder="e.g. Stanford AI Lab"
                value={ownerId}
                onChange={(e) => setOwnerId(e.target.value)}
                className="premium-text-input"
              />
            </div>

            {/* Hyperparameters accordion/group */}
            <div className="form-section-title">
              <Settings2 size={14} />
              <span>HYPERPARAMETER CONFIGURATION</span>
            </div>

            <div className="sliders-grid">
              
              {/* Min Workers */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>Minimum Workers</label>
                  <span className="slider-bubble">{minWorkers}</span>
                </div>
                <input 
                  type="range" 
                  min="2" 
                  max="10" 
                  step="1"
                  value={minWorkers}
                  onChange={(e) => setMinWorkers(Number(e.target.value))}
                />
                <span className="slider-desc">Gradients needed to trigger aggregation</span>
              </div>

              {/* FedProx Mu */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>FedProx Mu (μ)</label>
                  <span className="slider-bubble">{fedproxMu}</span>
                </div>
                <input 
                  type="range" 
                  min="0.0" 
                  max="0.5" 
                  step="0.01"
                  value={fedproxMu}
                  onChange={(e) => setFedproxMu(Number(e.target.value))}
                />
                <span className="slider-desc">Weight variation proximal penalty coefficient</span>
              </div>

              {/* Local Steps */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>Local SGD Steps</label>
                  <span className="slider-bubble">{localSteps}</span>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  step="1"
                  value={localSteps}
                  onChange={(e) => setLocalSteps(Number(e.target.value))}
                />
                <span className="slider-desc">SGD steps trained per client batch</span>
              </div>

              {/* Top-K Sparsity */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>Top-K Sparsity Fraction</label>
                  <span className="slider-bubble">{(topK * 100).toFixed(0)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.05" 
                  max="1.0" 
                  step="0.05"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
                <span className="slider-desc">Dimension fraction transmitted per step</span>
              </div>

              {/* DP Clip C */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>DP L2 Clipping Bound (C)</label>
                  <span className="slider-bubble">{dpC}</span>
                </div>
                <input 
                  type="range" 
                  min="0.1" 
                  max="5.0" 
                  step="0.1"
                  value={dpC}
                  onChange={(e) => setDpC(Number(e.target.value))}
                />
                <span className="slider-desc">Sensitivity bounds for clipping. Lower C = stronger privacy, higher bias</span>
              </div>

              {/* DP Sigma */}
              <div className="slider-item">
                <div className="slider-meta">
                  <label>DP Noise Multiplier (σ)</label>
                  <span className="slider-bubble">{dpSigma}</span>
                </div>
                <input 
                  type="range" 
                  min="0.0" 
                  max="1.0" 
                  step="0.05"
                  value={dpSigma}
                  onChange={(e) => setDpSigma(Number(e.target.value))}
                />
                <span className="slider-desc">Gaussian noise variance parameter. Higher σ = more privacy, slower learning</span>
              </div>

            </div>

            {/* Quantization select option */}
            <div className="form-section">
              <label>Decompression & Quantization Mode</label>
              <select 
                value={quantize} 
                onChange={(e) => setQuantize(e.target.value)}
                className="premium-select"
              >
                <option value="none">None (Full Float32 Gradients)</option>
                <option value="int8">INT8 Quantization (Compressed Bitrates)</option>
              </select>
            </div>

            <button 
              type="submit" 
              className="primary-btn submit-btn" 
              disabled={submittingJob}
            >
              {submittingJob ? (
                <>
                  <RefreshCw className="animate-spin" size={16} /> Creating session...
                </>
              ) : (
                <>
                  <Flame size={16} /> Deploy Federated Session
                </>
              )}
            </button>

          </form>
        </div>

      </div>

      <footer className="dashboard-footer">
        <p className="footer-copyright">
          FedLearn Asynchronous Distributed Learning Console • Pure JS Edge Compute
        </p>
      </footer>
    </div>
  );
}

export default App;
