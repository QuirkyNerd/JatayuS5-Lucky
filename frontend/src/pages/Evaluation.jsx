import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from '../components/Sidebar.jsx';
import TopBar from '../components/TopBar.jsx';
import { auditApi } from '../services/api.js';
import '../styles/dashboard.css';

const STATE_LABELS = {
  NOT_STARTED: 'Not started',
  PRELOADING: 'Preloading models…',
  RUNNING: 'Benchmark running…',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
};

function MetricCard({ label, value, gain, positive }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{(value * 100).toFixed(1)}%</span>
      {gain !== undefined && (
        <span className={`metric-gain ${positive ? 'text-success' : 'text-danger'}`}>
          {gain > 0 ? '+' : ''}{(gain * 100).toFixed(1)}% vs Baseline
        </span>
      )}
    </div>
  );
}

export default function Evaluation() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [benchmarkState, setBenchmarkState] = useState('NOT_STARTED');
  const [progress, setProgress] = useState({ cases_done: 0, cases_total: 0, percent: 0 });
  const [statusMessage, setStatusMessage] = useState('');
  const pollRef = useRef(null);

  const applyResults = (data) => {
    if (data.status === 'error') {
      throw new Error(data.message || 'Evaluation failed');
    }
    if (data.status === 'running') {
      setBenchmarkState(data.benchmark_state || 'RUNNING');
      setProgress(data.progress || { cases_done: 0, cases_total: 0, percent: 0 });
      setStatusMessage(data.message || 'Benchmark running…');
      return false;
    }
    if (data.status === 'success') {
      setResults({
        ...data,
        benchmark_state: data.benchmark_state || 'COMPLETED',
      });
      setBenchmarkState(data.benchmark_state || 'COMPLETED');
      return true;
    }
    return false;
  };

  const fetchStatus = useCallback(async () => {
    try {
      const res = await auditApi.evaluationStatus();
      const s = res.data;
      setBenchmarkState(s.benchmark_state || 'NOT_STARTED');
      setStatusMessage(s.message || '');
      if (s.progress) setProgress(s.progress);
      return s;
    } catch (err) {
      console.warn('Status poll failed:', err);
      return null;
    }
  }, []);

  const loadEvaluation = async (force = false) => {
    setLoading(true);
    setError(null);
    try {
      const response = await auditApi.evaluate(force);
      const data = response.data;
      const done = applyResults(data);
      if (!done && (data.status === 'running' || data.benchmark_state === 'RUNNING')) {
        startPolling();
      }
    } catch (err) {
      console.error('Evaluation Error:', err);
      setError('Failed to load evaluation data: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      const s = await fetchStatus();
      if (!s) return;
      if (s.benchmark_state === 'COMPLETED' && s.has_results) {
        clearInterval(pollRef.current);
        pollRef.current = null;
        await loadEvaluation(false);
      } else if (s.benchmark_state === 'FAILED') {
        clearInterval(pollRef.current);
        pollRef.current = null;
        setError(s.error || 'Benchmark failed');
        setLoading(false);
      }
    }, 2500);
  }, [fetchStatus]);

  useEffect(() => {
    (async () => {
      const s = await fetchStatus();
      if (s?.benchmark_state === 'COMPLETED' && s?.has_results) {
        await loadEvaluation(false);
      } else if (s?.benchmark_state === 'RUNNING' || s?.benchmark_state === 'PRELOADING') {
        setLoading(true);
        startPolling();
      } else if (s?.benchmark_state === 'NOT_STARTED') {
        await loadEvaluation(false);
      } else {
        await loadEvaluation(false);
      }
    })();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const renderContent = () => {
    const isRunning = benchmarkState === 'RUNNING' || benchmarkState === 'PRELOADING';

    if (loading || isRunning) {
      const pct = progress.percent ?? 0;
      return (
        <div className="eval-loading-state">
          <div className="eval-spinner" />
          <p>{STATE_LABELS[benchmarkState] || 'Loading evaluation…'}</p>
          {statusMessage && <p style={{ color: 'var(--clr-text-muted)', fontSize: '0.85rem' }}>{statusMessage}</p>}
          {progress.cases_total > 0 && (
            <p style={{ marginTop: '0.5rem' }}>
              Progress: {progress.cases_done} / {progress.cases_total} ({pct}%)
            </p>
          )}
          <p style={{ color: 'var(--clr-text-muted)', fontSize: '0.8rem', marginTop: '0.75rem' }}>
            Target runtime: 60–120 seconds. Cross-encoder reranking active on top candidates.
          </p>
        </div>
      );
    }

    if (error) return <div className="eval-alert-danger">{error}</div>;

    if (results && results.status === 'empty') {
      return (
        <div className="eval-empty-state">
          <h3>No benchmark dataset available</h3>
          <p>{results.message || 'The system requires a benchmark dataset to run evaluation.'}</p>
        </div>
      );
    }

    if (results && results.status === 'success') {
      return (
        <div className="eval-results">
          <section className="eval-section">
            <h3 className="eval-section-title">Enhanced Intelligence Performance</h3>
            <div className="metrics-grid">
              <MetricCard
                label="Accuracy"
                value={results.metrics.accuracy}
                gain={results.improvements.accuracy_gain}
                positive={results.improvements.accuracy_gain > 0}
              />
              <MetricCard
                label="F1 Score"
                value={results.metrics.f1_score}
                gain={results.improvements.f1_gain}
                positive={results.improvements.f1_gain > 0}
              />
              <MetricCard
                label="MRR@10 (Retrieval)"
                value={results.metrics.mrr}
              />
              <MetricCard
                label="nDCG@10 (Retrieval)"
                value={results.metrics.ndcg_at_10}
              />
            </div>
          </section>

          <div className="eval-comparisons">
            <div className="eval-table-wrap">
              <h4 className="eval-table-title">Performance Comparison</h4>
              <table className="eval-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline (RAG Only)</th>
                    <th>Enhanced (Clinical Intelligence)</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Avg. False Positives / Case</td>
                    <td>{results.baseline.avg_false_positive}</td>
                    <td>{results.enhanced.avg_false_positive}</td>
                    <td className={results.improvements.fp_reduction >= 0 ? 'text-success' : 'text-danger'}>
                      {results.improvements.fp_reduction > 0 ? '-' : ''}{results.improvements.fp_reduction}
                    </td>
                  </tr>
                  <tr>
                    <td>Avg. Missed Codes / Case</td>
                    <td>{results.baseline.avg_missed}</td>
                    <td>{results.enhanced.avg_missed}</td>
                    <td className={results.improvements.missed_reduction >= 0 ? 'text-success' : 'text-danger'}>
                      {results.improvements.missed_reduction > 0 ? '-' : ''}{results.improvements.missed_reduction}
                    </td>
                  </tr>
                  <tr>
                    <td>Accuracy Score</td>
                    <td>{(results.baseline.accuracy * 100).toFixed(1)}%</td>
                    <td>{(results.enhanced.accuracy * 100).toFixed(1)}%</td>
                    <td className="text-success">+{(results.improvements.accuracy_gain * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {results.confusion_matrix && (
            <div className="eval-confusion-section">
              <h4 className="eval-table-title">Confusion Matrix (Enhanced Model)</h4>
              <div className="cm-grid">
                <div className="cm-label" />
                <div className="cm-header">Predicted Positive</div>
                <div className="cm-header">Predicted Negative</div>
                <div className="cm-row-label">Actual Positive</div>
                <div className="cm-cell tp">
                  <div className="cm-val">{results.confusion_matrix.TP}</div>
                  <div className="cm-tag">True Positive</div>
                </div>
                <div className="cm-cell fn">
                  <div className="cm-val">{results.confusion_matrix.FN}</div>
                  <div className="cm-tag">False Negative</div>
                </div>
                <div className="cm-row-label">Actual Negative</div>
                <div className="cm-cell fp">
                  <div className="cm-val">{results.confusion_matrix.FP}</div>
                  <div className="cm-tag">False Positive</div>
                </div>
                <div className="cm-cell tn">
                  <div className="cm-val">{results.confusion_matrix.TN}</div>
                  <div className="cm-tag">True Negative</div>
                </div>
              </div>
            </div>
          )}

          <div className="eval-improvement-summary">
            <h4 className="eval-summary-title">Optimization Impact</h4>
            <p style={{ marginBottom: '1rem', color: 'var(--clr-text-muted)' }}>{results.summary}</p>
            <div className="eval-impact-stats">
              <div className="eval-impact-stat">
                <span className="eval-impact-val">{results.improvements.fp_reduction}</span>
                <span className="eval-impact-desc">False Positives Reduced</span>
              </div>
              <div className="eval-impact-stat">
                <span className="eval-impact-val">{results.improvements.missed_reduction}</span>
                <span className="eval-impact-desc">Missed Codes Captured</span>
              </div>
            </div>
          </div>

          {results.interpretable_metrics && results.interpretable_metrics.top_rejection_rationales && (
            <div className="eval-insights-section" style={{ marginTop: '2rem' }}>
              <h4 className="eval-table-title">Audit Trail Insights</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div className="insight-card" style={{ padding: '1rem', background: 'var(--clr-surface-2)', borderRadius: '12px' }}>
                  <h5 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#ef4444' }}>Top Rejection Rationales</h5>
                  <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
                    {results.interpretable_metrics.top_rejection_rationales.map(([rat, count], i) => (
                      <li key={i} style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                        <strong>{count}x</strong> {rat.replace(/_/g, ' ')}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="insight-card" style={{ padding: '1rem', background: 'var(--clr-surface-2)', borderRadius: '12px' }}>
                  <h5 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#f59e0b' }}>Top Hallucination Patterns</h5>
                  <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
                    {results.interpretable_metrics.top_hallucination_rationales?.map(([rat, count], i) => (
                      <li key={i} style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                        <strong>{count}x</strong> {rat}
                      </li>
                    )) || <li style={{ fontSize: '0.85rem', color: 'var(--clr-text-muted)' }}>No hallucinations detected in this run.</li>}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  const stateBadge = STATE_LABELS[benchmarkState] || benchmarkState;
  const casesEvaluated =
    results?.status === 'success' && results.dataset_size != null
      ? results.dataset_size
      : null;

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <main className="dashboard-main" id="main-content">
        <TopBar
          pageTitle="System Benchmark & Evaluation"
          pageSubtitle="Clinical AI performance metrics against benchmark dataset"
        />
        <div className="dashboard-content">
          <div className="eval-header">
            <div className="eval-intro">
              <p className="eval-header-desc">
                Evaluation metrics are computed against a curated clinical benchmark dataset and reflect
                system performance under controlled conditions.
              </p>
              <div className="eval-header-status">
                <p className="eval-status-line">
                  Status: <strong>{stateBadge}</strong>
                </p>
                {casesEvaluated != null && (
                  <p className="eval-cases-line">{casesEvaluated} cases evaluated</p>
                )}
              </div>
            </div>
            <button
              className="new-analysis-btn eval-run-btn"
              onClick={() => loadEvaluation(true)}
              disabled={loading || benchmarkState === 'RUNNING' || benchmarkState === 'PRELOADING'}
            >
              {loading || benchmarkState === 'RUNNING' || benchmarkState === 'PRELOADING'
                ? 'Running…'
                : 'Run Full Evaluation'}
            </button>
          </div>

          {renderContent()}
        </div>
      </main>
    </div>
  );
}
