import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import Sidebar from '../components/Sidebar.jsx';
import TopBar from '../components/TopBar.jsx';
import { analyticsApi } from '../services/api.js';
import { REVENUE_EXPLANATION } from '../data/reimbursementMap.js';
import '../styles/dashboard.css';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
const n = (v) => (typeof v === 'number' && !isNaN(v) ? v : 0);

function RevenueTooltip({ visible, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!visible) return;
    function handler(e) { if (ref.current && !ref.current.contains(e.target)) onClose(); }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [visible, onClose]);

  if (!visible) return null;
  return (
    <div className="revenue-tooltip" ref={ref} role="tooltip">
      <strong> How this is calculated</strong>
      {REVENUE_EXPLANATION}
      <br /><br />
      <span style={{ color: 'var(--clr-text-muted)', fontSize: '0.68rem' }}>
        Ranges based on CMS Medicare Fee Schedule 2024. Actual facility rates may vary.
      </span>
    </div>
  );
}

function KpiCard({ icon, label, value, sub, color, showInfo, onInfo }) {
  return (
    <div className="kpi-card" style={{ borderTopColor: color }}>
      <div className="kpi-icon" aria-hidden="true">{icon}</div>
      <div className="kpi-value" aria-label={`${label}: ${value}`}>{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
      {showInfo && (
        <button
          className="kpi-info-btn"
          onClick={onInfo}
          aria-label="Show revenue calculation details"
          title="How is this calculated?"
        >
          ℹ
        </button>
      )}
    </div>
  );
}

export default function AnalyticsPage() {
  const [overview, setOverview] = useState(null);
  const [trends,   setTrends]   = useState([]);
  const [days,     setDays]     = useState(30);
  const [currency, setCurrency] = useState('usd');
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState('');
  const [showRevTooltip, setShowRevTooltip] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [ovRes, trRes] = await Promise.all([
        analyticsApi.overview(days, currency),
        analyticsApi.trends(days, currency),
      ]);
      setOverview(ovRes.data?.data ?? null);
      const raw = Array.isArray(trRes.data?.trends) ? trRes.data.trends : [];
      setTrends(raw.map(row => ({
        date:         row.date        ?? '',
        cases:        n(row.cases),
        revenue:      n(row.revenue),
        avg_risk:     n(row.avg_risk),
      })));
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load analytics.');
    } finally {
      setLoading(false);
    }
  }, [days, currency]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const errorPieData = overview
    ? [
        { name: 'Under-coding', value: n(overview.undercoding_count) },
        { name: 'Over-coding',  value: n(overview.overcoding_count) },
        { name: 'Correct',      value: n(overview.correct_code_count) },
      ].filter(d => d.value > 0)
    : [];

  const safeErrorPieData = errorPieData.length > 0
    ? errorPieData
    : [{ name: 'No data yet', value: 1 }];

  const sym = currency === 'inr' ? '₹' : '$';

  const headerActions = (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <div style={{ background: 'var(--clr-surface-2)', borderRadius: '6px', padding: '0.2rem', display: 'flex', border: '1px solid var(--clr-border)' }}>
        {['usd', 'inr'].map(c => (
          <button key={c} onClick={() => setCurrency(c)}
            aria-pressed={currency === c}
            style={{
              background: currency === c ? '#3b82f6' : 'transparent',
              color: '#fff', border: 'none', padding: '0.3rem 0.75rem',
              borderRadius: '4px', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600, fontFamily: 'inherit',
            }}>
            {c.toUpperCase()}
          </button>
        ))}
      </div>
      <select
        className="days-select"
        value={days}
        onChange={e => setDays(Number(e.target.value))}
        aria-label="Select time period"
      >
        <option value={7}>Last 7 days</option>
        <option value={30}>Last 30 days</option>
        <option value={90}>Last 90 days</option>
        <option value={365}>Last 365 days</option>
      </select>
      <button className="new-analysis-btn" onClick={fetchData} disabled={loading} aria-label="Refresh analytics">
        ↻ Refresh
      </button>
    </div>
  );

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <main className="dashboard-main" id="main-content">
        <TopBar
          pageTitle="Dashboard"
          actions={headerActions}
        />

        <div className="dashboard-content">
          {error && (
            <div className="error-banner" role="alert">
              <span>⚠</span> {error}
              <button className="error-banner-retry" onClick={fetchData}>Retry</button>
            </div>
          )}

          {loading ? (
            <div className="loading-center" role="status">
              <div className="big-spinner" aria-hidden="true" /> Loading analytics…
            </div>
          ) : overview ? (
            <>
              <div className="kpi-grid">
                <KpiCard icon="" label="Total Cases"         value={n(overview.total_cases)}                                         color="#6366f1" />
                <div className="kpi-card" style={{ borderTopColor: '#f59e0b', position: 'relative' }}>
                  <div className="kpi-icon" aria-hidden="true"></div>
                  <div className="kpi-value">{sym}{n(overview.total_revenue_impact).toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
                  <div className="kpi-label">Est. Revenue Impact</div>
                  <div className="kpi-sub">Potential reimbursement gap from coding errors</div>
                  <div style={{ position: 'relative' }}>
                    <button
                      className="kpi-info-btn"
                      onClick={() => setShowRevTooltip(v => !v)}
                      aria-label="Show revenue methodology details"
                      aria-expanded={showRevTooltip}
                      style={{ position: 'static', display: 'inline-flex', marginTop: '0.4rem', gap: '0.25rem', fontSize: '0.72rem', color: 'var(--clr-text-muted)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', padding: 0 }}
                    >
                      ℹ Methodology
                    </button>
                    <RevenueTooltip visible={showRevTooltip} onClose={() => setShowRevTooltip(false)} />
                  </div>
                </div>

                <KpiCard icon="" label="High Risk Cases"    value={n(overview.high_risk_cases)}   color="#ef4444" />
                <KpiCard icon="" label="Under-coding"        value={n(overview.undercoding_count)} color="#8b5cf6" />
                <KpiCard icon="" label="Over-coding"         value={n(overview.overcoding_count)}  color="#06b6d4" />
                
                <KpiCard icon="" label="Approval Rate"      value={`${n(overview.approval_rate_pct)}%`} color="#10b981" sub="Resolved cases approved" />
                <KpiCard icon="" label="Rejection Rate"     value={`${n(overview.rejection_rate_pct)}%`} color="#ef4444" sub="Resolved cases rejected" />
                <KpiCard icon="" label="Avg Processing Time" value={`${n(overview.avg_processing_time)}s`} color="#6366f1" sub="End-to-end audit duration" />
              </div>

              {n(overview.total_cases) === 0 && (
                <div className="empty-state-card" role="status">
                  <div className="empty-state-icon"></div>
                  <div className="empty-state-title">No data for this period</div>
                  <p className="empty-state-desc">Run an audit from the Analyse tab to generate analytics data.</p>
                </div>
              )}

              <div className="charts-row">
                <div className="chart-card">
                  <h3>Cases Over Time</h3>
                  {trends.length === 0 ? (
                    <div className="empty-state">No trend data available yet.</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--clr-border)" />
                        <XAxis dataKey="date" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} />
                        <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} allowDecimals={false} />
                        <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: '8px', color: 'var(--clr-text-primary)' }} />
                        <Line type="monotone" dataKey="cases" stroke="#6366f1" strokeWidth={2} dot={false} name="Cases" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <div className="chart-card">
                  <h3>Est. Revenue Impact ({currency.toUpperCase()})</h3>
                  {trends.length === 0 ? (
                    <div className="empty-state">No trend data available yet.</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--clr-border)" />
                        <XAxis dataKey="date" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} />
                        <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: '8px' }} formatter={(v) => [`${sym}${n(v).toFixed(0)}`, 'Revenue Impact']} />
                        <Bar dataKey="revenue" fill="#10b981" name={`Revenue (${sym})`} radius={[4,4,0,0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="charts-row">
                <div className="chart-card">
                  <h3>Avg Risk Score Over Time</h3>
                  {trends.length === 0 ? (
                    <div className="empty-state">No trend data available yet.</div>
                  ) : (
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={trends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--clr-border)" />
                        <XAxis dataKey="date" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} />
                        <YAxis domain={[0, 100]} tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: '8px' }} />
                        <Line type="monotone" dataKey="avg_risk" stroke="#ef4444" strokeWidth={2} dot={false} name="Avg Risk" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <div className="chart-card">
                  <h3>Error Type Distribution</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie
                        data={safeErrorPieData}
                        cx="50%" cy="50%" outerRadius={75}
                        dataKey="value"
                        label={({ name, percent }) => percent > 0.05 ? `${name} ${(percent * 100).toFixed(0)}%` : ''}
                      >
                        {safeErrorPieData.map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)' }} />
                      <Legend wrapperStyle={{ color: 'var(--clr-text-secondary)', fontSize: '12px' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="empty-state-card" role="status">
              <div className="empty-state-icon"></div>
              <div className="empty-state-title">No analytics data</div>
              <p className="empty-state-desc">Run an audit from the Analyse tab to see data here.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
