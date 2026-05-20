/**
 * AdminDashboard.jsx – Governance & Model Performance Admin Panel
 *
 * Features:
 *   1. System Metrics Dashboard (precision, recall, FPR, FNR, avg confidence, acceptance rate)
 *   2. Code Rejection Management table
 *   3. Top Hallucinated Codes ranking
 *   4. Feedback submission (accept / reject AI codes)
 */

import { useEffect, useState, useCallback } from "react";
import axios from "axios";
import "./AdminDashboard.css";

const API = import.meta.env.VITE_API_URL || "http://161.118.217.29:8000/api/v1";

function pct(v) {
    return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}

function MetricCard({ label, value, color, sub }) {
    return (
        <div className={`adm-card adm-card--${color}`}>
            <span className="adm-card__label">{label}</span>
            <span className="adm-card__value">{value}</span>
            {sub && <span className="adm-card__sub">{sub}</span>}
        </div>
    );
}

function StatusBadge({ value, threshold, invert = false }) {
    const good = invert ? value <= threshold : value >= threshold;
    return (
        <span className={`adm-badge ${good ? "adm-badge--ok" : "adm-badge--warn"}`}>
            {good ? "✓ On Target" : "⚠ Below Target"}
        </span>
    );
}

export default function AdminDashboard() {
    const [metrics, setMetrics] = useState(null);
    const [feedback, setFeedback] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const [metricsRes, feedbackRes] = await Promise.all([
                axios.get(`${API}/admin/metrics`),
                axios.get(`${API}/admin/feedback-stats`),
            ]);
            setMetrics(metricsRes.data);
            setFeedback(feedbackRes.data);
        } catch (e) {
            setError(e.message || "Failed to load admin data.");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchData(); }, [fetchData, refreshKey]);

    if (loading) return (
        <div className="adm-center">
            <div className="adm-spinner" />
            <p>Loading admin data…</p>
        </div>
    );

    if (error) return (
        <div className="adm-center adm-error">
            <h2>⚠ Error</h2>
            <p>{error}</p>
            <button className="adm-btn" onClick={() => setRefreshKey(k => k + 1)}>Retry</button>
        </div>
    );

    const m = metrics || {};
    const f = feedback || {};

    return (
        <div className="adm-root">
            {/* ── Header ── */}
            <header className="adm-header">
                <div className="adm-header__title">
                    <span className="adm-header__icon">🛡️</span>
                    <div>
                        <h1>Model Governance Dashboard</h1>
                        <p>CodePerfectAuditor — AI Performance &amp; Human Feedback</p>
                    </div>
                </div>
                <div className="adm-header__actions">
                    <span className="adm-header__audits">{m.total_audits ?? 0} total audits</span>
                    <button className="adm-btn adm-btn--primary" onClick={() => setRefreshKey(k => k + 1)}>
                        ↻ Refresh
                    </button>
                </div>
            </header>

            {/* ── Section 1: System Metrics ── */}
            <section className="adm-section">
                <h2 className="adm-section__title">📊 Model Performance Metrics</h2>
                <div className="adm-metrics-grid">
                    <MetricCard
                        label="Precision"
                        value={pct(m.precision)}
                        color={m.precision >= 0.85 ? "green" : "orange"}
                        sub={<StatusBadge value={m.precision} threshold={0.85} />}
                    />
                    <MetricCard
                        label="Recall"
                        value={pct(m.recall)}
                        color={m.recall >= 0.75 ? "green" : "orange"}
                    />
                    <MetricCard
                        label="False Positive Rate"
                        value={pct(m.false_positive_rate)}
                        color={m.false_positive_rate <= 0.10 ? "green" : "red"}
                        sub={<StatusBadge value={m.false_positive_rate} threshold={0.10} invert />}
                    />
                    <MetricCard
                        label="False Negative Rate"
                        value={pct(m.false_negative_rate)}
                        color={m.false_negative_rate <= 0.15 ? "green" : "orange"}
                    />
                    <MetricCard
                        label="Avg Confidence"
                        value={m.avg_confidence != null ? m.avg_confidence.toFixed(2) : "—"}
                        color="blue"
                    />
                    <MetricCard
                        label="Human Acceptance Rate"
                        value={pct(m.human_acceptance_rate)}
                        color={m.human_acceptance_rate >= 0.75 ? "green" : "orange"}
                    />
                </div>

                {/* Target bands */}
                <div className="adm-targets">
                    <span>🎯 Targets: Precision &gt; 85% &nbsp;|&nbsp; FPR &lt; 10%</span>
                </div>
            </section>

            {/* ── Section 2: Feedback Overview ── */}
            <section className="adm-section">
                <h2 className="adm-section__title">👤 Human Reviewer Activity</h2>
                <div className="adm-feedback-summary">
                    <div className="adm-feedback-summary__item">
                        <span className="adm-feedback-summary__count">{f.total_feedback ?? 0}</span>
                        <span className="adm-feedback-summary__label">Total Reviews</span>
                    </div>
                    <div className="adm-feedback-summary__item adm-feedback-summary__item--green">
                        <span className="adm-feedback-summary__count">{f.total_acceptances ?? 0}</span>
                        <span className="adm-feedback-summary__label">Accepted</span>
                    </div>
                    <div className="adm-feedback-summary__item adm-feedback-summary__item--red">
                        <span className="adm-feedback-summary__count">{f.total_rejections ?? 0}</span>
                        <span className="adm-feedback-summary__label">Rejected</span>
                    </div>
                </div>

                {/* Rejection reasons breakdown */}
                {f.rejection_reasons && Object.keys(f.rejection_reasons).length > 0 && (
                    <div className="adm-reasons">
                        <h3>Rejection Reasons</h3>
                        <div className="adm-reasons__grid">
                            {Object.entries(f.rejection_reasons).sort((a, b) => b[1] - a[1]).map(([reason, count]) => (
                                <div key={reason} className="adm-reason-chip">
                                    <span className="adm-reason-chip__label">{reason.replace(/_/g, " ")}</span>
                                    <span className="adm-reason-chip__count">{count}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </section>

            {/* ── Section 3: Top Hallucinated Codes ── */}
            <section className="adm-section">
                <h2 className="adm-section__title">🚫 Top Hallucinated Codes</h2>
                {m.top_hallucinated_codes && m.top_hallucinated_codes.length > 0 ? (
                    <table className="adm-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Code</th>
                                <th>Rejection Count</th>
                                <th>Severity</th>
                            </tr>
                        </thead>
                        <tbody>
                            {m.top_hallucinated_codes.map((item, idx) => (
                                <tr key={item.code}>
                                    <td className="adm-table__rank">{idx + 1}</td>
                                    <td><code className="adm-code">{item.code}</code></td>
                                    <td>
                                        <div className="adm-bar-wrap">
                                            <div
                                                className="adm-bar"
                                                style={{
                                                    width: `${Math.min(100, (item.rejection_count / (m.top_hallucinated_codes[0]?.rejection_count || 1)) * 100)}%`,
                                                }}
                                            />
                                            <span>{item.rejection_count}</span>
                                        </div>
                                    </td>
                                    <td>
                                        <span className={`adm-badge ${idx < 3 ? "adm-badge--red" : "adm-badge--warn"}`}>
                                            {idx < 3 ? "High" : "Medium"}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <p className="adm-empty">No rejected codes recorded yet. Run audits and submit reviewer feedback to populate this table.</p>
                )}
            </section>

            {/* ── Section 4: Top Rejected Codes (from feedback-stats) ── */}
            {f.top_rejected_codes && f.top_rejected_codes.length > 0 && (
                <section className="adm-section">
                    <h2 className="adm-section__title">📋 Code Rejection Log</h2>
                    <table className="adm-table">
                        <thead>
                            <tr>
                                <th>Code</th>
                                <th>Rejection Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {f.top_rejected_codes.map(item => (
                                <tr key={item.code}>
                                    <td><code className="adm-code">{item.code}</code></td>
                                    <td>{item.count}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </section>
            )}
        </div>
    );
}
