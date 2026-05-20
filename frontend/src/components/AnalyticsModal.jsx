import React, { useEffect, useState } from 'react'
import { getAnalytics } from '../services/api'

export default function AnalyticsModal({ onClose }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        getAnalytics().then(res => {
            setData(res.data)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    return (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="card" style={{ width: 500, padding: '2rem', background: 'var(--clr-bg)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', color: 'var(--clr-text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span></span> System Analytics
                    </h2>
                    <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--clr-text-muted)', cursor: 'pointer', fontSize: '1.5rem' }}>&times;</button>
                </div>
                {loading ? <p style={{ textAlign: 'center', padding: '2rem', color: 'var(--clr-text-muted)' }}><span className="pulse">Loading...</span></p> : data ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div className="card" style={{ textAlign: 'center', padding: '1.5rem', background: 'var(--clr-surface-2)' }}>
                            <div style={{ fontSize: '2rem', color: 'var(--clr-primary)', fontWeight: 700 }}>{data.total_audits}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--clr-text-muted)' }}>Total Audits</div>
                        </div>
                        <div className="card" style={{ textAlign: 'center', padding: '1.5rem', background: 'var(--clr-surface-2)' }}>
                            <div style={{ fontSize: '2rem', color: 'var(--clr-success)', fontWeight: 700 }}>{data.estimated_cost}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--clr-text-muted)' }}>Estimated Cost</div>
                        </div>
                        <div className="card" style={{ textAlign: 'center', padding: '1.5rem', gridColumn: '1 / -1', background: 'var(--clr-surface-2)' }}>
                            <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--clr-text-muted)' }}>Feedback Loop / Self-Improvement</h3>
                            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '1.25rem' }}>
                                <div>
                                    <div style={{ fontSize: '1.5rem', color: 'var(--clr-success)', fontWeight: 700 }}>{data.feedback_stats.accepted_suggestions}</div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-secondary)' }}>Accepted</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '1.5rem', color: 'var(--clr-warning)', fontWeight: 700 }}>{Math.round(data.feedback_stats.acceptance_rate * 100)}%</div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-secondary)' }}>Accept Rate</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '1.5rem', color: 'var(--clr-danger)', fontWeight: 700 }}>{data.feedback_stats.rejected_suggestions}</div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-secondary)' }}>Rejected</div>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : <p style={{ color: 'var(--clr-danger)', textAlign: 'center' }}>Failed to load analytics.</p>}
            </div>
        </div>
    )
}
