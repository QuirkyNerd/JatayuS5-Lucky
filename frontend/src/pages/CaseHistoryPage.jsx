import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar.jsx';
import TopBar, { FullPageLoader } from '../components/TopBar.jsx';
import { caseApi } from '../services/api.js';
import { useAuth } from '../main.jsx';
import { generatePdf } from '../utils/generatePdfReport.js';
import { Download, ChevronDown, ChevronUp, Activity, ShieldCheck, AlertTriangle } from 'lucide-react';
import '../styles/dashboard.css';
import { CodeExplainabilityPanel, RemovedCodesPanel } from '../components/AuditResults.jsx';

const STATUS_COLORS = {
  draft:        '#94a3b8',
  pending:      '#f59e0b',
  submitted:    '#6366f1',
  under_review: '#8b5cf6',
  approved:     '#10b981',
  rejected:     '#ef4444',
};

const STATUS_LABELS = {
  draft:        'Draft',
  pending:      'Pending',
  submitted:    'Submitted',
  under_review: 'In Review',
  approved:     'Approved',
  rejected:     'Rejected',
};

import { authApi } from '../services/api.js';

function StatusSelect({ caseId, currentStatus, onUpdate }) {
  const [busy, setBusy] = useState(false);
  const handleChange = async (e) => {
    const newStatus = e.target.value;
    if (newStatus === currentStatus) return;
    setBusy(true);
    await onUpdate(caseId, newStatus);
    setBusy(false);
  };
  return (
    <select
      className="status-select"
      value={currentStatus}
      onChange={handleChange}
      disabled={busy}
      aria-label={`Change case ${caseId} status`}
    >
      <option value="submitted">Submitted</option>
      <option value="under_review">In Review</option>
      <option value="approved">Approved</option>
      <option value="rejected">Rejected</option>
    </select>
  );
}

function ReviewerAssignmentPanel({ caseData, onActionComplete }) {
  const [reviewers, setReviewers] = useState([]);
  const [selectedReviewer, setSelectedReviewer] = useState(caseData.assigned_to || '');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    authApi.users().then(res => {
      const filtered = (res.data || []).filter(u => u.role === 'REVIEWER');
      setReviewers(filtered);
    }).catch(console.error);
  }, []);

  const handleAssign = async () => {
    if (!selectedReviewer) return;
    setBusy(true);
    try {
      await caseApi.assign(caseData.id, selectedReviewer);
      onActionComplete();
    } catch (e) {
      alert(e.response?.data?.detail || 'Assignment failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="reviewer-panel assignment" style={{ marginBottom: '1.5rem', border: '1px solid var(--clr-border)' }}>
      <h4>Assign Reviewer</h4>
      <div className="action-row" style={{ display: 'flex', gap: '0.5rem' }}>
        <select 
          value={selectedReviewer} 
          onChange={e => setSelectedReviewer(e.target.value)}
          disabled={busy}
          className="role-select"
          style={{ flex: 1, margin: 0, padding: '0.5rem', borderRadius: '6px' }}
        >
          <option value="">Select a Reviewer...</option>
          {reviewers.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
        </select>
        <button 
          className="approve-btn" 
          onClick={handleAssign} 
          disabled={busy || !selectedReviewer}
          style={{ padding: '0.5rem 1rem', borderRadius: '6px' }}
        >
          {busy ? 'Assigning...' : 'Assign'}
        </button>
      </div>
      {caseData.assigned_at && (
        <p style={{ fontSize: '0.72rem', color: 'var(--clr-text-muted)', marginTop: '0.5rem' }}>
          Currently assigned since {new Date(caseData.assigned_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}

function ReviewerActionPanel({ caseData, onActionComplete }) {
  const [notes, setNotes] = useState(caseData.reviewer_notes || '');
  const [confidence, setConfidence] = useState(caseData.review_confidence || 1.0);
  const [busy, setBusy] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [finalCodes, setFinalCodes] = useState(
    caseData.final_code_set && caseData.final_code_set.length > 0
      ? caseData.final_code_set
      : caseData.ai_codes || []
  );

  const handleApprove = async () => {
    setBusy(true);
    try {
      await caseApi.approve(caseData.id, confidence);
      onActionComplete();
    } catch (e) {
      alert(e.response?.data?.detail || 'Approval failed');
    } finally {
      setBusy(false);
    }
  };

  const handleReject = async () => {
    if (!notes.trim()) {
      alert('Justification is required for rejection.');
      return;
    }
    setBusy(true);
    try {
      await caseApi.reject(caseData.id, notes, confidence);
      onActionComplete();
    } catch (e) {
      alert(e.response?.data?.detail || 'Rejection failed');
    } finally {
      setBusy(false);
    }
  };

  const handleUpdateCodes = async () => {
    if (!notes.trim()) {
      alert('Justification is required when modifying codes.');
      return;
    }
    setBusy(true);
    try {
      await caseApi.updateCodes(caseData.id, finalCodes, notes);
      setIsEditing(false);
      onActionComplete();
    } catch (e) {
      alert(e.response?.data?.detail || 'Update failed');
    } finally {
      setBusy(false);
    }
  };

  const removeCode = (idx) => {
    setFinalCodes(prev => prev.filter((_, i) => i !== idx));
  };

  const addCode = () => {
    const code = window.prompt("Enter new code (e.g. E11.9):");
    if (code) {
      setFinalCodes(prev => [...prev, { code, description: 'Added by Reviewer', confidence: 1.0, type: 'ICD-10' }]);
    }
  };

  if (caseData.status === 'approved' || caseData.status === 'rejected') {
    return (
      <div className="reviewer-panel completed">
        <h4>Review Decision: <span className={caseData.status}>{caseData.status.toUpperCase()}</span></h4>
        <div style={{ fontSize: '0.8rem', color: 'var(--clr-text-secondary)', marginBottom: '0.75rem' }}>
          <strong>Confidence:</strong> {(caseData.review_confidence * 100).toFixed(0)}%
        </div>
        {caseData.reviewer_notes && (
          <div className="notes-display">
            <strong>Notes:</strong> {caseData.reviewer_notes}
          </div>
        )}
        <div className="final-codes">
          <strong>Final Code Set:</strong>
          <ul>
            {finalCodes.map((c, i) => <li key={i}>{c.code} - {c.description}</li>)}
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="reviewer-panel active">
      <h4>Reviewer Validation</h4>
      
      <div className="code-comparison">
        <div className="code-column">
          <h5>AI Suggestion</h5>
          <ul>
            {(caseData.ai_codes || []).map((c, i) => <li key={i}>{c.code}</li>)}
          </ul>
        </div>
        <div className="code-column">
          <h5>Human Entry</h5>
          <ul>
            {(caseData.human_codes || []).map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
        <div className="code-column final">
          <h5>Final Selection</h5>
          <ul>
            {finalCodes.map((c, i) => (
              <li key={i}>
                {c.code}
                {isEditing && <button onClick={() => removeCode(i)}>✕</button>}
              </li>
            ))}
            {isEditing && <li className="add-code" onClick={addCode}>+ Add Code</li>}
          </ul>
        </div>
      </div>

      {!isEditing ? (
        <button className="edit-btn" onClick={() => setIsEditing(true)}>Edit Codes</button>
      ) : (
        <button className="save-btn" onClick={handleUpdateCodes} disabled={busy}>Save Final Set</button>
      )}

      <div className="notes-input" style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.4rem', fontSize: '0.85rem', fontWeight: 500 }}>Justification / Reviewer Notes:</label>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Required for Rejection or Code Updates..."
          rows={3}
          style={{ width: '100%', padding: '0.6rem', borderRadius: '8px', border: '1px solid var(--clr-border)', background: 'var(--clr-surface-2)', color: 'var(--clr-text-primary)' }}
        />
      </div>

      <div className="confidence-input" style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', fontSize: '0.85rem', fontWeight: 500 }}>
          Review Confidence: <span style={{ color: 'var(--clr-primary)' }}>{(confidence * 100).toFixed(0)}%</span>
        </label>
        <input 
          type="range" min="0" max="1" step="0.05" 
          value={confidence} 
          onChange={e => setConfidence(parseFloat(e.target.value))}
          style={{ width: '100%', accentColor: 'var(--clr-primary)', cursor: 'pointer' }}
        />
      </div>

      <div className="action-row" style={{ display: 'flex', gap: '0.75rem' }}>
        <button className="approve-btn" onClick={handleApprove} disabled={busy || isEditing} style={{ flex: 1, padding: '0.75rem', borderRadius: '8px' }}>Approve</button>
        <button className="reject-btn" onClick={handleReject} disabled={busy || isEditing} style={{ flex: 1, padding: '0.75rem', borderRadius: '8px' }}>Reject</button>
      </div>
    </div>
  );
}

export default function CaseHistoryPage() {
  const { user, isCoder } = useAuth();
  const isReviewer = user?.role === 'REVIEWER';
  const isAdmin    = user?.role === 'ADMIN';
  const navigate = useNavigate();
  const isDemoSession = localStorage.getItem('demo_session') === 'true';


  const [cases,   setCases]   = useState([]);
  const [total,   setTotal]   = useState(0);
  const [page,    setPage]    = useState(1);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');
  const [selected, setSelected] = useState(null);
  const [auditTrail, setAuditTrail] = useState([]);

  const [status,  setStatus]  = useState('');
  const [minRisk, setMinRisk] = useState('');
  const [fromDate, setFrom]   = useState('');
  const [toDate,  setTo]      = useState('');
  const [assignedOnly, setAssignedOnly] = useState(isReviewer);

  const pageSize = 15;

  const fetchCases = useCallback(async () => {
    setLoading(true);
    setError('');
    setCases([]);
    try {
      // Both demo and production use the same filter params.
      // Backend handles role isolation and is_demo filtering automatically.
      const params = {
        page,
        page_size: pageSize,
        ...(status   ? { status }                   : {}),
        ...(minRisk  ? { min_risk: Number(minRisk) } : {}),
        ...(fromDate ? { from_date: fromDate }       : {}),
        ...(toDate   ? { to_date: toDate }           : {}),
        // Only send assigned_to filter for non-demo reviewers
        ...(!isDemoSession && isReviewer && assignedOnly ? { assigned_to: user.id } : {}),
      };

      console.log('Applied filters:', params);
      const res = await caseApi.list(params);
      console.log('Cases returned:', res.data);

      setCases(Array.isArray(res.data?.cases) ? res.data.cases : []);
      setTotal(res.data?.total || 0);
    } catch (e) {
      const status_code = e.response?.status;
      console.warn('[CaseHistory] API error:', status_code, e.message);
      if (status_code === 423) {
        setError('The selected case is currently locked by another reviewer.');
      } else if (status_code === 401) {
        setError('Session expired or unauthorised. Please log in again.');
      } else if (status_code === 403) {
        setError('Access denied. You do not have permission to view cases.');
      } else if (status_code >= 500) {
        setError('The case history service is temporarily unavailable. Please try again shortly.');
      } else {
        const detail = e.response?.data?.detail;
        setError(typeof detail === 'string' ? detail : e.message || 'Unable to load cases.');
      }
    } finally {
      setLoading(false);
    }
  }, [page, status, minRisk, fromDate, toDate, assignedOnly, isDemoSession, isReviewer]);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const handleStatusChange = useCallback(async (caseId, newStatus) => {
    try {
      if (newStatus === 'approved') {
        await caseApi.approve(caseId, 1.0);  // confidence required by backend
      } else if (newStatus === 'rejected') {
        const comment = window.prompt('Enter justification for rejection:') || 'No justification provided';
        await caseApi.reject(caseId, comment, 1.0);  // justification + confidence required
      } else {
        await caseApi.update(caseId, { status: newStatus });
      }
      fetchCases();
    } catch (e) {
      const detail = e.response?.data?.detail;
      const msg = typeof detail === 'string' ? detail : JSON.stringify(detail) || e.message || 'Unknown error';
      alert('Failed to update status: ' + msg);
    }
  }, [fetchCases]);

  const handleSelectCase = useCallback(async (c) => {
    setSelected(c);
    setAuditTrail([]);
    
    // Fetch audit trail
    caseApi.getAuditTrail(c.id).then(res => setAuditTrail(res.data)).catch(console.error);

    if (isReviewer && c.status === 'submitted') {
      try {
        const res = await caseApi.get(c.id);
        setSelected(res.data);
        // Refresh list to show 'under_review'
        fetchCases();
      } catch (e) {
        console.error('Failed to auto-transition case:', e);
      }
    }
  }, [isReviewer, fetchCases]);

  const clearFilters = useCallback(() => {
    setStatus(''); setMinRisk(''); setFrom(''); setTo(''); setPage(1);
  }, []);

  const totalPages = Math.ceil(total / pageSize);

  const headerActions = (
    <button className="new-analysis-btn" onClick={fetchCases} disabled={loading} aria-label="Refresh case list">
      ↻ Refresh
    </button>
  );

  const renderContent = () => {
    if (loading) {
      return (
        <div className="loading-center" role="status" aria-live="polite">
          <div className="big-spinner" aria-hidden="true" />
          Loading cases…
        </div>
      );
    }

    if (error) {
      return (
        <div className="error-banner" role="alert">
          <span aria-hidden="true">⚠</span>
          <div style={{ flex: 1 }}>
            <strong>Failed to load cases</strong>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.8rem', opacity: 0.85 }}>{error}</p>
          </div>
          <button className="error-banner-retry" onClick={fetchCases} aria-label="Retry loading cases">
            Retry
          </button>
        </div>
      );
    }

    if (cases.length === 0) {
      return (
        <div className="empty-state-card" role="status">
          <div className="empty-state-icon" aria-hidden="true" style={{ fontSize: '2rem', lineHeight: 1 }}>--</div>
          {/* ✅ STEP 6: FIX EMPTY STATE LOGIC */}
          <div className="empty-state-title">No cases yet</div>
          <p className="empty-state-desc">
            Try creating a case in Coder mode if you are in a demo session.
          </p>
          {(status || minRisk || fromDate || toDate) && (
            <button className="new-analysis-btn" onClick={clearFilters} style={{ marginTop: '0.5rem' }}>
              Clear Filters
            </button>
          )}
        </div>
      );
    }

    return (
      <>
        <div className="cases-table-wrapper">
          <table className="cases-table" aria-label="Case history table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Date</th>
                <th scope="col">Priority</th>
                <th scope="col">Reviewer</th>
                <th scope="col">Status</th>
                {(isReviewer || isAdmin) && <th scope="col">Update</th>}
                <th scope="col">Report</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c, index) => (
                <tr
                  key={c.id}
                  onClick={() => handleSelectCase(c)}
                  className="case-row"
                  tabIndex={0}
                  onKeyDown={e => e.key === 'Enter' && handleSelectCase(c)}
                  aria-label={`Case ${c.id}: ${c.summary || 'No summary'}`}
                >
                  <td>{(page - 1) * pageSize + index + 1}</td>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    {c.created_at ? new Date(c.created_at).toLocaleDateString() : '—'}
                  </td>
                  <td>
                    <span className={`priority-badge ${c.priority || 'normal'}`}>
                      {(c.priority || 'normal').toUpperCase()}
                    </span>
                  </td>
                  <td>
                    {c.reviewer_name || 'Unassigned'}
                    {c.assignment_status === 'reassigned' && <span style={{ fontSize: '0.6rem', color: '#f59e0b', marginLeft: '4px' }}>(Re)</span>}
                  </td>
                  <td>
                    <span
                      className="status-badge"
                      style={{ background: STATUS_COLORS[c.status] || '#64748b' }}
                    >
                      {STATUS_LABELS[c.status] || c.status}
                    </span>
                  </td>
                  {(isReviewer || isAdmin) && (
                    <td onClick={e => e.stopPropagation()}>
                      <StatusSelect
                        caseId={c.id}
                        currentStatus={c.status}
                        onUpdate={handleStatusChange}
                      />
                    </td>
                  )}
                  <td onClick={e => e.stopPropagation()}>
                    <button
                      className="download-report-btn"
                      title="Download Report"
                      aria-label={`Download PDF report for case ${c.id}`}
                      onClick={() => generatePdf(c)}
                    >
                      <Download size={18} strokeWidth={2} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <nav className="pagination" aria-label="Case list pagination">
            <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} aria-label="Previous page">← Prev</button>
            <span aria-current="page">Page {page} of {totalPages}</span>
            <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)} aria-label="Next page">Next →</button>
          </nav>
        )}
      </>
    );
  };

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <main className="dashboard-main" id="main-content">
        <TopBar
          pageTitle="Case History"
          pageSubtitle={!loading && !error && `${total} case${total !== 1 ? 's' : ''} found`}
          actions={headerActions}
        />

        <div className="dashboard-content">
          {/* Filter bar */}
          <div className="filter-bar" role="group" aria-label="Case filters">
            {/* Status filter buttons — use real backend status values */}
            {[
              { label: 'All',         value: '' },
              { label: 'Submitted',   value: 'submitted' },
              { label: 'In Review',   value: 'under_review' },
              { label: 'Approved',    value: 'approved' },
              { label: 'Rejected',    value: 'rejected' },
            ].map(({ label, value }) => (
              <button
                key={value || 'all'}
                style={{
                  padding: '0.45rem 1rem', borderRadius: '6px', cursor: 'pointer',
                  border: status === value ? '2px solid #6366f1' : '1px solid var(--clr-border)',
                  background: status === value ? '#6366f1' : 'transparent',
                  color: status === value ? '#fff' : 'var(--clr-text-secondary)',
                  fontSize: '0.8rem', fontFamily: 'inherit', fontWeight: status === value ? 600 : 400,
                  transition: 'all 0.15s',
                }}
                onClick={() => { setStatus(value); setPage(1); }}
                aria-pressed={status === value}
              >
                {label}
              </button>
            ))}
            <input
              type="number" placeholder="Min risk %" value={minRisk} min="0" max="100"
              aria-label="Minimum risk score filter"
              style={{ width: '110px', padding: '0.45rem 0.75rem', borderRadius: '6px', border: '1px solid var(--clr-border)', background: 'var(--clr-surface)', color: 'var(--clr-text-primary)', fontFamily: 'inherit' }}
              onChange={e => { setMinRisk(e.target.value); setPage(1); }}
            />
            <input type="date" value={fromDate} aria-label="From date filter"
              style={{ padding: '0.45rem 0.75rem', borderRadius: '6px', border: '1px solid var(--clr-border)', background: 'var(--clr-surface)', color: 'var(--clr-text-primary)', fontFamily: 'inherit' }}
              onChange={e => { setFrom(e.target.value); setPage(1); }}
            />
            <input type="date" value={toDate} aria-label="To date filter"
              style={{ padding: '0.45rem 0.75rem', borderRadius: '6px', border: '1px solid var(--clr-border)', background: 'var(--clr-surface)', color: 'var(--clr-text-primary)', fontFamily: 'inherit' }}
              onChange={e => { setTo(e.target.value); setPage(1); }}
            />
            {/* Hide 'Assigned to me' in demo — irrelevant there */}
            {isReviewer && !isDemoSession && (
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--clr-text-secondary)', cursor: 'pointer' }}>
                <input 
                  type="checkbox" 
                  checked={assignedOnly} 
                  onChange={e => { setAssignedOnly(e.target.checked); setPage(1); }} 
                />
                Assigned to me
              </label>
            )}
            {(status || minRisk || fromDate || toDate) && (
              <button className="new-analysis-btn" style={{ background: 'rgba(100,116,139,0.2)', fontSize: '0.78rem', padding: '0.45rem 0.85rem' }} onClick={clearFilters}>
                ✕ Clear
              </button>
            )}
          </div>

          {renderContent()}
        </div>
      </main>

      {selected && (
        <div className="drawer-overlay" onClick={() => setSelected(null)} role="dialog" aria-modal="true" aria-label={`Case ${selected.id} details`}>
          <div className="drawer" onClick={e => e.stopPropagation()}>
            <div className="drawer-header">
              <h3>Case #{selected.id}</h3>
              <button onClick={() => setSelected(null)} aria-label="Close case detail">✕</button>
            </div>
            <div className="drawer-body">
              {/* Assignment identity row */}
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', padding: '0.6rem 0.75rem', background: 'var(--clr-surface-2)', borderRadius: '8px', fontSize: '0.78rem', fontFamily: 'monospace', flexWrap: 'wrap' }}>
                <span title="Coder who created this case">
                  🖊 Coder: <strong>#{selected.coder_id ?? '—'}</strong>
                  {selected.creator_name && <span style={{ fontFamily: 'inherit', fontWeight: 400, color: 'var(--clr-text-muted)', marginLeft: '0.3rem' }}>({selected.creator_name})</span>}
                </span>
                <span style={{ color: 'var(--clr-border)' }}>|</span>
                <span title="Reviewer assigned to this case">
                  👁 Reviewer: <strong>#{selected.reviewer_id ?? '—'}</strong>
                  {selected.assigned_reviewer_name && selected.assigned_reviewer_name !== 'Unassigned' && (
                    <span style={{ fontFamily: 'inherit', fontWeight: 400, color: 'var(--clr-text-muted)', marginLeft: '0.3rem' }}>({selected.assigned_reviewer_name})</span>
                  )}
                </span>
              </div>

              <div className="case-metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <div className="metric-box">
                  <span className="label">Risk Score</span>
                  <span className={`value risk-${selected.risk_score >= 70 ? 'high' : selected.risk_score >= 30 ? 'medium' : 'low'}`}>
                    {selected.risk_score?.toFixed(1) ?? '—'}
                  </span>
                </div>
                <div className="metric-box">
                  <span className="label">Accuracy</span>
                  <span className="value">{selected.coding_accuracy?.toFixed(1) ?? '—'}%</span>
                </div>
                <div className="metric-box">
                  <span className="label">Revenue Impact</span>
                  <span className="value text-success">${(selected.revenue_impact || 0).toFixed(0)}</span>
                </div>
                <div className="metric-box">
                  <span className="label">AI Confidence</span>
                  <span className="value">{(selected.avg_confidence * 100).toFixed(0)}%</span>
                </div>
              </div>

              <div className="case-detail-section">
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <Activity size={18} color="var(--clr-primary)" />
                  Clinical Audit Traceability
                </h4>
                
                <div className="trace-tabs" style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid var(--clr-border)', marginBottom: '1rem' }}>
                  <button 
                    className={`trace-tab ${!selected._activeTab || selected._activeTab === 'accepted' ? 'active' : ''}`}
                    onClick={() => setSelected({...selected, _activeTab: 'accepted'})}
                    style={{ padding: '0.5rem 0.25rem', background: 'none', border: 'none', borderBottom: (!selected._activeTab || selected._activeTab === 'accepted') ? '2px solid var(--clr-primary)' : 'none', color: (!selected._activeTab || selected._activeTab === 'accepted') ? 'var(--clr-primary)' : 'var(--clr-text-muted)', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}
                  >
                    Accepted Codes ({selected.ai_codes?.length || 0})
                  </button>
                  <button 
                    className={`trace-tab ${selected._activeTab === 'removed' ? 'active' : ''}`}
                    onClick={() => setSelected({...selected, _activeTab: 'removed'})}
                    style={{ padding: '0.5rem 0.25rem', background: 'none', border: 'none', borderBottom: selected._activeTab === 'removed' ? '2px solid #ef4444' : 'none', color: selected._activeTab === 'removed' ? '#ef4444' : 'var(--clr-text-muted)', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 600 }}
                  >
                    Removed/Rejected ({selected.removed_codes?.length || 0})
                  </button>
                </div>

                {(!selected._activeTab || selected._activeTab === 'accepted') && (
                  <div className="fadein">
                    <CodeExplainabilityPanel codes={selected.ai_codes || []} />
                  </div>
                )}
                
                {selected._activeTab === 'removed' && (
                  <div className="fadein">
                    <RemovedCodesPanel removed={selected.removed_codes || []} />
                  </div>
                )}
              </div>

              <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--clr-surface-2)', borderRadius: '12px', border: '1px solid var(--clr-border)' }}>
                <h5 style={{ marginTop: 0, marginBottom: '0.75rem', fontSize: '0.9rem', color: 'var(--clr-text-primary)' }}>Clinical Summary & Explanation</h5>
                <p style={{ fontSize: '0.88rem', lineHeight: 1.6, color: 'var(--clr-text-muted)', marginBottom: '1rem' }}>{selected.summary || '—'}</p>
                {selected.explanation && (
                  <div style={{ padding: '0.75rem', background: 'rgba(99, 102, 241, 0.05)', borderRadius: '8px', borderLeft: '3px solid var(--clr-primary)' }}>
                    <p style={{ fontSize: '0.85rem', margin: 0, color: 'var(--clr-text-primary)', fontStyle: 'italic' }}>"{selected.explanation}"</p>
                  </div>
                )}
              </div>

              <div style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
                <p style={{ fontSize: '0.85rem' }}><strong>Input Note:</strong> {selected.input_text ? selected.input_text.slice(0, 500) + (selected.input_text.length > 500 ? '…' : '') : '—'}</p>
                <p style={{ fontSize: '0.85rem' }}><strong>Human Codes:</strong> {(selected.human_codes || []).join(', ') || '—'}</p>
              </div>
              
              {selected.locked_by && (
                <div style={{ marginTop: '1rem', padding: '0.75rem', background: selected.lock_is_stale ? '#fffbeb' : '#fef2f2', border: selected.lock_is_stale ? '1px solid #fef3c7' : '1px solid #fee2e2', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ color: selected.lock_is_stale ? '#d97706' : '#ef4444' }}>{selected.lock_is_stale ? '🕒' : '🔒'}</span>
                  <span style={{ fontSize: '0.85rem', color: selected.lock_is_stale ? '#92400e' : '#991b1b' }}>
                    {selected.lock_is_stale ? 'Stale Lock: ' : 'Locked by '}
                    <strong>{selected.locker_name || `User ${selected.locked_by}`}</strong> 
                    {selected.lock_is_stale ? ' (Auto-release on open)' : ` since ${new Date(selected.locked_at).toLocaleTimeString()}`}
                  </span>
                </div>
              )}

              {isAdmin && (
                <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--clr-surface-2)', borderRadius: '8px' }}>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 600 }}>Case Priority</label>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {['low', 'normal', 'high'].map(p => (
                      <button
                        key={p}
                        className={`priority-btn ${p} ${selected.priority === p ? 'active' : ''}`}
                        onClick={async () => {
                          try {
                            await caseApi.update(selected.id, { priority: p });
                            const res = await caseApi.get(selected.id);
                            setSelected(res.data);
                            fetchCases();
                          } catch (e) { alert('Update failed'); }
                        }}
                      >
                        {p.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Reviewer assignment panel — hidden in demo (auto-assigned) */}
              {isAdmin && !isDemoSession && selected.status === 'submitted' && (
                <ReviewerAssignmentPanel 
                  caseData={selected}
                  onActionComplete={() => {
                    fetchCases();
                    caseApi.get(selected.id).then(res => setSelected(res.data));
                  }}
                />
              )}

              {isReviewer ? (
                <ReviewerActionPanel
                  caseData={selected}
                  onActionComplete={() => {
                    fetchCases();
                    caseApi.get(selected.id).then(res => setSelected(res.data));
                  }}
                />
              ) : (
                selected.reviewer_notes && (
                  <div style={{ marginTop: '0.75rem', padding: '0.9rem', background: 'var(--clr-surface-2)', borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
                    <p style={{ color: 'var(--clr-text-muted)', marginBottom: '0.25rem', fontSize: '0.75rem' }}>
                      Reviewer Notes {selected.reviewed_at ? `(${new Date(selected.reviewed_at).toLocaleDateString()})` : ''}
                    </p>
                    <p style={{ color: 'var(--clr-text-primary)' }}>{selected.reviewer_notes}</p>
                  </div>
                )
              )}

              {auditTrail.length > 0 && (
                <div className="audit-timeline">
                  <h4>Activity Timeline</h4>
                  <div className="timeline-items">
                    {auditTrail.map((log, i) => (
                      <div key={i} className="timeline-item">
                        <div className="timeline-marker" />
                        <div className="timeline-content">
                          <div className="timeline-header">
                            <strong>{log.action.toUpperCase()}</strong>
                            <span>{new Date(log.timestamp).toLocaleString()}</span>
                          </div>
                          <div className="timeline-user">by {log.user} ({log.role || 'user'})</div>
                          {(log.previous_state || log.new_state) && (
                            <div className="timeline-changes">
                              {log.action === 'update' && (
                                <div className="code-diff">
                                  <div className="diff-item prev">
                                    <span>Was:</span> {log.previous_state?.codes?.map(c => c.code).join(', ') || '—'}
                                  </div>
                                  <div className="diff-item next">
                                    <span>Now:</span> {log.new_state?.codes?.map(c => c.code).join(', ') || '—'}
                                  </div>
                                </div>
                              )}
                              {log.action === 'assignment' && (
                                <div className="detail-item">Assigned to: {log.metadata?.reviewer_name}</div>
                              )}
                              {log.metadata?.confidence && (
                                <div className="detail-item">Confidence: {(log.metadata.confidence * 100).toFixed(0)}%</div>
                              )}
                            </div>
                          )}
                          {log.metadata?.justification && <div className="timeline-details">"{log.metadata.justification}"</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="drawer-actions">
                {isAdmin && (selected.status === 'approved' || selected.status === 'rejected') && (
                  <button 
                    className="reopen-btn"
                    onClick={async () => {
                      if (window.confirm('Re-open this case for review?')) {
                        try {
                          await caseApi.reopen(selected.id);
                          fetchCases();
                          setSelected(null);
                        } catch (e) {
                          const detail = e.response?.data?.detail;
                          const msg = typeof detail === 'string' ? detail : JSON.stringify(detail) || e.message || 'Re-open failed';
                          alert('Re-open failed: ' + msg);
                        }
                      }
                    }}
                  >
                    ↺ Admin Override: Re-open
                  </button>
                )}
                
                {isCoder && selected.status === 'draft' && (
                  <button
                    className="new-analysis-btn"
                    onClick={() => navigate('/', { state: { caseData: selected } })}
                  >
                    Re-open in editor
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
