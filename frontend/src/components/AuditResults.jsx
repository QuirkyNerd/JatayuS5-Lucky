import React, { useState } from 'react';
import { auditApi } from '../services/api.js';
import './AuditResults.css';


function ConfidenceBar({ value }) {
  const pct  = Math.round(value * 100);
  const color = pct >= 85 ? '#10b981' : pct >= 65 ? '#f59e0b' : '#ef4444';
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar" style={{ width: `${pct}%`, background: color }} />
      <span style={{ color }}>{pct}%</span>
    </div>
  );
}

function CodeCard({ code, noteHash }) {
  const [feedback, setFeedback] = useState(null);   
  const [sending, setSending]   = useState(false);

  const sendFeedback = async (decision) => {
    if (feedback || sending) return;
    setSending(true);
    try {
      await auditApi.submitFeedback({ note_hash: noteHash, ai_code: code.code, decision });
      setFeedback(decision);
    } catch {
      alert('Failed to send feedback. Please try again.');
    } finally {
      setSending(false);
    }
  };

  const typeColor = code.type === 'ICD-10' ? '#6366f1' : code.type === 'CPT' ? '#10b981' : '#f59e0b';

  return (
    <div className={`code-card ${feedback ? `feedback-${feedback}` : ''}`}>
      <div className="code-card-top">
        <div className="code-main">
          <span className="code-code">{code.code}</span>
          <span className="code-type" style={{ background: `${typeColor}22`, color: typeColor, border: `1px solid ${typeColor}44` }}>
            {code.type}
          </span>
          {code.source && (
            <span className={`code-source ${code.source}`}>{code.source}</span>
          )}
        </div>
        <div className="feedback-btns">
          {feedback ? (
            <span className="feedback-done">{feedback === 'accepted' ? 'Accepted' : 'Rejected'}</span>
          ) : (
            <>
              <button className="fb-btn accept" onClick={() => sendFeedback('accepted')} disabled={sending} title="Accept this code">✓</button>
              <button className="fb-btn reject" onClick={() => sendFeedback('rejected')} disabled={sending} title="Reject this code">✕</button>
            </>
          )}
        </div>
      </div>

      <div className="code-desc">{code.description}</div>
      <ConfidenceBar value={code.confidence} />

      {code.rationale && (
        <div className="code-rationale">
          <span className="rationale-label">Clinical rationale</span>
          <p>{code.rationale}</p>
        </div>
      )}

      {code.guideline && (
        <div className="code-guideline">
          <span className="guideline-label">Coding Guideline: {code.guideline.source}</span>
          <p>"{code.guideline.text}"</p>
        </div>
      )}
    </div>
  );
}

function DiscrepancyCard({ disc }) {
  const TYPE_MAP = {
    missed_code:     { label: 'MISSED CODE',     cls: 'disc-missed',     tip: 'Code identified by audit but not submitted by the human coder. Potential revenue leakage.' },
    unsupported_code:{ label: 'UNSUPPORTED',      cls: 'disc-unsupported', tip: 'Code submitted by the human coder without sufficient clinical documentation support.' },
    correct_code:    { label: 'CORRECT',          cls: 'disc-correct',    tip: 'Code correctly submitted by the human coder and confirmed by audit.' },
    extra_code:      { label: 'EXTRA CODE',       cls: 'disc-extra',      tip: 'Code added by audit engine but not submitted by the human coder.' },
  };

  const meta = TYPE_MAP[disc.type] || { label: disc.type?.toUpperCase(), cls: '', tip: '' };

  return (
    <div className={`disc-card ${meta.cls}`}>
      <div className="disc-header">
        <span className="disc-code">{disc.code}</span>
        <span className={`disc-badge ${meta.cls}`}>{meta.label}</span>
        {disc.risk_level && (
          <span className={`risk-badge risk-${disc.risk_level}`}>{disc.risk_level.toUpperCase()} RISK</span>
        )}
      </div>
      <p className="disc-msg">{disc.message}</p>
      {disc.guideline && (
        <div className="disc-guideline">
          <span className="guideline-label">Guideline: {disc.guideline.source}</span>
          <p>"{disc.guideline.text}"</p>
        </div>
      )}
      {meta.tip && <p className="disc-tip">{meta.tip}</p>}
    </div>
  );
}


function PipelineTracePanel({ trace }) {
  if (!trace || trace.length === 0) return <p className="empty-state">No pipeline trace available.</p>;
  return (
    <div className="pipeline-trace-panel">
      {trace.map((step, i) => (
        <div key={i} className="trace-step">
          <div className="trace-header">
            <span className="trace-label">{step.label}</span>
            <span className="trace-status">{step.status.toUpperCase()}</span>
          </div>
          <div className="trace-metrics">
            <span className="trace-count">{step.input_count} → {step.output_count}</span>
            <span className="trace-summary">{step.changes}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Step 5, 6: Code Explainability Panel — Clinical Audit Traceability ──────
export function CodeExplainabilityPanel({ codes }) {
  const [expanded, setExpanded] = React.useState({});
  if (!codes || codes.length === 0)
    return <p className="empty-state">No explainability data available.</p>;

  const toggle = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }));

  return (
    <div className="explainability-panel">
      {codes.map((code, i) => {
        const exp   = code.audit_explanation || {};
        const bd    = exp.scoring_breakdown || code.scoring_breakdown || {};
        const conf  = exp.confidence_score ?? code.confidence ?? 0;
        const confPct = Math.round(conf * 100);
        const strength = exp.evidence_strength ?? code.evidence_strength ?? 0;
        const isOpen = expanded[i];

        // Step 6: colour logic
        const statusColor  = confPct >= 80 ? '#10b981' : confPct >= 55 ? '#f59e0b' : '#ef4444';
        const statusLabel  = confPct >= 80 ? 'HIGH CONFIDENCE' : confPct >= 55 ? 'MODERATE' : 'LOW CONFIDENCE';
        const statusBadgeCls = confPct >= 80 ? 'exp-badge-accepted' : confPct >= 55 ? 'exp-badge-moderate' : 'exp-badge-low';

        const evidenceSources  = exp.evidence_sources  || [];
        const matchedSections  = exp.matched_sections  || [];
        const traceHistory     = exp.trace_history     || [];
        const humanRationale   = exp.human_rationale   || code.rationale || '';
        const tier             = exp.calibration_tier  || code.calibration_tier || '';
        const anatomyMatch     = exp.anatomy_match     || '';
        const relMatch         = exp.relationship_match || '';
        const specReason       = exp.specificity_reason || '';

        const scoreItems = [
          { label: 'Evidence',     val: bd.evidence_score,     color: '#6366f1' },
          { label: 'Section',      val: bd.section_score,      color: '#8b5cf6' },
          { label: 'Anatomy',      val: bd.anatomy_score,      color: '#10b981' },
          { label: 'Relationship', val: bd.relationship_score, color: '#f59e0b' },
          { label: 'Specificity',  val: bd.specificity_score,  color: '#06b6d4' },
        ].filter(s => s.val !== undefined && s.val !== null);

        return (
          <div key={i} className="exp-card">
            {/* Header row */}
            <div className="exp-card-header" onClick={() => toggle(i)} style={{ cursor: 'pointer' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: 1 }}>
                <span className="exp-code">{code.code}</span>
                <span className="exp-desc">{code.description}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {/* Step 6: accepted badge */}
                <span className={`exp-status-badge ${statusBadgeCls}`}>
                  ✓ ACCEPTED · {statusLabel}
                </span>
                <span style={{ color: statusColor, fontWeight: 700, fontSize: '0.9rem' }}>
                  {confPct}%
                </span>
                <span style={{ color: '#64748b', fontSize: '0.8rem' }}>{isOpen ? '▲' : '▼'}</span>
              </div>
            </div>

            {/* Human rationale always visible */}
            {humanRationale && (
              <div className="exp-rationale-bar">
                <span className="exp-rationale-icon">✦</span>
                <span className="exp-rationale-text">{humanRationale}</span>
              </div>
            )}

            {/* Expandable detail */}
            {isOpen && (
              <div className="exp-detail">
                {/* Evidence sources */}
                {evidenceSources.length > 0 && (
                  <div className="exp-row">
                    <span className="exp-label">Evidence Sources</span>
                    <div className="exp-tag-group">
                      {evidenceSources.map((s, j) => (
                        <span key={j} className="exp-tag exp-tag-green">{s.replace(/_/g,' ')}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Matched sections */}
                {matchedSections.length > 0 && (
                  <div className="exp-row">
                    <span className="exp-label">Clinical Section</span>
                    <div className="exp-tag-group">
                      {matchedSections.map((s, j) => (
                        <span key={j} className="exp-tag exp-tag-purple">{s}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Anatomy / Relationship / Specificity */}
                <div className="exp-meta-row">
                  {anatomyMatch && (
                    <div className="exp-meta-item">
                      <span className="exp-label">Anatomy</span>
                      <span className={`exp-tag ${anatomyMatch === 'confirmed' ? 'exp-tag-green' : 'exp-tag-gray'}`}>
                        {anatomyMatch}
                      </span>
                    </div>
                  )}
                  {relMatch && (
                    <div className="exp-meta-item">
                      <span className="exp-label">Relationship</span>
                      <span className={`exp-tag ${relMatch === 'reinforced' ? 'exp-tag-green' : 'exp-tag-gray'}`}>
                        {relMatch}
                      </span>
                    </div>
                  )}
                  {tier && (
                    <div className="exp-meta-item">
                      <span className="exp-label">Tier</span>
                      <span className="exp-tag exp-tag-blue">{tier.replace(/_/g,' ')}</span>
                    </div>
                  )}
                </div>

                {/* Specificity */}
                {specReason && (
                  <div className="exp-row">
                    <span className="exp-label">Specificity</span>
                    <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>{specReason}</span>
                  </div>
                )}

                {/* Scoring breakdown bars */}
                {scoreItems.length > 0 && (
                  <div className="exp-scores">
                    <span className="exp-label" style={{ marginBottom: '0.4rem', display:'block' }}>Scoring Dimensions</span>
                    {scoreItems.map((s, j) => {
                      const pct = Math.round((s.val || 0) * 100);
                      return (
                        <div key={j} className="exp-score-row">
                          <span className="exp-score-lbl">{s.label}</span>
                          <div className="exp-score-bar-wrap">
                            <div className="exp-score-bar" style={{ width: `${pct}%`, background: s.color }} />
                          </div>
                          <span className="exp-score-pct" style={{ color: s.color }}>{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Trace history */}
                {traceHistory.length > 0 && (
                  <div className="exp-trace">
                    <span className="exp-label">Reasoning Trace</span>
                    <ol className="exp-trace-list">
                      {traceHistory.map((step, j) => (
                        <li key={j} className="exp-trace-step">{step}</li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Step 5: Removed Codes Panel — Rejection Trace ────────────────────────────
export function RemovedCodesPanel({ removed }) {
  const [expanded, setExpanded] = React.useState({});
  if (!removed || removed.length === 0)
    return <p className="empty-state">No codes were removed during clinical reasoning.</p>;

  const toggle = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }));

  // Normalise: removed items may be plain {code,description,reason} or full rejection_trace dicts
  const normalise = (item) => ({
    code:             item.code        || '',
    description:      item.description || '',
    rejection_stage:  item.rejection_stage  || 'clinical_reasoning',
    rejection_reason: item.rejection_reason || item.reason || 'insufficient_evidence',
    failed_dimension: item.failed_dimension || 'evidence',
    threshold:        item.threshold    ?? null,
    actual_score:     item.actual_score ?? null,
    calibration_tier: item.calibration_tier || '',
    human_rationale:  item.human_rationale  || item.reason || '',
  });

  return (
    <div className="removed-codes-panel">
      {removed.map((raw, i) => {
        const item    = normalise(raw);
        const isOpen  = expanded[i];
        const stageLabel = item.rejection_stage.replace(/_/g, ' ').toUpperCase();
        const reasonLabel = item.rejection_reason.replace(/_/g, ' ');

        return (
          <div key={i} className="exp-card exp-card-rejected">
            <div className="exp-card-header" onClick={() => toggle(i)} style={{ cursor: 'pointer' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: 1 }}>
                <span className="exp-code">{item.code}</span>
                <span className="exp-desc">{item.description}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {/* Step 6: rejected badge */}
                <span className="exp-status-badge exp-badge-rejected">✕ REJECTED</span>
                <span style={{ color: '#64748b', fontSize: '0.8rem' }}>{isOpen ? '▲' : '▼'}</span>
              </div>
            </div>

            {/* Human rationale always visible */}
            {item.human_rationale && (
              <div className="exp-rationale-bar exp-rationale-rejected">
                <span className="exp-rationale-icon">⚑</span>
                <span className="exp-rationale-text">{item.human_rationale}</span>
              </div>
            )}

            {isOpen && (
              <div className="exp-detail">
                <div className="exp-meta-row">
                  <div className="exp-meta-item">
                    <span className="exp-label">Rejection Stage</span>
                    <span className="exp-tag exp-tag-red">{stageLabel}</span>
                  </div>
                  <div className="exp-meta-item">
                    <span className="exp-label">Failed Dimension</span>
                    <span className="exp-tag exp-tag-red">{item.failed_dimension}</span>
                  </div>
                  {item.calibration_tier && (
                    <div className="exp-meta-item">
                      <span className="exp-label">Tier</span>
                      <span className="exp-tag exp-tag-gray">{item.calibration_tier.replace(/_/g,' ')}</span>
                    </div>
                  )}
                </div>
                {item.threshold !== null && (
                  <div className="exp-threshold-row">
                    <span className="exp-label">Evidence Gate</span>
                    <span className="exp-threshold-info">
                      Score <strong>{item.actual_score?.toFixed(2)}</strong> vs threshold <strong>{item.threshold?.toFixed(2)}</strong>
                    </span>
                  </div>
                )}
                <div className="exp-row">
                  <span className="exp-label">Reason</span>
                  <span className="exp-tag exp-tag-amber">{reasonLabel}</span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}


function FinancialImpactPanel({ metrics }) {
  if (!metrics) return <p className="empty-state">Impact analysis not available.</p>;
  
  const { total_revenue_impact, total_exposure, risk_level, impact_summary, financial_breakdown, disclaimer } = metrics;
  
  return (
    <div className="financial-impact-panel">
      <div className="impact-grid">
        <div className="impact-main">
          <span className="impact-label">Total Estimated Revenue Impact</span>
          <span className="impact-value">${total_revenue_impact.toLocaleString()}</span>
        </div>
        <div className="impact-secondary">
          <div className="impact-sub">
            <span className="impact-label">Risk Exposure</span>
            <span className="impact-value-sub">${total_exposure.toLocaleString()}</span>
          </div>
          <div className="impact-sub">
            <span className="impact-label">Risk Level</span>
            <span className={`impact-level level-${risk_level}`}>{risk_level.toUpperCase()}</span>
          </div>
        </div>
      </div>
      
      <div className="impact-summary-box">
        <span className="summary-label">Analysis Summary</span>
        <p className="summary-text">{impact_summary}</p>
      </div>

      {financial_breakdown && financial_breakdown.length > 0 && (
        <div className="impact-breakdown">
          <table className="impact-table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Type</th>
                <th>Impact/Risk</th>
                <th>Justification</th>
              </tr>
            </thead>
            <tbody>
              {financial_breakdown.map((item, i) => (
                <tr key={i}>
                  <td className="imp-code">{item.code}</td>
                  <td><span className={`imp-type type-${item.type.toLowerCase()}`}>{item.type}</span></td>
                  <td className="imp-val">${item.impact.toLocaleString()}</td>
                  <td className="imp-reason">{item.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="impact-disclaimer">
        <span className="disclaimer-icon">ⓘ</span> {disclaimer}
      </div>
    </div>
  );
}


function SimulationPanel({ originalCodes }) {
  const [modifiedCodes, setModifiedCodes] = useState([...originalCodes]);
  const [newCode, setNewCode] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const runSimulation = async (codes) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_codes: originalCodes,
          modified_codes: codes
        })
      });
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Simulation failed', err);
    } finally {
      setLoading(false);
    }
  };

  const removeCode = (code) => {
    const updated = modifiedCodes.filter(c => c !== code);
    setModifiedCodes(updated);
    runSimulation(updated);
  };

  const addCode = () => {
    if (!newCode || modifiedCodes.includes(newCode.toUpperCase())) return;
    const updated = [...modifiedCodes, newCode.toUpperCase()];
    setModifiedCodes(updated);
    setNewCode('');
    runSimulation(updated);
  };

  return (
    <div className="simulation-panel">
      <div className="sim-controls">
        <div className="sim-input-group">
          <input 
            type="text" 
            placeholder="Add CPT/ICD code..." 
            value={newCode}
            onChange={(e) => setNewCode(e.target.value.toUpperCase())}
            onKeyPress={(e) => e.key === 'Enter' && addCode()}
          />
          <button className="btn-add" onClick={addCode}>Add</button>
        </div>
        
        <div className="sim-code-list">
          {modifiedCodes.map((c, i) => (
            <div key={i} className="sim-code-tag">
              <span>{c}</span>
              <button className="btn-rm" onClick={() => removeCode(c)}>✕</button>
            </div>
          ))}
        </div>
      </div>

      {loading && <div className="sim-loading">Calculating impact...</div>}

      {results && (
        <div className="sim-results fadein">
          <div className="sim-metric-row">
            <div className="sim-metric">
              <span className="sim-label">Financial Impact</span>
              <span className={`sim-val ${results.delta_revenue >= 0 ? 'text-success' : 'text-danger'}`}>
                {results.delta_revenue >= 0 ? '+' : ''}${results.delta_revenue.toLocaleString()}
              </span>
            </div>
            <div className="sim-metric">
              <span className="sim-label">Risk Level Change</span>
              <span className={`sim-risk risk-${results.risk_change}`}>
                {results.risk_change.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>

          <div className="sim-summary-box">
            <p className="sim-summary">{results.summary}</p>
          </div>

          {results.compliance_flags.length > 0 && (
            <div className="sim-flags">
              <span className="flags-label">Compliance Notes</span>
              <ul>
                {results.compliance_flags.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}

          <div className="sim-disclaimer">
            <span className="disclaimer-icon">ⓘ</span> {results.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}


export default function AuditResults({ result, noteHash }) {
  const [activeTab, setActiveTab] = useState('explanation');

  if (!result) return null;

  const ai_codes             = Array.isArray(result.ai_codes)             ? result.ai_codes             : [];
  const low_confidence_codes = Array.isArray(result.low_confidence_codes) ? result.low_confidence_codes : [];
  const discrepancies        = Array.isArray(result.discrepancies)        ? result.discrepancies        : [];
  const evidence             = Array.isArray(result.evidence)             ? result.evidence             : [];
  const summary              = result.summary       || '';
  const pipeline_trace       = Array.isArray(result.pipeline_trace)      ? result.pipeline_trace       : [];
  const removed_codes        = Array.isArray(result.removed_codes)       ? result.removed_codes        : [];
  const impact_metrics       = result.impact_metrics                     || null;
  const tokens_used          = result.tokens_used   || 0;
  const explanation          = result.explanation   || '';

  const missedCount    = discrepancies.filter(d => d.type === 'missed_code').length;
  const unsupported    = discrepancies.filter(d => d.type === 'unsupported_code').length;
  const correctCount   = discrepancies.filter(d => d.type === 'correct_code').length;
  const totalCodes     = ai_codes.length;
  const avgConfidence  = totalCodes > 0 ? Math.round(ai_codes.reduce((s, c) => s + c.confidence, 0) / totalCodes * 100) : 0;

  const TABS = [
    { id: 'explanation', label: 'Summary' },
    { id: 'codes', label: `Codes (${totalCodes})` },
    { id: 'explainability', label: 'Explainability' },
    { id: 'removed', label: `Removed (${removed_codes.length})` },
    { id: 'evidence', label: 'Evidence' },
  ];

  return (
    <div className="audit-results fadein">
      <div className="audit-banner">
        <div className="banner-title">Audit Complete</div>
        <p className="banner-summary">{summary}</p>
        <div className="banner-metrics">
          <div className="metric">
            <span className="metric-val">{totalCodes}</span>
            <span className="metric-lbl">AI Codes</span>
          </div>
          <div className="metric">
            <span className="metric-val text-success">{correctCount}</span>
            <span className="metric-lbl">Correct</span>
          </div>
          <div className="metric">
            <span className="metric-val text-danger">{missedCount}</span>
            <span className="metric-lbl">Missed</span>
          </div>
          <div className="metric">
            <span className="metric-val text-warning">{unsupported}</span>
            <span className="metric-lbl">Unsupported</span>
          </div>
          <div className="metric">
            <span className="metric-val text-info">{avgConfidence}%</span>
            <span className="metric-lbl">Avg Confidence</span>
          </div>
        </div>
      </div>

      {missedCount > 0 && (
        <div className="missed-alert">
          <strong>{missedCount} code(s) were missed by the human coder.</strong> Review these to prevent revenue leakage.
        </div>
      )}

      <div className="audit-tabs">
        {TABS.map(t => (
          <button key={t.id} className={`audit-tab ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="tab-content">

        {activeTab === 'explanation' && (
          <div className="explanation-panel">
            <div className="explanation-header">
              <span>Clinical Coding Audit Explanation</span>
              <span className="explanation-badge">CDI Analysis</span>
            </div>
            {explanation ? (
              <div className="explanation-body">
                {explanation.split('\n').filter(Boolean).map((para, i) => (
                  <p key={i}>{para}</p>
                ))}
              </div>
            ) : (
              <p className="empty-state">Clinical explanation not available for this audit.</p>
            )}
          </div>
        )}

        {activeTab === 'codes' && (
          <div>
            {ai_codes.length === 0 && <p className="empty-state">No codes were generated. Try a more detailed clinical note.</p>}
            {ai_codes.map((c, i) => <CodeCard key={i} code={c} noteHash={noteHash} />)}
            {low_confidence_codes.length > 0 && (
              <>
                <div className="section-divider">Low Confidence Codes (below threshold)</div>
                {low_confidence_codes.map((c, i) => <CodeCard key={i} code={c} noteHash={noteHash} />)}
              </>
            )}
          </div>
        )}

        {activeTab === 'explainability' && (
          <CodeExplainabilityPanel codes={ai_codes} />
        )}

        {activeTab === 'removed' && (
          <RemovedCodesPanel removed={removed_codes} />
        )}

        {activeTab === 'evidence' && (
          <div>
            {evidence.length === 0 && <p className="empty-state">No evidence mapped.</p>}
            {evidence.map((ev, i) => (
              <div key={i} className="evidence-card">
                <div className="ev-header">
                  <span className="ev-code">{ev.code}</span>
                  <span className={`ev-strength strength-${ev.strength || 'medium'}`}>{ev.strength?.toUpperCase() || 'MEDIUM'}</span>
                </div>
                <blockquote className="ev-text">{ev.sentence_text}</blockquote>
                {ev.start_char != null && (
                  <span className="ev-span">Offsets: {ev.start_char}–{ev.end_char}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
