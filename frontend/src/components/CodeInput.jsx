import React, { useState, useMemo, useCallback } from 'react';
import { getContextualCodes, ICD_CODE_POOL } from '../data/icdCodePool.js';

function getCodeLabel(code) {
  const entry = ICD_CODE_POOL.find(c => c.code === code);
  return entry ? `${entry.code} – ${entry.description}` : code;
}

export default function CodeInput({ codes = [], onChange = () => {}, noteText = '', disabled = false }) {
  const [inputVal, setInputVal] = useState('');

  const quickCodes = useMemo(() => {
    return getContextualCodes(noteText, 5);
  }, [noteText]);

  const addCode = useCallback((raw) => {
    const trimmed = raw.trim().toUpperCase();
    const safeCodes = Array.isArray(codes) ? codes : [];
    if (trimmed && !safeCodes.includes(trimmed)) {
      onChange([...safeCodes, trimmed]);
    }
    setInputVal('');
  }, [codes, onChange]);

  const removeCode = useCallback((code) => {
    onChange((Array.isArray(codes) ? codes : []).filter(c => c !== code));
  }, [codes, onChange]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addCode(inputVal);
    } else if (e.key === 'Backspace' && !inputVal && (codes?.length || 0) > 0) {
      const safeCodes = Array.isArray(codes) ? codes : [];
      onChange(safeCodes.slice(0, -1));
    }
  }, [addCode, inputVal, codes, onChange]);

  const hasContextNote = noteText && noteText.trim().length >= 20;

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '1.1rem' }} aria-hidden="true"></span>
          <h2 style={{ fontSize: '1rem', fontWeight: 600, margin: 0, color: 'var(--clr-text-primary)' }}>
            Human-Entered Codes
          </h2>
        </div>
        {(codes?.length || 0) > 0 && (
          <span style={{ fontSize: '0.72rem', color: 'var(--clr-text-muted)' }}>
            {codes.length} code{codes.length !== 1 ? 's' : ''} entered
          </span>
        )}
      </div>

      <p style={{ fontSize: '0.78rem', color: 'var(--clr-text-muted)', margin: 0 }}>
        Enter the codes provided by the human coder.&nbsp;
        Press <kbd style={{ background: 'var(--clr-surface-2)', border: '1px solid var(--clr-border)', borderRadius: 3, padding: '0 4px', fontFamily: 'monospace', fontSize: '0.75rem' }}>Enter</kbd> or
        <kbd style={{ background: 'var(--clr-surface-2)', border: '1px solid var(--clr-border)', borderRadius: 3, padding: '0 4px', fontFamily: 'monospace', fontSize: '0.75rem' }}>,</kbd> after each code.
      </p>

      <div
        role="listbox"
        aria-label="Entered codes"
        style={{
          display: 'flex', flexWrap: 'wrap', gap: '0.5rem',
          background: 'var(--clr-surface-2)', border: '1px solid var(--clr-border)',
          borderRadius: 'var(--radius-md)', padding: '0.75rem', minHeight: '3.5rem', cursor: 'text',
        }}
        onClick={() => document.getElementById('code-input-field')?.focus()}
      >
        {(Array.isArray(codes) ? codes : []).map(code => (
          <span
            key={code}
            role="option"
            aria-selected="true"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.35rem',
              background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.4)',
              borderRadius: '6px', color: 'var(--clr-primary)',
              fontFamily: "'JetBrains Mono', monospace", fontSize: '0.82rem', fontWeight: 500,
              padding: '0.2rem 0.6rem',
            }}
          >
            {code}
            <button
              type="button"
              onClick={(e) => { if (!disabled) { e.stopPropagation(); removeCode(code); } }}
              disabled={disabled}
              style={{ background: 'none', border: 'none', color: 'inherit', cursor: disabled ? 'default' : 'pointer', fontSize: '0.9rem', lineHeight: 1, padding: 0, opacity: disabled ? 0.3 : 0.7 }}
              aria-label={`Remove code ${code}`}
            >×</button>
          </span>
        ))}

        <input
          id="code-input-field"
          type="text"
          value={inputVal}
          onChange={e => setInputVal(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => !disabled && inputVal.trim() && addCode(inputVal)}
          disabled={disabled}
          placeholder={(codes?.length || 0) === 0 ? 'Type a code, e.g. I10' : '+ Add code'}
          aria-label="Enter ICD-10 or CPT code"
          style={{
            background: 'none', border: 'none', color: 'var(--clr-text-primary)',
            fontFamily: "'JetBrains Mono', monospace", fontSize: '0.85rem',
            outline: 'none', minWidth: '120px', flex: 1,
          }}
        />
      </div>

      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--clr-text-muted)' }}>
            {hasContextNote ? 'Suggestions:' : 'Quick add:'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          {quickCodes.map(code => {
            const alreadyAdded = (Array.isArray(codes) ? codes : []).includes(code);
            const label = getCodeLabel(code);
            return (
              <button
                key={code}
                type="button"
                onClick={() => !disabled && addCode(code)}
                disabled={disabled || alreadyAdded}
                title={label}
                aria-label={`Quick add: ${label}`}
                style={{
                  background: alreadyAdded ? 'var(--clr-surface-2)' : 'rgba(99,102,241,0.07)',
                  border: '1px dashed ' + (alreadyAdded ? 'var(--clr-border)' : 'rgba(99,102,241,0.35)'),
                  borderRadius: 'var(--radius-sm)',
                  color: alreadyAdded ? 'var(--clr-text-muted)' : 'var(--clr-info)',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '0.72rem', fontWeight: 500,
                  padding: '0.2rem 0.55rem',
                  cursor: alreadyAdded ? 'default' : 'pointer',
                  transition: 'all 0.15s',
                  opacity: alreadyAdded ? 0.5 : 1,
                }}
                onMouseEnter={e => { if (!alreadyAdded) e.currentTarget.style.background = 'rgba(99,102,241,0.15)'; }}
                onMouseLeave={e => { if (!alreadyAdded) e.currentTarget.style.background = 'rgba(99,102,241,0.07)'; }}
              >
                {code}
              </button>
            );
          })}
        </div>
        {hasContextNote && (
          <p style={{ fontSize: '0.68rem', color: 'var(--clr-text-muted)', marginTop: '0.35rem' }}>
          </p>
        )}
      </div>
    </div>
  );
}
