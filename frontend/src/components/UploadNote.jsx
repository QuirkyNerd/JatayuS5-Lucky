import React, { useRef, useCallback, useEffect } from 'react';
import { SAMPLE_NOTES, getNextSampleIndex } from '../data/sampleNotes.js';

export default function UploadNote({
  value = '',
  onChange = () => {},
  file = null,
  onFileSelected = () => {},
  disabled = false,
}) {
  const charCount   = value?.length || 0;
  const fileInputRef = useRef(null);
  const lastSampleIdx = useRef(-1);



  const handleFileChange = useCallback((e) => {
    if (e.target.files?.[0]) {
      onFileSelected(e.target.files[0]);
      onChange('');
    }
  }, [onFileSelected, onChange]);

  const clearFile = useCallback(() => {
    onFileSelected(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [onFileSelected]);

  const handleLoadSample = useCallback(() => {
    clearFile();
    const idx = getNextSampleIndex(lastSampleIdx.current);
    lastSampleIdx.current = idx;
    onChange(SAMPLE_NOTES[idx]);
  }, [clearFile, onChange]);

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '1.1rem' }} aria-hidden="true"></span>
          <h2 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--clr-text-primary)', margin: 0 }}>
            Clinical Note
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".txt,.pdf,.docx,.png,.jpg,.jpeg"
            style={{ display: 'none' }}
            aria-label="Upload clinical note file"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            style={{
              background: 'rgba(16,185,129,0.1)',
              border: '1px solid rgba(16,185,129,0.3)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--clr-success)',
              padding: '0.3rem 0.8rem',
              fontSize: '0.78rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: 'inherit',
            }}
            onMouseEnter={e => e.target.style.background = 'rgba(16,185,129,0.2)'}
            onMouseLeave={e => e.target.style.background = 'rgba(16,185,129,0.1)'}
            disabled={disabled}
          >
            ↑ Upload File
          </button>
          <button
            type="button"
            onClick={handleLoadSample}
            title="Load a different sample clinical note"
            style={{
              background: 'rgba(59,130,246,0.1)',
              border: '1px solid rgba(59,130,246,0.3)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--clr-primary)',
              padding: '0.3rem 0.8rem',
              fontSize: '0.78rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: 'inherit',
            }}
            onMouseEnter={e => e.target.style.background = 'rgba(59,130,246,0.2)'}
            onMouseLeave={e => e.target.style.background = 'rgba(59,130,246,0.1)'}
            disabled={disabled}
          >
            ↻ Load Sample
          </button>
        </div>
      </div>

      {file ? (
        <div style={{
          padding: '0.8rem', background: 'rgba(16,185,129,0.05)',
          border: '1px dashed rgba(16,185,129,0.3)', borderRadius: 'var(--radius-md)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--clr-text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span></span>
            <strong>{file.name}</strong>
            <span style={{ fontSize: '0.7rem', color: 'var(--clr-text-muted)' }}>({Math.round(file.size / 1024)} KB)</span>
          </div>
          <button
            onClick={clearFile}
            style={{ background: 'none', border: 'none', color: 'var(--clr-danger)', cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'inherit' }}
            disabled={disabled}
            aria-label="Remove uploaded file"
          >
            ✕ Remove
          </button>
        </div>
      ) : (
        <>
          <p style={{ fontSize: '0.78rem', color: 'var(--clr-text-muted)', margin: 0 }}>
            Paste a clinical note, discharge summary, or upload a PDF/DOCX/Image.
          </p>
          <textarea
            id="clinical-note-input"
            value={value}
            onChange={e => onChange(e.target.value)}
            disabled={disabled}
            placeholder="Paste clinical documentation here…"
            rows={15}
            aria-label="Clinical note input"
            style={{
              width: '100%',
              background: 'var(--clr-surface-2)',
              border: '1px solid var(--clr-border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--clr-text-primary)',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: '0.8rem',
              lineHeight: 1.75,
              padding: '1rem',
              resize: 'vertical',
              outline: 'none',
              transition: 'border-color 0.2s',
              minHeight: '280px',
            }}
            onFocus={e => e.target.style.borderColor = 'var(--clr-primary)'}
            onBlur={e => e.target.style.borderColor = 'var(--clr-border)'}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', fontSize: '0.72rem', color: 'var(--clr-text-muted)' }}>
            {charCount.toLocaleString()} characters
          </div>
        </>
      )}
    </div>
  );
}
