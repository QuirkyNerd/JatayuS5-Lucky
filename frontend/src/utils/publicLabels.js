/**
 * Defensive mapping for code source/strategy labels shown in the UI.
 * Internal pipeline tags must never appear to end users.
 */

const ALLOWED_PUBLIC_SOURCES = new Set(['rag', 'deterministic', 'hybrid', 'validated']);

const INTERNAL_MARKERS = [
  'presentation_demo_anchor',
  'urology_demo_pathway',
  'human_seed',
  'human_entry',
  'rule_injection',
  'forensic',
  'debug',
  'anchor',
  'internal',
  'experimental',
  'fallback',
  'protected',
];

/**
 * @param {string|null|undefined} raw
 * @returns {'rag'|'deterministic'|'hybrid'|'validated'}
 */
export function sanitizePublicSource(raw) {
  if (!raw || typeof raw !== 'string') {
    return 'deterministic';
  }

  const s = raw.trim().toLowerCase();
  if (ALLOWED_PUBLIC_SOURCES.has(s)) {
    return s;
  }

  if (s === 'rag' || s.startsWith('rag')) {
    return 'rag';
  }

  if (s === 'llm' || s === 'hybrid') {
    return 'hybrid';
  }

  if (s === 'validated' || s === 'validation') {
    return 'validated';
  }

  if (
    s === 'deterministic' ||
    INTERNAL_MARKERS.some((marker) => s.includes(marker))
  ) {
    return 'deterministic';
  }

  return 'deterministic';
}

/**
 * @param {object} code
 * @returns {object}
 */
export function sanitizeCodeForDisplay(code) {
  if (!code || typeof code !== 'object') {
    return code;
  }
  const out = { ...code };
  if (out.source != null) {
    out.source = sanitizePublicSource(out.source);
  }
  if (out.strategy != null) {
    out.strategy = sanitizePublicSource(out.strategy);
  }
  if (out.source_type != null) {
    out.source_type = sanitizePublicSource(out.source_type);
  }
  return out;
}

/**
 * @param {Array<object>|null|undefined} codes
 * @returns {Array<object>}
 */
export function sanitizeCodesForDisplay(codes) {
  if (!Array.isArray(codes)) {
    return [];
  }
  return codes.map(sanitizeCodeForDisplay);
}
