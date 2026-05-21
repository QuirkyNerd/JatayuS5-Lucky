/** Canonical case status values — must match backend API. */
export const CASE_STATUS = {
  DRAFT: 'draft',
  SUBMITTED: 'submitted',
  IN_REVIEW: 'in_review',
  APPROVED: 'approved',
  REJECTED: 'rejected',
};

const ALIASES = {
  under_review: CASE_STATUS.IN_REVIEW,
  /** Legacy production API stores in-queue cases as "pending". */
  pending: CASE_STATUS.IN_REVIEW,
  review: CASE_STATUS.IN_REVIEW,
};

/** Map canonical UI status → legacy production API (pre-deploy). Remove after backend redeploy. */
export const LEGACY_API_STATUS_MAP = {
  [CASE_STATUS.DRAFT]: 'pending',
  [CASE_STATUS.SUBMITTED]: 'pending',
  [CASE_STATUS.IN_REVIEW]: 'pending',
  [CASE_STATUS.APPROVED]: 'approved',
  [CASE_STATUS.REJECTED]: 'rejected',
};

/** Map legacy/DB values to canonical enum for UI + API. */
export function normalizeCaseStatus(status) {
  if (!status) return status;
  const key = String(status).toLowerCase().replace(/-/g, '_').replace(/ /g, '_');
  return ALIASES[key] || key;
}

export const STATUS_LABELS = {
  [CASE_STATUS.DRAFT]: 'Draft',
  [CASE_STATUS.SUBMITTED]: 'Submitted',
  [CASE_STATUS.IN_REVIEW]: 'In Review',
  [CASE_STATUS.APPROVED]: 'Approved',
  [CASE_STATUS.REJECTED]: 'Rejected',
};

export const STATUS_COLORS = {
  [CASE_STATUS.DRAFT]: '#94a3b8',
  [CASE_STATUS.SUBMITTED]: '#6366f1',
  [CASE_STATUS.IN_REVIEW]: '#8b5cf6',
  [CASE_STATUS.APPROVED]: '#10b981',
  [CASE_STATUS.REJECTED]: '#ef4444',
};

/** Single source of truth for dropdown — value is always lowercase API enum. */
export const STATUS_OPTIONS = [
  { label: 'Draft', value: CASE_STATUS.DRAFT },
  { label: 'Submitted', value: CASE_STATUS.SUBMITTED },
  { label: 'In Review', value: CASE_STATUS.IN_REVIEW },
  { label: 'Approved', value: CASE_STATUS.APPROVED },
  { label: 'Rejected', value: CASE_STATUS.REJECTED },
];

/** Admin/reviewer dropdown options (canonical values only). */
export const REVIEW_STATUS_OPTIONS = STATUS_OPTIONS;

export const FILTER_STATUS_OPTIONS = [
  { label: 'All', value: '' },
  { label: 'Draft', value: CASE_STATUS.DRAFT },
  { label: 'Submitted', value: CASE_STATUS.SUBMITTED },
  { label: 'In Review', value: CASE_STATUS.IN_REVIEW },
  { label: 'Approved', value: CASE_STATUS.APPROVED },
  { label: 'Rejected', value: CASE_STATUS.REJECTED },
];
