/** Parse FastAPI error detail for user-facing messages. */
export function formatApiError(err, fallback = 'Request failed') {
  const status = err?.response?.status;
  const detail = err?.response?.data?.detail;

  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d?.msg || JSON.stringify(d)).join('; ');
  }
  if (detail && typeof detail === 'object') {
    return detail.message || JSON.stringify(detail);
  }
  if (status === 404) return 'Case not found.';
  if (status === 403) return 'Permission denied.';
  if (status === 401) return 'Session expired. Please log in again.';
  return err?.message || fallback;
}
