import axios from "axios";
import { LEGACY_API_STATUS_MAP } from "../constants/caseStatus.js";

const BASE_URL =
  import.meta.env.VITE_API_URL || "http://161.118.217.29:8000/api/v1";

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: false,
});

/** null = unknown; true = canonical in_review schema; false = legacy pending-only PATCH */
let caseStatusApiV2 = null;

export async function detectCaseStatusApiVersion() {
  if (caseStatusApiV2 !== null) return caseStatusApiV2;
  try {
    const root = BASE_URL.replace(/\/api\/v1\/?$/, "");
    const { data } = await axios.get(`${root}/openapi.json`, { timeout: 8000 });
    const props = data?.components?.schemas?.CaseStatusUpdate?.properties || {};
    caseStatusApiV2 = Boolean(props.feedback || props.rejection_reason);
  } catch {
    caseStatusApiV2 = false;
  }
  console.info("[caseApi] status API v2 (canonical in_review):", caseStatusApiV2);
  return caseStatusApiV2;
}

function listStatusParam(uiStatus) {
  if (!uiStatus) return uiStatus;
  if (uiStatus === "in_review" && caseStatusApiV2 === false) return "pending";
  return uiStatus;
}

function patchStatusBody(data) {
  if (!data?.status || caseStatusApiV2 !== false) return data;
  const legacy = LEGACY_API_STATUS_MAP[data.status];
  if (legacy && legacy !== data.status) return { ...data, status: legacy };
  return data;
}

const getToken = () => localStorage.getItem("access_token");

const token = getToken();
if (token) {
  api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const status = err.response?.status;
    const url    = err.config?.url || '';

    // Log every API error for debugging
    console.error(
      `[API Error] ${err.config?.method?.toUpperCase()} ${url}:`,
      status,
      err.response?.data || err.message
    );

    // On 401: clear stale/expired token and force back to login.
    // Skip for login/demo-login endpoints themselves (prevents redirect loop).
    const isAuthEndpoint = url.includes('/auth/login') || url.includes('/auth/demo-login');
    if (status === 401 && !isAuthEndpoint) {
      console.warn('[API] 401 received — session invalid or expired. Clearing auth state.');

      // Preserve theme, clear everything else
      sessionStorage.clear();
      const theme = localStorage.getItem('theme');
      localStorage.clear();
      if (theme) localStorage.setItem('theme', theme);
      window.dispatchEvent(new CustomEvent('app:session-reset'));

      // Fire event so React UI can show a toast before redirect
      window.dispatchEvent(new CustomEvent('auth:expired'));

      // Small delay so toast renders
      setTimeout(() => { window.location.href = '/login'; }, 600);
    }

    return Promise.reject(err);
  }
);

export const authApi = {
  login: async (data) => {
    const res = await api.post("/auth/login", data);
    const { access_token, user } = res.data;
    localStorage.setItem("access_token", access_token);
    localStorage.setItem("user", JSON.stringify(user));
    localStorage.removeItem("demo_session"); // Production login clears demo session
    api.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;
    return res;
  },
  
  demoLogin: async (role) => {
    console.log("DEBUG: Calling demo-login for role:", role);
    const res = await api.post("/auth/demo-login", { role });
    const { access_token, user } = res.data;
    localStorage.setItem("access_token", access_token);
    localStorage.setItem("user", JSON.stringify(user));
    localStorage.setItem("demo_session", "true");
    api.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;
    return res;
  },

  signup: (data) => api.post("/auth/signup", data),

  me: () => api.get("/auth/me"),

  refresh: () => api.post("/auth/refresh"),

  users: () => api.get("/auth/users"),

  createUser: (data) => api.post("/auth/users", data),

  resetPassword: (id, payload) =>
    api.patch(`/auth/users/${id}/reset-password`, payload),

  updateRole: (id, role) =>
    api.patch(`/auth/users/${id}/role`, { role }),

  deleteUser: (id) => api.delete(`/auth/users/${id}`),
  toggleActive: (id) => api.patch(`/auth/users/${id}/toggle-active`),

  orgs: () => api.get("/auth/org"),

  createOrg: (data) => api.post("/auth/org", data),

  branches: () => api.get("/auth/branches"),

  createBranch: (data) => api.post("/auth/branches", data),
};

export const auditApi = {
  runAudit: (data) => `${BASE_URL}/audit`,
  submitFeedback: (data) => api.post("/feedback", data),
  evaluate: (force = true) => api.get("/evaluation", { params: { force_refresh: force } }),
  evaluationStatus: () => api.get("/evaluation/status"),
  health: () => api.get("/health"),
};

export const caseApi = {
  list: (params) => {
    const p = { ...params };
    if (p.status) p.status = listStatusParam(p.status);
    return api.get("/cases", { params: p });
  },
  get: (id) => api.get(`/cases/${id}`),
  update: (id, data) => api.patch(`/cases/${id}`, data),
  updateStatus: async (id, data) => {
    await detectCaseStatusApiVersion();
    const url = `/cases/${id}/status`;
    const body = patchStatusBody(data);
    console.info("[caseApi.updateStatus]", { method: "PATCH", url, body });
    try {
      return await api.patch(url, body);
    } catch (err) {
      const detail = err.response?.data?.detail;
      const isInvalidStatus =
        err.response?.status === 400 &&
        typeof detail === "string" &&
        detail.toLowerCase().includes("invalid status");
      const legacyStatus = data?.status && LEGACY_API_STATUS_MAP[data.status];
      if (isInvalidStatus && legacyStatus && legacyStatus !== data.status) {
        const legacyBody = { ...data, status: legacyStatus };
        console.warn("[caseApi.updateStatus] Legacy server retry:", legacyBody);
        caseStatusApiV2 = false;
        return await api.patch(url, legacyBody);
      }
      throw err;
    }
  },
  submit: (id) => api.post(`/cases/${id}/submit`),
  approve: (id, confidence) => api.post(`/cases/${id}/approve`, null, { params: { review_confidence: confidence } }),
  reject: (id, justification, confidence) => api.post(`/cases/${id}/reject`, { justification, review_confidence: confidence }),
  updateCodes: (id, final_codes, justification) => api.post(`/cases/${id}/update-codes`, { final_codes, justification }),
  assign: (id, reviewer_id) => api.post(`/cases/${id}/assign`, { reviewer_id }),
  reopen: (id) => api.post(`/cases/${id}/reopen`),
  getAuditTrail: (id) => api.get(`/cases/${id}/audit`),
  delete: (id) => api.delete(`/cases/${id}`),
};

export const analyticsApi = {
  overview: (days = 30, currency = "usd") =>
    api.get("/analytics/overview", { params: { days, currency } }),

  trends: (days = 30, currency = "usd") =>
    api.get("/analytics/trends", { params: { days, currency } }),
};

export default api;