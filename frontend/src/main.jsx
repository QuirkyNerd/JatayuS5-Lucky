import React, { createContext, useContext, useState, useEffect, useCallback, Component } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage.jsx';
import Dashboard from './pages/Dashboard.jsx';
import CaseHistoryPage from './pages/CaseHistoryPage.jsx';
import AnalyticsPage from './pages/AnalyticsPage.jsx';
import UsersPage from './pages/UsersPage.jsx';
import Evaluation from './pages/Evaluation.jsx';
import './styles/index.css';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary] Uncaught render error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', minHeight: '100vh',
          background: 'var(--clr-bg)', color: 'var(--clr-text-primary)',
          fontFamily: 'inherit', gap: '1rem', padding: '2rem',
        }}>
          <div style={{ fontSize: '2.5rem' }}>⚕️</div>
          <h2 style={{ margin: 0, color: 'var(--clr-danger)' }}>Something went wrong</h2>
          <p style={{ color: 'var(--clr-text-muted)', margin: 0, maxWidth: '420px', textAlign: 'center', lineHeight: 1.7 }}>
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          {/* Section 8: Reload failsafe */}
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '0.5rem', padding: '0.65rem 1.8rem',
              background: 'var(--clr-primary)', border: 'none', borderRadius: '10px',
              color: '#fff', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 600,
              fontFamily: 'inherit',
            }}
          >
            Reload Page
          </button>
          <button
            onClick={() => { window.location.href = '/login'; }}
            style={{
              padding: '0.65rem 1.8rem', background: 'transparent',
              border: '1px solid var(--clr-border)', borderRadius: '10px',
              color: 'var(--clr-text-secondary)', cursor: 'pointer',
              fontSize: '0.9rem', fontFamily: 'inherit',
            }}
          >
            Return to Login
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export const ThemeContext = createContext(null);
export function useTheme() { return useContext(ThemeContext); }

function applyTheme(theme) {
  const root = document.documentElement;
  root.setAttribute('data-theme', theme);
}

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem('theme');
    return (stored === 'light' || stored === 'dark') ? stored : 'dark';
  });

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const cycleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, cycleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const AuthContext = createContext(null);
export function useAuth() { return useContext(AuthContext); }

function AuthProvider({ children }) {
  const [user,         setUser]         = useState(null);     // start null — always
  const [token,        setToken]        = useState(null);     // start null — always
  const [authChecked,  setAuthChecked]  = useState(false);   // true once /auth/me resolves
  const [sessionToast, setSessionToast] = useState(false);   // show 'session expired' banner

  // ── On mount: validate stored token against the backend ──────────────────
  useEffect(() => {
    async function validateSession() {
      const storedToken = localStorage.getItem('access_token');
      const storedUser  = (() => {
        try { return JSON.parse(localStorage.getItem('user')); } catch { return null; }
      })();

      if (!storedToken) {
        // No token at all — not authenticated
        setAuthChecked(true);
        return;
      }

      try {
        // Validate against backend — catches expired / Docker-restarted tokens
        const { default: api } = await import('./services/api.js');
        const res = await api.get('/auth/me', {
          headers: { Authorization: `Bearer ${storedToken}` },
        });
        // Token valid — hydrate state
        const freshUser = res.data;
        setToken(storedToken);
        setUser(freshUser);
        localStorage.setItem('user', JSON.stringify(freshUser));
      } catch {
        // Token invalid / expired — wipe everything except theme
        console.warn('[Auth] Stored token rejected by backend. Clearing session.');
        const theme = localStorage.getItem('theme');
        localStorage.clear();
        if (theme) localStorage.setItem('theme', theme);
      } finally {
        setAuthChecked(true);
      }
    }
    validateSession();
  }, []);

  // ── Listen for 401 events fired by the axios interceptor ─────────────────
  useEffect(() => {
    const handleExpired = () => {
      setSessionToast(true);
      setUser(null);
      setToken(null);
    };
    window.addEventListener('auth:expired', handleExpired);
    return () => window.removeEventListener('auth:expired', handleExpired);
  }, []);

  const login = useCallback((userData, newToken) => {
    localStorage.setItem('access_token', newToken);
    localStorage.setItem('user', JSON.stringify(userData));
    setUser(userData);
    setToken(newToken);
    setSessionToast(false);
  }, []);

  const logout = useCallback(() => {
    const theme = localStorage.getItem('theme');  // preserve theme
    localStorage.clear();
    if (theme) localStorage.setItem('theme', theme);
    setUser(null);
    setToken(null);
  }, []);

  const [isDemoEnabled, setIsDemoEnabled] = useState(true);

  const value = {
    user,
    token,
    login,
    logout,
    authChecked,
    isAdmin:    user?.role === 'ADMIN',
    isCoder:    user?.role === 'CODER',
    isReviewer: user?.role === 'REVIEWER',
    isDemoEnabled,
  };

  return (
    <AuthContext.Provider value={value}>
      {/* Session expired toast — shown before the hard redirect fires */}
      {sessionToast && (
        <div style={{
          position: 'fixed', top: '1rem', left: '50%', transform: 'translateX(-50%)',
          zIndex: 99999, background: '#ef4444', color: '#fff',
          padding: '0.75rem 1.5rem', borderRadius: '10px',
          fontFamily: 'inherit', fontSize: '0.9rem', fontWeight: 600,
          boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
        }}>
          ⚠️ Session expired — redirecting to login…
        </div>
      )}
      {children}
    </AuthContext.Provider>
  );
}

export const AuditContext = createContext(null);
export function useAudit() { return useContext(AuditContext); }

function AuditProvider({ children }) {
  const [noteText,      setNoteText]      = useState('');
  const [humanCodes,    setHumanCodes]    = useState([]);
  const [auditResult,   setAuditResult]   = useState(null);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [isRunning,     setIsRunning]     = useState(false);
  const [auditError,    setAuditError]    = useState('');
  const [file,          setFile]          = useState(null);

  const resetAudit = useCallback(() => {
    setAuditResult(null);
    setPipelineSteps([]);
    setAuditError('');
    setNoteText('');
    setHumanCodes([]);
    setFile(null);
  }, []);

  const value = {
    noteText, setNoteText,
    humanCodes, setHumanCodes,
    auditResult, setAuditResult,
    pipelineSteps, setPipelineSteps,
    isRunning, setIsRunning,
    auditError, setAuditError,
    file, setFile,
    resetAudit,
  };

  return (
    <AuditContext.Provider value={value}>
      {children}
    </AuditContext.Provider>
  );
}

export const ROLE_HOME = {
  ADMIN:    '/case-history',
  CODER:    '/',
  REVIEWER: '/case-history',
};

// ProtectedRoute: blocks render until backend auth check is complete.
// If no valid token → /login. Never shows any UI in an unauthenticated state.
function ProtectedRoute({ children }) {
  const { token, authChecked } = useAuth();

  // Still validating stored token against backend — render nothing (no flash)
  if (!authChecked) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '100vh', background: 'var(--clr-bg)', color: 'var(--clr-text-muted)',
        fontFamily: 'inherit', gap: '0.75rem', fontSize: '0.9rem',
      }}>
        <div style={{ width: 20, height: 20, border: '2px solid var(--clr-primary)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
        Checking session…
      </div>
    );
  }

  if (!token) return <Navigate to="/login" replace />;
  return children;
}

// RoleRoute: only reached by authenticated users (ProtectedRoute runs first).
// Unauthorized role → /unauthorized (they are logged in, just wrong role).
// No token → /login (safety net).
function RoleRoute({ allowed, children }) {
  const { user, token } = useAuth();
  if (!token || !user) return <Navigate to="/login" replace />;
  if (!allowed.includes(user.role)) return <Navigate to="/unauthorized" replace />;
  return children;
}

function UnauthorizedPage() {
  const { user } = useAuth();
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: '100vh',
      background: 'var(--clr-bg)', color: 'var(--clr-text-primary)',
      fontFamily: 'inherit', gap: '1rem',
    }}>
      <div style={{ fontSize: '3rem' }}>🔒</div>
      <h2 style={{ margin: 0 }}>Access Denied</h2>
      <p style={{ color: 'var(--clr-text-muted)', margin: 0 }}>
        You don't have permission to view this page.
      </p>
      <a
        href={ROLE_HOME[user?.role] || '/login'}
        style={{
          marginTop: '0.5rem', padding: '0.65rem 1.8rem',
          background: 'var(--clr-primary)', borderRadius: '10px',
          color: '#fff', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 600,
        }}
      >
        Go to My Home
      </a>
    </div>
  );
}

function AppInner() {
  const navigate = useNavigate();
  return (
    <ErrorBoundary navigate={navigate}>
      <ThemeProvider>
        <AuthProvider>
          <AuditProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/unauthorized" element={<UnauthorizedPage />} />

              {/* Coder home: dashboard */}
              <Route path="/" element={
                <ProtectedRoute>
                  <RoleRoute allowed={['CODER']}>
                    <Dashboard />
                  </RoleRoute>
                </ProtectedRoute>
              } />

              {/* Case History — all roles */}
              <Route path="/case-history" element={
                <ProtectedRoute>
                  <RoleRoute allowed={['ADMIN', 'CODER', 'REVIEWER']}>
                    <CaseHistoryPage />
                  </RoleRoute>
                </ProtectedRoute>
              } />

              {/* Legacy /cases alias */}
              <Route path="/cases" element={<Navigate to="/case-history" replace />} />

              <Route path="/analytics" element={
                <ProtectedRoute>
                  <RoleRoute allowed={['ADMIN', 'REVIEWER']}>
                    <AnalyticsPage />
                  </RoleRoute>
                </ProtectedRoute>
              } />

              <Route path="/users" element={
                <ProtectedRoute>
                  <RoleRoute allowed={['ADMIN']}>
                    <UsersPage />
                  </RoleRoute>
                </ProtectedRoute>
              } />

              <Route path="/evaluation" element={
                <ProtectedRoute>
                  <RoleRoute allowed={['ADMIN']}>
                    <Evaluation />
                  </RoleRoute>
                </ProtectedRoute>
              } />

              {/* Catch-all: send to login, never loop back to / */}
              <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
          </AuditProvider>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
