import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../main.jsx';
import { useTheme } from '../main.jsx';
import { clearAppState } from '../utils/demoUtils.js';
import { Sun, Moon, ChevronDown, ChevronUp, LogOut, RefreshCw } from 'lucide-react';
import { authApi } from '../services/api.js';
import { ROLE_HOME } from '../main.jsx';

const ROLE_LABELS = { ADMIN: 'Admin', CODER: 'Coder', REVIEWER: 'Reviewer' };

const AVATAR_COLORS = {
  ADMIN:    'linear-gradient(135deg, #10b981, #059669)',
  CODER:    'linear-gradient(135deg, #10b981, #059669)',
  REVIEWER: 'linear-gradient(135deg, #10b981, #059669)',
};

export function FullPageLoader({ message = "Loading..." }) {
  return (
    <div className="full-screen-loader">
      <div className="spinner" />
      <p>{message}</p>
    </div>
  );
}

export default function TopBar({ pageTitle, pageSubtitle, actions }) {
  const { user, logout } = useAuth();
  const { theme, cycleTheme } = useTheme();
  const navigate = useNavigate();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isSwitching, setIsSwitching] = useState(false);
  const dropdownRef = useRef(null);

  const handleLogout = useCallback(() => {
    logout();
    navigate('/login');
  }, [logout, navigate]);

  const isDemoSession = localStorage.getItem('demo_session') === 'true';

  const switchRole = async (targetRole, retryCount = 0) => {
    if (isSwitching && retryCount === 0) return;
    setIsSwitching(true);
    try {
      // Remove ONLY auth keys — never touch 'theme' or 'demo_session'
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      sessionStorage.clear();

      // 2. Login as new role
      const res = await authApi.demoLogin(targetRole.toLowerCase());
      if (res.data && res.data.access_token) {
        const { user, access_token } = res.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('user', JSON.stringify(user));
        localStorage.setItem('demo_session', 'true');

        // 3. Force full reload
        window.location.href = ROLE_HOME[user.role] || '/';
      }
    } catch (e) {
      if (retryCount < 1) {
        console.warn('Switch failed, retrying...', e);
        return switchRole(targetRole, retryCount + 1);
      }
      console.error('Role switch failed', e);
      alert('Demo action failed. Please retry.');
      setIsSwitching(false);
    }
  };

  const resetDemo = () => {
    clearAppState();
    logout();
    window.location.href = '/login';
  };

  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    }
    if (dropdownOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);

  useEffect(() => {
    function handleKey(e) {
      if (e.key === 'Escape') setDropdownOpen(false);
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, []);

  const initials = (user?.name || 'U')
    .split(' ')
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  const role = user?.role || 'CODER';

  const ThemeIcon = theme === 'dark' ? Moon : Sun;

  return (
    <>
      {isSwitching && <FullPageLoader message="Switching role..." />}
      <div className="topbar" role="banner">
      <div className="topbar-left">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {pageTitle && <h1 className="main-title topbar-title">{pageTitle}</h1>}
          {isDemoSession && (
            <span className="demo-badge">DEMO MODE</span>
          )}
        </div>
        {pageSubtitle && <p className="topbar-subtitle">{pageSubtitle}</p>}
      </div>

      {/* SECTION 9 & 11: Top-Center Role Switch Bar */}
      {isDemoSession && (
        <div className="demo-role-selector">
          {['CODER', 'REVIEWER', 'ADMIN'].map(r => (
            <button
              key={r}
              data-role={r}
              className={`demo-role-btn ${role === r ? 'active' : ''}`}
              onClick={() => switchRole(r)}
              disabled={role === r || isSwitching}
            >
              {ROLE_LABELS[r]}
            </button>
          ))}
        </div>
      )}

      <div className="topbar-right">
        {actions && <div className="topbar-actions">{actions}</div>}

        <button
          className="theme-toggle"
          onClick={cycleTheme}
          aria-label="Toggle theme"
        >
          <ThemeIcon size={18} strokeWidth={1.8} />
        </button>

        <div className="user-dropdown-wrapper" ref={dropdownRef}>
          <button
            className="user-chip"
            onClick={() => setDropdownOpen(o => !o)}
            aria-haspopup="true"
            aria-expanded={dropdownOpen}
          >
            <div
              className="user-chip-avatar"
              style={{ background: AVATAR_COLORS[role] || AVATAR_COLORS.CODER }}
            >
              {initials}
            </div>

            <div className="user-chip-info">
              <span className="user-chip-name">{user?.name || 'User'}</span>
              <span className="role-badge">
                {ROLE_LABELS[role]}
              </span>
            </div>

            {dropdownOpen ? (
              <ChevronUp size={16} />
            ) : (
              <ChevronDown size={16} />
            )}
          </button>

          {dropdownOpen && (
            <div className="user-dropdown" role="menu">
              <div className="user-dropdown-header">
                <div className="user-chip-avatar" style={{ background: AVATAR_COLORS[role] || AVATAR_COLORS.CODER, width: 40, height: 40, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem', fontWeight: 700, color: '#fff', marginBottom: '0.5rem' }}>
                  {initials}
                </div>
                <div className="user-dropdown-name">{user?.name}</div>
                <div className="user-dropdown-email">{user?.email}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--clr-text-muted)', marginTop: '0.2rem', fontFamily: 'monospace', letterSpacing: '0.03em' }}>
                  {ROLE_LABELS[role]} &nbsp;·&nbsp; ID: #{user?.id ?? '—'}
                </div>
              </div>

              <div className="user-dropdown-divider" />

              <button
                className="user-dropdown-item"
                role="menuitem"
                onClick={() => {
                  setDropdownOpen(false);
                  handleLogout();
                }}
              >
                Sign Out
              </button>

              {isDemoSession && (
                <>
                  <div className="user-dropdown-divider" />
                  <button
                    className="user-dropdown-item"
                    style={{ color: 'var(--clr-danger)' }}
                    onClick={resetDemo}
                    disabled={isSwitching}
                  >
                    <RefreshCw size={16} />
                    Reset Demo Environment
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
    </>
  );
}