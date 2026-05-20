import React, { useEffect, useState, useCallback, useRef } from 'react';
import Sidebar from '../components/Sidebar.jsx';
import TopBar from '../components/TopBar.jsx';
import { authApi } from '../services/api.js';
import { useAuth } from '../main.jsx';
import { useNavigate } from 'react-router-dom';
import '../styles/dashboard.css';

const ROLE_COLORS = { ADMIN: '#10b981', CODER: '#10b981', REVIEWER: '#10b981' };

function calcPwStrength(pw) {
  if (!pw) return { score: 0, label: '', color: 'transparent' };
  let s = 0;
  if (pw.length >= 8)  s++;
  if (pw.length >= 12) s++;
  if (/[A-Z]/.test(pw)) s++;
  if (/[0-9]/.test(pw)) s++;
  if (/[^A-Za-z0-9]/.test(pw)) s++;
  if (s <= 1) return { score: 1, label: 'Weak',   color: '#ef4444' };
  if (s <= 3) return { score: 3, label: 'Fair',   color: '#f59e0b' };
  if (s <= 4) return { score: 4, label: 'Good',   color: '#10b981' };
  return            { score: 5, label: 'Strong',  color: '#6366f1' };
}

function useToast() {
  const [toasts, setToasts] = useState([]);
  const add = useCallback((msg, type = 'success') => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3500);
  }, []);
  return { toasts, add };
}

function ToastContainer({ toasts }) {
  if (!toasts.length) return null;
  return (
    <div className="toast-container" aria-live="polite" aria-label="Notifications">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`} role="status">
          {t.type === 'success' ? '✓' : t.type === 'error' ? '✗' : 'ℹ'} {t.msg}
        </div>
      ))}
    </div>
  );
}

function ConfirmModal({ title, message, onConfirm, onCancel, danger = true }) {
  const confirmBtn = useRef(null);
  useEffect(() => { confirmBtn.current?.focus(); }, []);

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onCancel(); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="confirm-modal-title">
      <div className="modal-box">
        <h3 id="confirm-modal-title">{title}</h3>
        <p>{message}</p>
        <div className="modal-actions">
          <button
            onClick={onCancel}
            style={{ padding: '0.55rem 1.2rem', background: 'var(--clr-surface-2)', border: '1px solid var(--clr-border)', borderRadius: '8px', color: 'var(--clr-text-secondary)', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500 }}
          >
            Cancel
          </button>
          <button
            ref={confirmBtn}
            onClick={onConfirm}
            style={{
              padding: '0.55rem 1.2rem',
              background: danger ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
              border: `1px solid ${danger ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)'}`,
              borderRadius: '8px',
              color: danger ? '#fca5a5' : '#6ee7b7',
              cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600,
            }}
          >
            {danger ? '⚠ Confirm Delete' : 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ResetPasswordModal({ user, onClose, onSuccess, toast }) {
  const [passwords, setPasswords] = useState({ new_password: '', confirm_password: '' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const firstField = useRef(null);

  useEffect(() => { firstField.current?.focus(); }, []);

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose(); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!passwords.new_password || passwords.new_password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (passwords.new_password !== passwords.confirm_password) {
      setError('Passwords do not match.');
      return;
    }
    setError('');
    setBusy(true);
    try {
      await authApi.resetPassword(user.id, { new_password: passwords.new_password });
      toast(`Password reset for ${user.name}`, 'success');
      setPasswords({ new_password: '', confirm_password: '' });
      onSuccess();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to reset password.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="reset-modal-title">
      <div className="modal-box">
        <h3 id="reset-modal-title">Reset Password - {user.name}</h3>
        {error && <div className="error-banner" style={{marginBottom: '1rem', padding: '0.5rem', fontSize: '0.8rem'}}>{error}</div>}
        <form onSubmit={handleSubmit} className="drawer-form">
          <label>
            New Password
            <input type="password" ref={firstField} value={passwords.new_password} onChange={e => setPasswords(p => ({...p, new_password: e.target.value}))} />
          </label>
          <label>
            Confirm Password
            <input type="password" value={passwords.confirm_password} onChange={e => setPasswords(p => ({...p, confirm_password: e.target.value}))} />
          </label>
          <div className="modal-actions" style={{marginTop: '1.5rem'}}>
            <button type="button" onClick={onClose} style={{ padding: '0.55rem 1.2rem', background: 'var(--clr-surface-2)', border: '1px solid var(--clr-border)', borderRadius: '8px', color: 'var(--clr-text-secondary)', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500 }}>Cancel</button>
            <button type="submit" disabled={busy} style={{ padding: '0.55rem 1.2rem', background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.4)', borderRadius: '8px', color: '#a5b4fc', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600 }}>{busy ? 'Working...' : 'Reset Password'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

function AddUserDrawer({ onClose, onSuccess, toast }) {
  const [form, setForm]     = useState({ name: '', email: '', password: '', role: 'CODER' });
  const [errors, setErrors] = useState({});
  const [busy, setBusy]     = useState(false);
  const pwStrength = calcPwStrength(form.password);

  const firstField = useRef(null);
  useEffect(() => { firstField.current?.focus(); }, []);

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose(); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = 'Full name is required.';
    if (!form.email.trim()) e.email = 'Email is required.';
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) e.email = 'Enter a valid email address.';
    if (!form.password) e.password = 'Password is required.';
    else if (form.password.length < 8) e.password = 'Password must be at least 8 characters.';
    else if (pwStrength.score < 2) e.password = 'Password is too weak. Add numbers or symbols.';
    return e;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }
    setBusy(true);
    try {
      await authApi.createUser(form);
      toast('User created successfully.', 'success');
      onSuccess();
      onClose();
    } catch (err) {
      toast(err.response?.data?.detail || 'Failed to create user.', 'error');
    } finally {
      setBusy(false);
    }
  };

  const setField = (field) => (e) => {
    setForm(f => ({ ...f, [field]: e.target.value }));
    if (errors[field]) setErrors(er => ({ ...er, [field]: '' }));
  };

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="add-user-title">
        <div className="drawer-header">
          <h3 id="add-user-title">Add New User</h3>
          <button onClick={onClose} aria-label="Close add user panel">✕</button>
        </div>
        <div className="drawer-body">
          <p style={{ color: 'var(--clr-text-muted)', fontSize: '0.82rem', marginBottom: '0.5rem' }}>
            Create a new platform account. The user will be able to log in immediately.
          </p>
          <form onSubmit={handleSubmit} className="drawer-form" noValidate>
            <label htmlFor="new-name">
              Full Name
              <input
                id="new-name" ref={firstField}
                type="text" placeholder="Dr. Jane Smith"
                value={form.name} onChange={setField('name')}
                className={errors.name ? 'invalid' : ''}
                autoComplete="off"
              />
              {errors.name && <span className="field-error">{errors.name}</span>}
            </label>

            <label htmlFor="new-email">
              Email Address
              <input
                id="new-email" type="email"
                placeholder="doctor@hospital.com"
                value={form.email} onChange={setField('email')}
                className={errors.email ? 'invalid' : ''}
                autoComplete="off"
              />
              {errors.email && <span className="field-error">{errors.email}</span>}
            </label>

            <label htmlFor="new-password">
              Password
              <input
                id="new-password" type="password"
                placeholder="Min 8 characters"
                value={form.password} onChange={setField('password')}
                className={errors.password ? 'invalid' : ''}
                autoComplete="new-password"
              />
              {form.password && (
                <>
                  <div className="pw-strength-bar">
                    <div className="pw-strength-bar-fill" style={{ width: `${(pwStrength.score / 5) * 100}%`, background: pwStrength.color }} />
                  </div>
                  <span className="pw-strength-label" style={{ color: pwStrength.color }}>{pwStrength.label}</span>
                </>
              )}
              {errors.password && <span className="field-error">{errors.password}</span>}
            </label>

            <label htmlFor="new-role">
              Role
              <select id="new-role" value={form.role} onChange={setField('role')}>
                <option value="CODER">Coder</option>
                <option value="REVIEWER">Reviewer</option>
                <option value="ADMIN">Admin</option>
              </select>
            </label>

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
              <button
                type="submit"
                disabled={busy}
                className="new-analysis-btn"
                style={{ flex: 1, justifyContent: 'center' }}
              >
                {busy ? <><span className="spinner" aria-hidden="true" /> Creating…</> : 'Create User'}
              </button>
              <button type="button" className="cancel-btn" onClick={onClose}>Cancel</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

function UserCard({ u, currentUser, onDelete, onReset, onRoleChange, busy }) {
  const isSelf = u.id === currentUser.id;
  const roleColor = ROLE_COLORS[u.role] || '#6366f1';

  return (
    <div className="uc">
      {/* Top: avatar + name + role badge */}
      <div className="uc-top">
        <div
          className="uc-avatar"
          style={{ background: `linear-gradient(135deg, ${roleColor}bb, ${roleColor})` }}
          aria-hidden="true"
        >
          {u.name
            ? u.name.split(' ').map(w => w[0]).filter(Boolean).slice(0, 2).join('').toUpperCase()
            : '?'}
        </div>
        <div className="uc-identity">
          <div className="uc-name">
            {u.name}
            {isSelf && <span className="uc-you">you</span>}
          </div>
          <div className="uc-email" title={u.email}>{u.email}</div>
          <div style={{ fontSize: '0.68rem', fontFamily: 'monospace', color: 'var(--clr-text-muted)', marginTop: '0.15rem', letterSpacing: '0.04em' }}>
            ID: #{u.id}
          </div>
        </div>
        <span className="uc-badge" style={{ background: roleColor }}>{u.role}</span>
      </div>

      {/* Middle: role dropdown */}
      <select
        className="uc-select"
        value={u.role}
        onChange={(e) => onRoleChange(u.id, e.target.value)}
        disabled={busy || isSelf}
        aria-label="Change role"
      >
        <option value="CODER">Coder</option>
        <option value="REVIEWER">Reviewer</option>
        <option value="ADMIN">Admin</option>
      </select>

      {/* Bottom: actions + joined */}
      <div className="uc-footer">
        <div className="uc-actions">
          <button className="uc-btn-ghost" onClick={() => onReset(u)} disabled={busy}>
            Reset PW
          </button>
          {isSelf ? (
            <span className="uc-self">Current User</span>
          ) : (
            <button className="uc-btn-danger" onClick={() => onDelete(u)} disabled={busy}>
              Delete
            </button>
          )}
        </div>
        <div className="uc-joined">Joined {new Date(u.created_at).toLocaleDateString()}</div>
      </div>
    </div>
  );
}

export default function UsersPage() {
  const { user: currentUser, isAdmin } = useAuth();
  const navigate   = useNavigate();
  const { toasts, add: toast } = useToast();

  const [users,      setUsers]      = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState('');
  const [showDrawer, setShowDrawer] = useState(false);
  const [busyIds,    setBusyIds]    = useState(new Set());
  const [confirm,    setConfirm]    = useState(null); 
  const [resetUser,  setResetUser]  = useState(null);

  useEffect(() => {
    if (!isAdmin) { navigate('/'); return; }
    fetchUsers();
  }, []);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await authApi.users();
      const list = Array.isArray(res.data) ? res.data : [];
      console.log('Users API response:', list);
      setUsers(list);
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load users.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDelete = useCallback((u) => {
    setConfirm({
      user: u,
      title: 'Delete User',
      message: `Permanently delete "${u.name}" (${u.email})? This cannot be undone.`,
    });
  }, []);

  const handleRoleChange = useCallback(async (userId, newRole) => {
    setBusyIds(s => new Set(s).add(userId));
    try {
      await authApi.updateRole(userId, newRole);
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u));
      toast('User role updated.', 'success');
    } catch (e) {
      toast(e.response?.data?.detail || 'Role update failed.', 'error');
    } finally {
      setBusyIds(s => { const n = new Set(s); n.delete(userId); return n; });
    }
  }, [toast]);

  const confirmDelete = useCallback(async () => {
    if (!confirm) return;
    const target = confirm.user;
    setConfirm(null);
    setBusyIds(s => new Set(s).add(target.id));
    try {
      await authApi.deleteUser(target.id);
      toast(`${target.name} has been deleted.`, 'success');
      // Always re-fetch from server — no local filtering
      await fetchUsers();
    } catch (e) {
      console.error('USER DELETE ERROR:', e.response?.data || e.message);
      toast(e.response?.data?.detail || 'Delete failed.', 'error');
    } finally {
      setBusyIds(s => { const n = new Set(s); n.delete(target.id); return n; });
    }
  }, [confirm, toast, fetchUsers]);

  const coderUsers    = users.filter(u => u.role === 'CODER');
  const reviewerUsers = users.filter(u => u.role === 'REVIEWER');
  const adminUsers    = users.filter(u => u.role === 'ADMIN');


  const headerActions = (
    <button className="new-analysis-btn" onClick={() => setShowDrawer(true)} aria-label="Add new user">
      + Add User
    </button>
  );

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <main className="dashboard-main" id="main-content">
        <TopBar
          pageTitle="User Management"
          pageSubtitle={!loading && `Total users: ${users.length}`}
          actions={headerActions}
        />

        <div className="dashboard-content">
          <div style={{ maxWidth: '1200px', margin: '0 auto' }}> {/* STEP 14: EMPTY SPACE FIX */}
            {error && (
              <div className="error-banner" role="alert">
                <span>⚠</span> {error}
                <button className="error-banner-retry" onClick={fetchUsers}>Retry</button>
              </div>
            )}

            {loading ? (
              <div className="loading-center" role="status">
                <div className="big-spinner" aria-hidden="true" /> Loading users…
              </div>
            ) : (
              <>
                <div className="users-section-title">Coders ({coderUsers.length})</div>
                <div className="users-grid">
                  {coderUsers.length === 0 ? (
                    <div style={{ color: 'var(--clr-text-muted)', fontSize: '0.82rem', padding: '1rem 0' }}>No coders found.</div>
                  ) : coderUsers.map(u => (
                    <UserCard key={u.id} u={u} currentUser={currentUser} onDelete={handleDelete} onReset={setResetUser} onRoleChange={handleRoleChange} busy={busyIds.has(u.id)} />
                  ))}
                </div>

                <div className="users-section-title">Reviewers ({reviewerUsers.length})</div>
                <div className="users-grid">
                  {reviewerUsers.length === 0 ? (
                    <div style={{ color: 'var(--clr-text-muted)', fontSize: '0.82rem', padding: '1rem 0' }}>No reviewers found.</div>
                  ) : reviewerUsers.map(u => (
                    <UserCard key={u.id} u={u} currentUser={currentUser} onDelete={handleDelete} onReset={setResetUser} onRoleChange={handleRoleChange} busy={busyIds.has(u.id)} />
                  ))}
                </div>

                <div className="users-section-title">Admins ({adminUsers.length})</div>
                <div className="users-grid">
                  {adminUsers.length === 0 ? (
                    <div style={{ color: 'var(--clr-text-muted)', fontSize: '0.82rem', padding: '1rem 0' }}>No admins found.</div>
                  ) : adminUsers.map(u => (
                    <UserCard key={u.id} u={u} currentUser={currentUser} onDelete={handleDelete} onReset={setResetUser} onRoleChange={handleRoleChange} busy={busyIds.has(u.id)} />
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </main>

      {showDrawer && (
        <AddUserDrawer
          onClose={() => setShowDrawer(false)}
          onSuccess={fetchUsers}
          toast={toast}
        />
      )}

      {confirm && (
        <ConfirmModal
          title={confirm.title}
          message={confirm.message}
          onConfirm={confirmDelete}
          onCancel={() => setConfirm(null)}
          danger
        />
      )}

      {resetUser && (
        <ResetPasswordModal
          user={resetUser}
          onClose={() => setResetUser(null)}
          onSuccess={() => setResetUser(null)}
          toast={toast}
        />
      )}
      <ToastContainer toasts={toasts} />
    </div>
  );
}