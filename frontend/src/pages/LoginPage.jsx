import React, { useState, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth, ROLE_HOME } from "../main.jsx";
import { authApi } from "../services/api.js";
import { Eye, EyeOff } from "lucide-react";
import "../styles/auth.css";

export default function LoginPage() {
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const { login, logout, user, token, isDemoEnabled } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (token) {
      navigate(ROLE_HOME[user?.role] || '/', { replace: true });
    }
  }, [token, navigate]);

  const handleChange = useCallback(
    (e) => {
      setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
      if (error) setError("");
    },
    [error]
  );

  const performLogin = async (emailToUse, passwordToUse) => {
    if (!emailToUse || !passwordToUse) {
      setError("Please enter your email and password.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const res = await authApi.login({
        email: emailToUse,
        password: passwordToUse,
      });

      console.log("LOGIN RESPONSE:", res.data);

      if (res.data && res.data.access_token) {
        const { user, access_token } = res.data;

        localStorage.setItem("access_token", access_token);
        localStorage.setItem("user", JSON.stringify(user));

        login(user, access_token);

        navigate(ROLE_HOME[user.role] || "/", { replace: true });
      } else {
        setError("Invalid response from server");
      }
    } catch (err) {
      console.error("LOGIN ERROR:", err);

      const msg =
        err.response?.data?.detail ||
        "Authentication failed. Please check your credentials.";

      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await performLogin(form.email, form.password);
  };

  const handleDemoLogin = async () => {
    setLoading(true);
    setError("");
    try {
      console.log("DEBUG: Initiating demo login sequence...");
      const res = await authApi.demoLogin("coder");
      if (res.data && res.data.access_token) {
        const { user, access_token } = res.data;
        login(user, access_token);
        navigate(ROLE_HOME[user.role] || "/", { replace: true });
      }
    } catch (err) {
      console.error("DEMO LOGIN FAILURE:", err.response || err);
      const msg = err.response?.data?.detail || err.message || "Demo access is currently unavailable.";
      setError(`Demo Error: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-bg">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">CodePerfect Audit</h1>
          <p className="auth-tagline">Clinical Coding Auditor</p>
          <p className="auth-subtitle">Sign in to your account</p>
        </div>

        {error && (
          <div className="auth-error" role="alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <div className="form-group">
            <label htmlFor="auth-email">Email Address</label>
            <input
              id="auth-email"
              name="email"
              type="email"
              value={form.email}
              onChange={handleChange}
              autoComplete="email"
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="auth-password">Password</label>

            <div className="password-input-wrapper">
              <input
                id="auth-password"
                name="password"
                type={showPassword ? "text" : "password"}
                value={form.password}
                onChange={handleChange}
                autoComplete="current-password"
                disabled={loading}
                required
              />

              <button
                type="button"
                className="pw-toggle-btn"
                onClick={() => setShowPassword((v) => !v)}
              >
                {showPassword ? (
                  <EyeOff size={18} />
                ) : (
                  <Eye size={18} />
                )}
              </button>
            </div>
          </div>

          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" />
                Authenticating...
              </>
            ) : (
              "Sign In"
            )}
          </button>

          <div className="auth-divider">
            <span>or continue with demo access</span>
          </div>

          <button
            type="button"
            className="auth-submit demo-btn"
            onClick={handleDemoLogin}
            disabled={loading}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
          >
            {loading && <span className="spinner" style={{ width: 14, height: 14 }} />}
            Use Demo Access
          </button>
        </form>

        <p className="auth-note">
          Demo credentials are provided in the project README.
          <br />
          Access is provided by your system administrator.
        </p>
      </div>
    </div>
  );
}