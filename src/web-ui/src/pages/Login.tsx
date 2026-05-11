// src/pages/Login.tsx
export const Login = () => {
  const params = new URLSearchParams(window.location.search);
  const error = params.get("error");

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">GreenPulse</div>
        <div className="login-tagline">Real-time market intelligence</div>
        {error === "no_account" && (
          <div style={{ color: "var(--red)", fontFamily: "var(--mono)", fontSize: "11px", textAlign: "center", lineHeight: 1.5 }}>
            No account found.<br />Please create an account first.
          </div>
        )}
        {error === "account_exists" && (
          <div style={{ color: "var(--red)", fontFamily: "var(--mono)", fontSize: "11px", textAlign: "center", lineHeight: 1.5 }}>
            Account already exists.<br />Please sign in instead.
          </div>
        )}
        <hr className="login-divider" />
        <a className="login-btn login-btn-primary" href="http://localhost:8080/signin">
          Sign in with Google
        </a>
        <a className="login-btn login-btn-secondary" href="http://localhost:8080/signup">
          Create account
        </a>
      </div>
    </div>
  );
};
