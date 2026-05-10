// src/pages/Login.tsx
export const Login = () => (
  <div className="login-page">
    <div className="login-card">
      <div className="login-logo">GreenPulse</div>
      <div className="login-tagline">Real-time market intelligence</div>
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
