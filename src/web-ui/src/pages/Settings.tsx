// src/pages/Settings.tsx
import type { User } from "../types";

type ThemeId = "green" | "terminal" | "blue";

const THEMES: { id: ThemeId; label: string; desc: string; accent: string; bg: string }[] = [
  { id: "green",    label: "Green",             desc: "Default dark green",  accent: "#00d46a", bg: "#0d1210" },
  { id: "terminal", label: "Dark Terminal",     desc: "Amber monochrome",    accent: "#ffb700", bg: "#0f0d01" },
  { id: "blue",     label: "Blue Professional", desc: "Navy blue accent",    accent: "#4488ff", bg: "#0c1220" },
];

export const Settings = ({ user, onLogout, theme, setTheme }: {
  user: User;
  onLogout: () => void;
  theme: string;
  setTheme: (t: string) => void;
}) => (
  <div className="page">
    <div className="settings-col">
      <div className="card">
        <div className="card-title">Account</div>
        <div className="form-row">
          <div className="form-label">Email</div>
          <div className="dim mono">{user.email}</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Theme</div>
        <div className="theme-grid">
          {THEMES.map(t => (
            <button key={t.id} className={`theme-card ${theme === t.id ? "active" : ""}`} onClick={() => setTheme(t.id)}>
              <div className="theme-preview" style={{ background: t.bg, borderColor: t.accent }}>
                <div className="tp-bar" style={{ background: t.accent, width: "40%" }} />
                <div className="tp-bar" style={{ background: t.accent, width: "70%", opacity: 0.35 }} />
                <div className="tp-bar" style={{ background: t.accent, width: "55%", opacity: 0.2 }} />
              </div>
              <div className="theme-label">{t.label}</div>
              <div className="theme-desc">{t.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <button className="mini-btn danger" onClick={onLogout}>Sign Out</button>
    </div>
  </div>
);
