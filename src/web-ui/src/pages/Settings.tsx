// src/pages/Settings.tsx
import { useState } from "react";
import type { Alert } from "../types";
import { Pill } from "../primitives";
import { fmtPrice } from "../format";

const Toggle = ({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) => (
  <div className="form-row toggle-row" onClick={onClick}>
    <div className="form-label">{label}</div>
    <div className={`toggle ${on ? "on" : ""}`}><div className="toggle-knob" /></div>
  </div>
);

export const Settings = ({ alerts, setAlerts }: {
  alerts: Alert[];
  setAlerts: React.Dispatch<React.SetStateAction<Alert[]>>;
}) => {
  const [profile, setProfile] = useState({ name: "Alex Morgan", email: "alex@greenpulse.io", tier: "Pro" });
  const [prefs, setPrefs] = useState({ notifications: true, sound: false, news: true, marketHours: true, twoFA: true, biometric: false });
  const toggle = (k: keyof typeof prefs) => setPrefs(p => ({ ...p, [k]: !p[k] }));
  const toggleAlert = (id: number) => setAlerts(a => a.map(x => x.id === id ? { ...x, active: !x.active } : x));

  return (
    <div className="page">
      <div className="grid-settings">
        <div className="card">
          <div className="card-title">Account</div>
          <div className="form-row"><div className="form-label">Name</div>
            <input className="form-input" value={profile.name} onChange={e => setProfile({...profile, name: e.target.value})} /></div>
          <div className="form-row"><div className="form-label">Email</div>
            <input className="form-input" value={profile.email} onChange={e => setProfile({...profile, email: e.target.value})} /></div>
          <div className="form-row"><div className="form-label">Tier</div><div><Pill tone="up">{profile.tier}</Pill></div></div>
          <div className="form-row"><div className="form-label">Member since</div><div className="dim mono">2024-08-12</div></div>
        </div>
        <div className="card">
          <div className="card-title">Notifications</div>
          <Toggle label="Push notifications"   on={prefs.notifications} onClick={() => toggle("notifications")} />
          <Toggle label="Sound on price alert" on={prefs.sound}         onClick={() => toggle("sound")} />
          <Toggle label="Newswire digest"      on={prefs.news}          onClick={() => toggle("news")} />
          <Toggle label="Market hours only"    on={prefs.marketHours}   onClick={() => toggle("marketHours")} />
        </div>
        <div className="card">
          <div className="card-title">Security</div>
          <Toggle label="Two-factor auth" on={prefs.twoFA}     onClick={() => toggle("twoFA")} />
          <Toggle label="Biometric login" on={prefs.biometric} onClick={() => toggle("biometric")} />
          <div className="form-row"><div className="form-label">Active sessions</div><div className="dim mono">2 devices</div></div>
          <button className="mini-btn danger" style={{ marginTop: 8 }}>Sign out everywhere</button>
        </div>
        <div className="card span-2">
          <div className="card-title">Price Alerts</div>
          <table className="data-table">
            <thead><tr><th>Symbol</th><th>Condition</th><th>Target</th><th>Status</th><th></th></tr></thead>
            <tbody>
              {alerts.map(a => (
                <tr key={a.id}>
                  <td className="bold">{a.ticker}</td>
                  <td className="mono">{a.condition === ">" ? "above" : "below"}</td>
                  <td className="num">${fmtPrice(a.target)}</td>
                  <td><Pill tone={a.active ? "up" : "neutral"}>{a.active ? "armed" : "paused"}</Pill></td>
                  <td style={{ textAlign: "right" }}>
                    <button className="mini-btn" onClick={() => toggleAlert(a.id)}>{a.active ? "Pause" : "Arm"}</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
