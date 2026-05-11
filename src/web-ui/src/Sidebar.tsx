// src/Sidebar.tsx
import type { Dispatch, SetStateAction } from "react";
import { NavItems } from "./variable";

export const Sidebar = ({ activePage, setActivePage }: {
  activePage: string;
  setActivePage: Dispatch<SetStateAction<string>>;
}) => (
  <div className="sidebar">
    <div className="logo">
      <span className="logo-mark"></span>
      GREEN PULSE
    </div>
    <div className="nav">
      {NavItems.map(n => (
        <div key={n.id}
             className={`nav-item ${activePage === n.id ? "active" : ""}`}
             onClick={() => setActivePage(n.id)}>
          <span className={`nav-icon ${n.id}`}></span>
          {n.label}
        </div>
      ))}
    </div>
  </div>
);
