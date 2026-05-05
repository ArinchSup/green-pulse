// src/TradeModal.tsx
import { useState } from "react";
import type { Market } from "./types";
import { fmtPrice, fmtPct, fmtMoney } from "./format";

export const TradeModal = ({ market, onClose }: { market: Market; onClose: () => void }) => {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState<"MARKET" | "LIMIT" | "STOP">("MARKET");
  const [shares, setShares] = useState(10);
  const [limit, setLimit] = useState(market.price);
  const [confirmed, setConfirmed] = useState(false);

  const total = shares * (orderType === "MARKET" ? market.price : limit);
  const submit = () => {
    setConfirmed(true);
    setTimeout(onClose, 1400);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        {confirmed ? (
          <div className="trade-confirmed">
            <div className="confirm-icon">✓</div>
            <div className="confirm-title">Order placed</div>
            <div className="confirm-detail mono">
              {side} {shares} {market.ticker} @ ${fmtPrice(orderType === "MARKET" ? market.price : limit)}
            </div>
          </div>
        ) : (
          <>
            <div className="modal-header">
              <div>
                <div className="modal-title">{side === "BUY" ? "Buy" : "Sell"} {market.ticker}</div>
                <div className="modal-sub">{market.label} · ${fmtPrice(market.price)} <span className={market.up ? "up" : "down"}>{fmtPct(market.change)}</span></div>
              </div>
              <button className="x-btn" onClick={onClose}>×</button>
            </div>
            <div className="seg">
              <button className={side === "BUY"  ? "seg-btn buy active"  : "seg-btn"} onClick={() => setSide("BUY")}>Buy</button>
              <button className={side === "SELL" ? "seg-btn sell active" : "seg-btn"} onClick={() => setSide("SELL")}>Sell</button>
            </div>
            <div className="seg small">
              {(["MARKET","LIMIT","STOP"] as const).map(t => (
                <button key={t} className={orderType === t ? "seg-btn active" : "seg-btn"} onClick={() => setOrderType(t)}>{t}</button>
              ))}
            </div>
            <div className="form-row stack">
              <div className="form-label">Shares</div>
              <input className="form-input mono" type="number" value={shares}
                     onChange={e => setShares(Math.max(0, parseFloat(e.target.value) || 0))} />
            </div>
            {orderType !== "MARKET" && (
              <div className="form-row stack">
                <div className="form-label">{orderType === "LIMIT" ? "Limit" : "Stop"} price</div>
                <input className="form-input mono" type="number" value={limit}
                       onChange={e => setLimit(parseFloat(e.target.value) || 0)} />
              </div>
            )}
            <div className="trade-summary">
              <div className="ts-row"><span>Estimated</span><span className="mono">{fmtMoney(total)}</span></div>
              <div className="ts-row"><span>Fees</span><span className="mono">$0.00</span></div>
              <div className="ts-row total"><span>Total</span><span className="mono">{fmtMoney(total)}</span></div>
            </div>
            <button className={`primary-btn ${side.toLowerCase()}`} onClick={submit}>
              Place {side.toLowerCase()} order
            </button>
            <div className="legal-note">Paper trading mode · No real funds at risk</div>
          </>
        )}
      </div>
    </div>
  );
};
