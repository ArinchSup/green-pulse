import sqlite3
import datetime
import pandas as pd
import yfinance as yf

DB_FILE = "forward_test.db"
MAX_FORWARD_DAYS = 90

def update_trade_in_db(trade_id, status, exit_date, exit_price, pnl_pct, stop_at_be):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE trades
        SET status = ?, exit_date = ?, exit_price = ?, pnl_pct = ?, stop_at_breakeven = ?
        WHERE id = ?
    """, (status, exit_date, exit_price, pnl_pct, stop_at_be, trade_id))
    conn.commit()
    conn.close()

def evaluate_open_positions():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, ticker, signal_date, entry_price, target_price, stop_loss, halfway_target, stop_at_breakeven
        FROM trades
        WHERE status = 'OPEN'
    """)
    open_trades = cursor.fetchall()
    conn.close()

    if not open_trades:
        print("ℹ️ No active 'OPEN' positions to track.")
        return

    print(f"🔍 Evaluating {len(open_trades)} active position(s)...")
    today = datetime.date.today()

    for trade in open_trades:
        t_id, ticker, sig_date_str, entry, target, stop, halfway, be_flag = trade
        sig_date = datetime.datetime.strptime(sig_date_str, "%Y-%m-%d").date()
        
        # Download bars strictly after signal date
        start_date = sig_date + datetime.timedelta(days=1)
        if start_date > today:
            continue

        df = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"))
        if df.empty:
            continue

        current_stop = entry if be_flag else stop
        stop_at_be = be_flag

        resolved = False
        for current_dt, row in df.iterrows():
            high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])
            date_str = current_dt.strftime("%Y-%m-%d")

            # 1. Trailing Stop: lock permanently to breakeven once halfway is reached
            if high >= halfway and not stop_at_be:
                stop_at_be = 1
                current_stop = entry

            # 2. Target Hit
            if high >= target and low > current_stop:
                pnl = round(((target - entry) / entry) * 100, 2)
                update_trade_in_db(t_id, "WIN", date_str, target, pnl, stop_at_be)
                print(f"🎯 WIN: {ticker:<5} hit Target ${target:.2f} on {date_str} (+{pnl}%)")
                resolved = True
                break

            # 3. Stop Hit (Breakeven or Loss)
            if low <= current_stop:
                if stop_at_be:
                    update_trade_in_db(t_id, "BREAKEVEN", date_str, entry, 0.0, stop_at_be)
                    print(f"🛡️ BREAKEVEN: {ticker:<5} stopped out at Entry on {date_str} (0.00%)")
                else:
                    pnl = round(((current_stop - entry) / entry) * 100, 2)
                    update_trade_in_db(t_id, "LOSS", date_str, current_stop, pnl, stop_at_be)
                    print(f"❌ LOSS: {ticker:<5} stopped out on {date_str} ({pnl}%)")
                resolved = True
                break

        # 4. Check Expiration
        if not resolved:
            days_held = (today - sig_date).days
            if days_held >= MAX_FORWARD_DAYS:
                last_close = float(df["Close"].iloc[-1])
                pnl = round(((last_close - entry) / entry) * 100, 2)
                update_trade_in_db(t_id, "EXPIRED", today.strftime("%Y-%m-%d"), last_close, pnl, stop_at_be)
                print(f"⏱️ EXPIRED: {ticker:<5} reached {MAX_FORWARD_DAYS} days ({pnl:+0.2f}%)")
            elif stop_at_be != be_flag:
                # Update trailing stop status even if trade is still open
                conn = sqlite3.connect(DB_FILE)
                conn.cursor().execute("UPDATE trades SET stop_at_breakeven = 1 WHERE id = ?", (t_id,))
                conn.commit()
                conn.close()

def display_dashboard():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    if df.empty:
        return

    wins = len(df[df["status"] == "WIN"])
    losses = len(df[df["status"] == "LOSS"])
    be = len(df[df["status"] == "BREAKEVEN"])
    open_t = len(df[df["status"] == "OPEN"])
    exp = len(df[df["status"] == "EXPIRED"])

    active_completed = wins + losses + be
    resolved_total = wins + losses

    strict_wr = (wins / active_completed * 100) if active_completed > 0 else 0.0
    resolved_wr = (wins / resolved_total * 100) if resolved_total > 0 else 0.0
    cap_pres = ((wins + be) / active_completed * 100) if active_completed > 0 else 0.0

    resolved_df = df[df["status"].isin(["WIN", "LOSS", "BREAKEVEN", "EXPIRED"])]
    net_pnl = resolved_df["pnl_pct"].sum()
    avg_ret = resolved_df["pnl_pct"].mean() if not resolved_df.empty else 0.0

    print("\n" + "=" * 65)
    print("📊 LIVE FORWARD TEST PERFORMANCE DASHBOARD")
    print("=" * 65)
    print(f"Positions  : Total: {len(df)} | Open: {open_t} | W: {wins} | L: {losses} | BE: {be} | Exp: {exp}")
    print(f"Strict WR  [ W / (W+L+BE) ]       : {strict_wr:5.2f}%")
    print(f"Resolved WR [ W / (W+L) ]         : {resolved_wr:5.2f}%")
    print(f"Capital Preservation [(W+BE)/Act] : {cap_pres:5.2f}%")
    print(f"Net Realized Return (Sum)         : {net_pnl:+6.2f}%")
    print(f"Average Realized Return / Trade   : {avg_ret:+6.2f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    evaluate_open_positions()
    display_dashboard()