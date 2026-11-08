from pydantic import json
import yfinance as yf
import pandas as pd

pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))

def profitability(ticker):
    income_stmt = (yf.Ticker(ticker) if isinstance(ticker, str) else ticker).income_stmt
    if income_stmt.empty or not {'Total Revenue', 'Operating Income'}.issubset(income_stmt.index):
        return None

    revenue = pd.to_numeric(income_stmt.loc['Total Revenue'], errors='coerce')
    operating_income = pd.to_numeric(income_stmt.loc['Operating Income'], errors='coerce')
    margins = (operating_income / revenue.where(revenue > 0)).dropna().sort_index()
    if margins.empty:
        return None

    current_margin = margins.iloc[-1]
    score = 5.0 + (5.0 / 0.35) * current_margin

    return round(float(max(0.0, min(10.0, score))), 1)

def growth(ticker):
    income_stmt = (yf.Ticker(ticker) if isinstance(ticker, str) else ticker).income_stmt
    if income_stmt.empty or 'Total Revenue' not in income_stmt.index:
        return None

    revenue = pd.to_numeric(income_stmt.loc['Total Revenue'], errors='coerce').dropna().sort_index()
    if len(revenue) < 2 or (revenue <= 0).any():
        return None

    growth_rates = revenue.pct_change().dropna()
    latest_growth = growth_rates.iloc[-1]
    score = 4.0 + 20.0 * latest_growth

    earlier_growth = growth_rates.iloc[:-1]
    if not earlier_growth.empty:
        if (earlier_growth > 0).all():
            score += 0.5
        elif (earlier_growth < 0).all():
            score -= 0.5

    return round(float(max(0.0, min(10.0, score))), 1)

def financial_health(ticker):
    company = yf.Ticker(ticker) if isinstance(ticker, str) else ticker
    balance_sheet = company.balance_sheet
    cash_flow = company.cash_flow
    cash_row = next((name for name in (
        'Cash Cash Equivalents And Short Term Investments',
        'Cash And Cash Equivalents',
    ) if name in balance_sheet.index), None)
    required_rows = {'Current Assets', 'Current Liabilities', 'Total Debt'}
    shared_years = balance_sheet.columns.intersection(cash_flow.columns)
    if (cash_row is None or not required_rows.issubset(balance_sheet.index)
            or 'Operating Cash Flow' not in cash_flow.index or shared_years.empty):
        return None

    year = shared_years.sort_values()[-1]
    values = pd.to_numeric(pd.Series({
        'cash': balance_sheet.at[cash_row, year],
        'current_assets': balance_sheet.at['Current Assets', year],
        'current_liabilities': balance_sheet.at['Current Liabilities', year],
        'debt': balance_sheet.at['Total Debt', year],
        'operating_cash_flow': cash_flow.at['Operating Cash Flow', year],
    }), errors='coerce')
    if values.isna().any() or (values.drop('operating_cash_flow') < 0).any():
        return None

    cash = values['cash']
    operating_cash_flow = values['operating_cash_flow']
    runway_years = cash / -operating_cash_flow if operating_cash_flow < 0 else 3.0
    runway_score = 5.0 * min(runway_years / 3.0, 1.0)

    liabilities = values['current_liabilities']
    current_ratio = values['current_assets'] / liabilities if liabilities > 0 else 2.0
    coverage_score = 3.0 * min(current_ratio / 2.0, 1.0)

    debt = values['debt']
    debt_score = 2.0 * cash / (cash + debt) if cash + debt > 0 else 2.0

    return round(float(runway_score + coverage_score + debt_score), 1)

def give_me_foundation(ticker):
    feature1 = profitability(ticker)
    feature2 = growth(ticker)
    feature3 = financial_health(ticker)

    object = {
        "profitability": feature1,
        "growth": feature2,
        "financial_health": feature3
    }

    return object

if __name__ == "__main__":
    ticker = "NVDA"
    foundation = give_me_foundation(ticker)
    print(f"Foundation features for {ticker}:")
    print(foundation)