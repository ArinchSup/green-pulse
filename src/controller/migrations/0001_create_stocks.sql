-- 0001: daily price history for the demo backend.
-- Columns match StockRecord in function/dbfunc.go.
CREATE TABLE IF NOT EXISTS stocks (
    symbol       text             NOT NULL,
    date         date             NOT NULL,
    open         double precision,
    high         double precision,
    low          double precision,
    close        double precision,
    volume       bigint,
    dividends    double precision,
    stock_splits double precision,
    PRIMARY KEY (symbol, date)
);

ALTER TABLE stocks ENABLE ROW LEVEL SECURITY;