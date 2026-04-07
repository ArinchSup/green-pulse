package db

func InitSchema() error{
	_, err := DB.Exec(`
		CREATE TABLE IF NOT EXISTS stocks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			symbol TEXT NOT NULL,
			date TEXT NOT NULL,
			open REAL,
			high REAL,
			low REAL,
			close REAL,
			volume INTEGER,
			dividends REAL DEFAULT 0,
			stock_splits REAL DEFAULT 0,
			UNIQUE(symbol, date)
		);

		CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
		CREATE INDEX IF NOT EXISTS idx_stocks_date on stocks(date);
	`)
	return err
}