package db

import "fmt"

func GetSymbols() ([]string, error) {
	rows, err := DB.Query(`SELECT DISTINCT symbol FROM stocks ORDER BY symbol`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var symbols []string
	for rows.Next() {
		var s string
		rows.Scan(&s)
		symbols = append(symbols, s)
	}
	return symbols, nil
}

func SymbolExists(symbol string) (bool, error) {
	var count int
	err := DB.QueryRow(`SELECT COUNT(*) FROM stocks WHERE symbol = ?`, symbol).Scan(&count)
	return count > 0, err
}

func GetStockInfo(symbol, from, to string) error {
	rows, err := DB.Query(`
		SELECT date, open, high, low, close, volume
		FROM stocks
		WHERE symbol = ? AND date BETWEEN ? AND ?
		ORDER BY date
	`, symbol, from, to)
	if err != nil {
		return err
	}
	defer rows.Close()

	fmt.Printf("%-12s %-10s %-10s %-10s %-10s %-15s\n", "Date", "Open", "High", "Low", "Close", "Volume")
	fmt.Println("-------------------------------------------------------------------------")
	for rows.Next() {
		var date string
		var open, high, low, close float64
		var volume int64
		rows.Scan(&date, &open, &high, &low, &close, &volume)
		fmt.Printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f %-15d\n", date, open, high, low, close, volume)
	}
	return nil
}