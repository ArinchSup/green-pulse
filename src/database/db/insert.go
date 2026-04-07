package db

import (
	"encoding/csv"
	"os"
	"strconv"
)

func InsertFromCSV(symbol, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return err
	}

	tx, err := DB.Begin()
	
	if err != nil{
		return err
	}

	stmt, err := tx.Prepare(`
		INSERT OR IGNORE INTO stocks
			(symbol, date, open, high, low, close, volume, dividends, stock_splits)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)

	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, row := range rows[1:] { // skip header
		open, _         := strconv.ParseFloat(row[1], 64)
		high, _         := strconv.ParseFloat(row[2], 64)
		low, _          := strconv.ParseFloat(row[3], 64)
		close, _        := strconv.ParseFloat(row[4], 64)
		volume, _       := strconv.ParseInt(row[5], 10, 64)
		dividends, _    := strconv.ParseFloat(row[6], 64)
		stockSplits, _  := strconv.ParseFloat(row[7], 64)

		_, err = stmt.Exec(symbol, row[0], open, high, low, close, volume, dividends, stockSplits)
		if err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}