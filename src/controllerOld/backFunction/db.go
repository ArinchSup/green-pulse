package backfunction

import (
	"database/sql"
	"fmt"
	"strings"

	_ "modernc.org/sqlite"
)

var DB *sql.DB

func Init(path string) error {
	var err error
	DB, err = sql.Open("sqlite", path)
	if err != nil {
		return err
	}
	return DB.Ping()
}

func InsertToDB(username, password string) error {
	_, err := DB.Exec(`
		INSERT INTO users (username, password) VALUES (?, ?)
	`, username, password)
	return err
}

func CheckAllUsers() ([]string, error) {
	rows, err := DB.Query(`SELECT username FROM users`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []string
	for rows.Next() {
		var username string
		if err := rows.Scan(&username); err != nil {
			return nil, err
		}
		users = append(users, username)
	}
	return users, nil
}

func UpdateWatchlist(username string, symbols []string) error {
	tx, err := DB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	_, err = tx.Exec(`DELETE FROM watchlist WHERE username = ?`, username)
	if err != nil {
		return err
	}

	for _, symbol := range symbols {
		_, err := tx.Exec(`INSERT INTO watchlist (username, symbol) VALUES (?, ?)`, username, symbol)
		if err != nil {
			return err
		}
	}

	return tx.Commit()
}

func GetWatchlist(username string) ([]string, error) {
	rows, err := DB.Query(`SELECT symbol FROM watchlist WHERE username = ?`, username)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var watchlist []string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err != nil {
			return nil, err
		}
		watchlist = append(watchlist, symbol)
	}
	return watchlist, nil
}

func QueryCustom(query string) ([]string, error) {
	rows, err := DB.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	var results []string
	values := make([]interface{}, len(columns))
	valuePtrs := make([]interface{}, len(columns))
	for i := range values {
		valuePtrs[i] = &values[i]
	}

	for rows.Next() {
		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, err
		}

		parts := make([]string, len(columns))
		for i, v := range values {
			switch val := v.(type) {
			case nil:
				parts[i] = "NULL"
			case []byte:
				parts[i] = string(val)
			default:
				parts[i] = fmt.Sprint(val)
			}
		}
		results = append(results, strings.Join(parts, " | "))
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

func DeleteCustom(query string) error {
	_, err := DB.Exec(query)
	return err
}
