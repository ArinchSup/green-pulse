package backfunction

import (
	"database/sql"
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