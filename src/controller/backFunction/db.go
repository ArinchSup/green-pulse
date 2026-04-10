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