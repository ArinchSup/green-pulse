package function

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"

	pgxpool "github.com/jackc/pgx/v5/pgxpool"
	readenv "github.com/joho/godotenv"
)

var pool *pgxpool.Pool

func LoadEnv() {
	if err := readenv.Load("myEnv.env"); err != nil {
		log.Println("error to load env file")
		log.Printf("%v", err)
	}
}

func ConnectDB() *pgxpool.Pool {
	GetPass := os.Getenv("supaDBpass")
	password := url.QueryEscape(GetPass)
	database := os.Getenv("database")
	dbURL := fmt.Sprintf("postgresql://postgres:%s@db.%s.supabase.co:5432/postgres", password, database)

	dbPool, err := pgxpool.New(context.Background(), dbURL)
	if err != nil {
		log.Fatalf("Failed to connect to databse: %v", err)
	}

	fmt.Println("Successfully connected to the database")
	pool = dbPool
	return dbPool
}

func SyncGoogleUser(query string, googleID string, email string) (string, error) {
	if pool == nil {
		return "", errors.New("database pool is not initialized")
	}

	var internalID string
	err := pool.QueryRow(context.Background(), query, googleID, email).Scan(&internalID)
	if err != nil {
		return "", err
	}

	return internalID, nil
}
