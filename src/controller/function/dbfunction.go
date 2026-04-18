package function

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"

	readenv "github.com/joho/godotenv"
	pgxpool "github.com/jackc/pgx/v5/pgxpool"
)

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

	pool, err := pgxpool.New(context.Background(), dbURL)
	if err != nil {
		log.Fatalf("Failed to connect to databse: %v", err)
	}

	fmt.Println("Successfully connected to the database")
	return pool
}