package main

import (
	"context"
	"fmt"
	"os"
	
	readenv "github.com/joho/godotenv"
	pgx "github.com/jackc/pgx/v5"
)

func main() {

	err := readenv.Load("myenv.env")
	if err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
		return
	}

	config, err := pgx.ParseConfig("")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config init error: %v\n", err)
		os.Exit(1)
	}
	config.Host = "db.wrqibbqyjukbdzobmtkf.supabase.co"
	config.Port = 5432
	config.User = "postgres"
	config.Password = os.Getenv("supabaseDBPassword")
	config.Database = "postgres"

	conn, err := pgx.ConnectConfig(context.Background(), config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to connect to database: %v\n", err)
		os.Exit(1)
	}

	defer conn.Close(context.Background())

	err = conn.Ping(context.Background())
	if err != nil {
		fmt.Printf("Ping failed: %v\n", err)
		return
	}
	
	fmt.Println("Successfully connected to the database")

}
