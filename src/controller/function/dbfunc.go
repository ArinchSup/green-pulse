package function

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	pgxpool "github.com/jackc/pgx/v5/pgxpool"
	readenv "github.com/joho/godotenv"
)

var pool *pgxpool.Pool

type StockRecord struct {
	Symbol      string  `json:"symbol"`
	Date        string  `json:"date"`
	Open        float64 `json:"open"`
	High        float64 `json:"high"`
	Low         float64 `json:"low"`
	Close       float64 `json:"close"`
	Volume      int64   `json:"volume"`
	Dividends   float64 `json:"dividends"`
	StockSplits float64 `json:"stock_splits"`
}

func LoadEnv() {
	if err := readenv.Load("myEnv.env"); err != nil {
		log.Println("error to load env file")
		log.Printf("%v", err)
	}
}

func ConnectDB() *pgxpool.Pool {
	GetPass := os.Getenv("SUPABASEPASS")
	password := url.QueryEscape(GetPass)

	dbURL := fmt.Sprintf(
		"postgresql://postgres.wrqibbqyjukbdzobmtkf:%s@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres", password)

	config, err := pgxpool.ParseConfig(dbURL)
	if err != nil {
		log.Fatalf("Failed to parse db config: %v", err)
	}

	// Supabase transaction pooler requires these
	config.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol
	config.MaxConns = 10
	config.MinConns = 1

	dbPool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	// Force a real connection to verify
	if err := dbPool.Ping(context.Background()); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
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

func GetGoogleUserID(googleID string) (string, error) {
	if pool == nil {
		return "", errors.New("database pool is not initialized")
	}

	var internalID string
	err := pool.QueryRow(
		context.Background(),
		"SELECT id FROM users WHERE google_id = $1",
		googleID,
	).Scan(&internalID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", err
	}

	return internalID, nil
}

func CreateGoogleUser(googleID string, email string) (string, error) {
	if pool == nil {
		return "", errors.New("database pool is not initialized")
	}

	var internalID string
	err := pool.QueryRow(
		context.Background(),
		"INSERT INTO users (google_id, email) VALUES ($1, $2) RETURNING id",
		googleID,
		email,
	).Scan(&internalID)
	if err != nil {
		return "", err
	}

	return internalID, nil
}

func StockExists(symbol string) (bool, error) {
	if pool == nil {
		return false, errors.New("database pool is not initialized")
	}

	var exists bool
	err := pool.QueryRow(
		context.Background(),
		"SELECT EXISTS (SELECT 1 FROM stocks WHERE UPPER(symbol) = UPPER($1))",
		symbol,
	).Scan(&exists)
	if err != nil {
		return false, err
	}

	return exists, nil
}

func GetStockRecords(symbol string) ([]StockRecord, error) {
	if pool == nil {
		return nil, errors.New("database pool is not initialized")
	}

	rows, err := pool.Query(
		context.Background(),
		`
			SELECT symbol, date, open, high, low, close, volume, dividends, stock_splits
			FROM stocks
			WHERE UPPER(symbol) = UPPER($1)
			ORDER BY date DESC
		`,
		symbol,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []StockRecord
	for rows.Next() {
		var record StockRecord
		var dateValue time.Time
		if err := rows.Scan(
			&record.Symbol,
			&dateValue,
			&record.Open,
			&record.High,
			&record.Low,
			&record.Close,
			&record.Volume,
			&record.Dividends,
			&record.StockSplits,
		); err != nil {
			return nil, err
		}

		record.Date = dateValue.Format("2006-01-02")
		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return records, nil
}

func TrackedSymbols() ([]string, error) {
	print("working 2\n")

	if pool == nil {
		return nil, errors.New("database pool is not initialized")
	}
	print("working 3\n")


	rows, err := pool.Query(context.Background(), "SELECT DISTINCT symbol FROM stocks ORDER BY symbol")
	if err != nil {
		print("error here\n")
		return nil, err
	}
	print("working 4\n")

	defer rows.Close()

	var symbols []string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err != nil {
			return nil, err
		}
		symbols = append(symbols, symbol)
	}
	print("working 5\n")
	if err := rows.Err(); err != nil {
		return nil, err
	}
	print("working 6\n")
	return symbols, nil
}

func FetchStockData(symbol string) (string, error) {
	if pool == nil {
		return "", errors.New("database pool is not initialized")
	}

	database := os.Getenv("DATABASE")
	password := os.Getenv("SUPABASEPASS")
	region := os.Getenv("REGION")
	if database == "" || password == "" || region == "" {
		return "", errors.New("database credentials are not configured")
	}

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", errors.New("failed to resolve fetch script path")
	}

	scriptPath := filepath.Clean(filepath.Join(filepath.Dir(currentFile), "..", "script", "fetch.py"))

	cmd := exec.Command("python3", scriptPath, symbol, database, password, region)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 2 {
			return strings.TrimSpace(string(output)), errors.New("unknown symbol")
		}
		return strings.TrimSpace(string(output)), err
	}

	return strings.TrimSpace(string(output)), nil
}

func RefreshAllStocks() error {
	print("working \n")
	symbols, err := TrackedSymbols()
	if err != nil {
		return err
	}
	print("working \n")

	for _, symbol := range symbols {
		if output, err := FetchStockData(symbol); err != nil {
			log.Printf("refresh fetch failed for %s: %v (%s)", symbol, err, output)
		}
	}

	return nil
}

func GetUserFavorites(userID string) ([]map[string]string, error) {
	if pool == nil {
		return nil, errors.New("database pool is not initialized")
	}

	query := `SELECT symbol FROM favorites WHERE user_id = $1 ORDER BY added_at DESC;`

	rows, err := pool.Query(context.Background(), query, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var favorites []map[string]string
	for rows.Next() {
		var symbol string
		if err := rows.Scan(&symbol); err != nil {
			return nil, err
		}
		favorites = append(favorites, map[string]string{"symbol": symbol})
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return favorites, nil
}

func AddFavorite(userID string, symbol string) error {
	if pool == nil {
		return errors.New("database pool is not initialized")
	}

	query := `
		INSERT INTO favorites (user_id, symbol)
		VALUES ($1, $2)
		ON CONFLICT (user_id, symbol) DO NOTHING;
	`

	_, err := pool.Exec(context.Background(), query, userID, strings.ToUpper(symbol))
	return err
}

func RemoveFavorite(userID string, symbol string) error {
	if pool == nil {
		return errors.New("database pool is not initialized")
	}

	query := `DELETE FROM favorites WHERE user_id = $1 AND symbol = $2;`

	_, err := pool.Exec(context.Background(), query, userID, strings.ToUpper(symbol))
	return err
}
