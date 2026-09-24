package main

import (
	"log"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	function "github.com/ArinchSup/green-pulse/src/controller/function"
)

func ExitHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("exit server request")
	log.Println("shutdown server")
	os.Exit(0)
}

func ok(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func startServer() {
	log.Println("Start Server")
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", ok)
	mux.HandleFunc("GET /readyz", ok)
	mux.HandleFunc("/api/watchlist", function.WatchlistHandler)

	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Printf("Server error due to %v", err)
	}
}

func scheduleStockRefreshes() {
	go func() {
		if err := function.RefreshAllStocks(); err != nil {
			log.Printf("initial stock refresh failed: %v", err)
		}

		location, err := time.LoadLocation("America/New_York")
		if err != nil {
			log.Printf("failed to load market timezone: %v", err)
			return
		}

		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		lastRunDate := ""
		for range ticker.C {
			now := time.Now().In(location)
			if now.Weekday() == time.Saturday || now.Weekday() == time.Sunday {
				continue
			}

			marketClose := time.Date(now.Year(), now.Month(), now.Day(), 16, 5, 0, 0, location)
			if now.Before(marketClose) {
				continue
			}

			currentDate := now.Format("2006-01-02")
			if lastRunDate == currentDate {
				continue
			}

			if err := function.RefreshAllStocks(); err != nil {
				log.Printf("scheduled stock refresh failed: %v", err)
				continue
			}

			lastRunDate = currentDate
			log.Printf("scheduled stock refresh completed for %s", currentDate)
		}
	}()
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	if len(os.Args) < 2 {
		log.Println("Usage: go run main.go <startserver|connectdb>")
		return
	}
	arg := strings.ToLower(os.Args[1])
	switch arg {
	case "server", "startserver":
		db := function.ConnectDB()
		defer db.Close()
		startServer()
	case "connectdb":
		function.LoadEnv()
		db := function.ConnectDB()
		defer db.Close()
	case "migrate":
		runMigrate()
	default:
		log.Printf("Unknown command: %s", arg)
	}
}
