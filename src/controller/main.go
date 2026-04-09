package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

func handlerData(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		http.Error(w, "ticker parameter is required", http.StatusBadRequest)
		return
	}

	fmt.Printf("request for file ticker: %s", ticker)

	filepath := fmt.Sprintf("../database/data/raw/%s.csv", strings.ToUpper(ticker))

	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		http.Error(w, "ticker data not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.csv\"", strings.ToUpper(ticker)))
	http.ServeFile(w, r, filepath)
}

func handlerSignUP(w http.ResponseWriter, r *http.Request) {
	fmt.Printf("[%s] request (%s) from (%s)\n", time.Now().Format("2006-01-02 15:04:05"), r.URL.Path, r.RemoteAddr)
	fmt.Fprintf(w, "Signup endpoint - not implemented yet")
}

func main() {
	http.HandleFunc("/data", handlerData)
	http.HandleFunc("/signup", handlerSignUP)
	fmt.Println("Starting server on :8080")
	http.ListenAndServe(":8080", nil)
}
