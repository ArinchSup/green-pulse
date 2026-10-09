package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		log.Println("usage: green-pulse <server|migrate>")
		os.Exit(2)
	}

	switch os.Args[1] {
	case "server":
		runServer()
	case "migrate":
		runMigrate()
	default:
		log.Printf("unknown command %q (want server or migrate)", os.Args[1])
		os.Exit(2)
	}
}

func runServer() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", handleHealthz)
	mux.HandleFunc("GET /readyz", handleReadyz)

	log.Println("listening on :8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatalf("server stopped: %v", err)
	}
}

func runMigrate() {
	log.Println("migrate: no migration yet, nothing to apply")
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func handleReadyz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ready"}`))
}
