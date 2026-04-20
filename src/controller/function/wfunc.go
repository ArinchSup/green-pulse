package function

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

func WatchlistHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	symbol := strings.TrimSpace(r.URL.Query().Get("symbol"))
	if symbol == "" {
		http.Error(w, "missing symbol", http.StatusBadRequest)
		return
	}

	records, err := GetStockRecords(symbol)
	if err != nil {
		log.Printf("stock lookup error: %v", err)
		http.Error(w, "failed to look up symbol", http.StatusInternalServerError)
		return
	}

	if len(records) == 0 {
		output, err := FetchStockData(symbol)
		if err != nil {
			if err.Error() == "unknown symbol" {
				w.Header().Set("Content-Type", "text/plain; charset=utf-8")
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte("unknown symbol"))
				return
			}

			log.Printf("stock fetch error: %v (%s)", err, output)
			http.Error(w, "failed to fetch symbol", http.StatusInternalServerError)
			return
		}

		records, err = GetStockRecords(symbol)
		if err != nil {
			log.Printf("stock lookup after fetch error: %v", err)
			http.Error(w, "failed to load symbol data", http.StatusInternalServerError)
			return
		}
	}

	if len(records) == 0 {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("unknown symbol"))
		return
	}

	response := map[string]any{
		"symbol":  strings.ToUpper(symbol),
		"count":   len(records),
		"records": records,
	}

	encoded, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Printf("response encode error: %v", err)
		http.Error(w, "failed to encode data", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(encoded)
}

func FavoritesHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	if userID == "" {
		http.Error(w, "missing user_id", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		favorites, err := GetUserFavorites(userID)
		if err != nil {
			log.Printf("favorites lookup error: %v", err)
			http.Error(w, "failed to look up favorites", http.StatusInternalServerError)
			return
		}

		if favorites == nil {
			favorites = []map[string]string{}
		}

		encoded, err := json.MarshalIndent(favorites, "", "  ")
		if err != nil {
			log.Printf("response encode error: %v", err)
			http.Error(w, "failed to encode data", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(encoded)

	case http.MethodPost:
		symbol := strings.TrimSpace(r.URL.Query().Get("symbol"))
		if symbol == "" {
			http.Error(w, "missing symbol", http.StatusBadRequest)
			return
		}

		err := AddFavorite(userID, symbol)
		if err != nil {
			log.Printf("add favorite error: %v", err)
			http.Error(w, "failed to add favorite", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"added"}`))

	case http.MethodDelete:
		symbol := strings.TrimSpace(r.URL.Query().Get("symbol"))
		if symbol == "" {
			http.Error(w, "missing symbol", http.StatusBadRequest)
			return
		}

		err := RemoveFavorite(userID, symbol)
		if err != nil {
			log.Printf("remove favorite error: %v", err)
			http.Error(w, "failed to remove favorite", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"removed"}`))

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
