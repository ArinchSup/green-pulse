package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/ArinchSup/green-pulse/src/controller/backFunction"
	"golang.org/x/crypto/bcrypt"
)

type User struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func hashPassword(password string) (string, error) {
	hashedBytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(hashedBytes), nil
}

func handlerData(w http.ResponseWriter, r *http.Request) {
	ticker := r.URL.Query().Get("ticker")
	if ticker == "" {
		http.Error(w, "ticker parameter is required", http.StatusBadRequest)
		return
	}

	log.Printf("request for file ticker: %s", ticker)

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
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("request (%s) from (%s)", r.URL.Path, r.RemoteAddr)
	log.Println("Connecting to database")

	if err := backfunction.Init("../database/data/app.db"); err != nil {
		log.Printf("database init error: %v", err)
		http.Error(w, "database connection error", http.StatusInternalServerError)
		return
	}
	defer backfunction.DB.Close()
	log.Println("DB connected")

	var body User
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	hashedPassword, err := hashPassword(body.Password)
	if err != nil {
		http.Error(w, "error hashing password", http.StatusInternalServerError)
		return
	}

	if err := backfunction.InsertToDB(body.Username, hashedPassword); err != nil {
		http.Error(w, "error inserting user into database", http.StatusInternalServerError)
		log.Printf("error inserting user %s: %v", body.Username, err)
		return
	}

	log.Printf("user %s signed up successfully", body.Username)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("User signed up successfully"))
}

func adminCommand() {
	log.Println("Admin command executed")
	if err := backfunction.Init("../database/data/app.db"); err != nil {
		log.Printf("database init error: %v", err)
		return
	}
	defer backfunction.DB.Close()

	fmt.Println("select options:")
	fmt.Println("1. List all users")
	fmt.Println("2. Delete a user")
	fmt.Println("3. Run custom query")
	fmt.Print("Enter option number: ")

	var option int
	_, err := fmt.Scan(&option)
	if err != nil {
		log.Printf("invalid input: %v", err)
		return
	}

	switch option {
	case 1:
		users, err := backfunction.CheckAllUsers()
		if err != nil {
			log.Printf("error checking users: %v", err)
			return
		}
		log.Printf("All users: %v", users)

	case 2:
		fmt.Print("Enter username to delete: ")
		var username string
		_, err := fmt.Scan(&username)
		if err != nil {
			log.Printf("invalid input: %v", err)
			return
		}
		if err := backfunction.DeleteCustom(fmt.Sprintf("DELETE FROM users WHERE username = '%s'", username)); err != nil {
			log.Printf("error deleting user: %v", err)
			return
		}
		log.Printf("User %s deleted successfully", username)

	case 3:
		fmt.Print("Enter custom query: ")
		reader := bufio.NewReader(os.Stdin)
		query := ""
		for {
			line, err := reader.ReadString('\n')
			if err != nil && err != io.EOF {
				log.Printf("invalid input: %v", err)
				return
			}

			line = strings.TrimSpace(line)
			if line != "" {
				query = line
				break
			}

			if err == io.EOF {
				log.Println("invalid input: empty query")
				return
			}
		}
		results, err := backfunction.QueryCustom(query)
		if err != nil {
			log.Printf("error running custom query: %v", err)
			return
		}
		log.Printf("Query results: %v", results)
	default:
		log.Println("Invalid option selected")
	}
}

func handlerSignIn(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("request (%s) from (%s)", r.URL.Path, r.RemoteAddr)
	log.Println("Connecting to database")

	if err := backfunction.Init("../database/data/app.db"); err != nil {
		log.Printf("database init error: %v", err)
		http.Error(w, "database connection error", http.StatusInternalServerError)
		return
	}
	defer backfunction.DB.Close()
	log.Println("DB connected")

	var body User
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	var storedHash string
	err := backfunction.DB.QueryRow(`SELECT password FROM users WHERE username = ?`, body.Username).Scan(&storedHash)
	if err != nil {
		log.Printf("error querying user: %v", err)
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
}

func handlerWatchlist(w http.ResponseWriter, r *http.Request) {
	username := r.URL.Query().Get("username")
	if username == "" {
		http.Error(w, "username parameter is required", http.StatusBadRequest)
		return
	}

	log.Printf("request for watchlist of user: %s", username)
	watchlist, err := backfunction.GetWatchlist(username)
	if err != nil {
		log.Printf("error getting watchlist: %v", err)
		http.Error(w, "error retrieving watchlist", http.StatusInternalServerError)
		return
	}
	response := map[string]interface{}{
		"watchlist": watchlist,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handlerUpdateWatchlist(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var body struct {
		Username  string   `json:"username"`
		Watchlist []string `json:"watchlist"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	log.Printf("request to update watchlist for user: %s", body.Username)
	if err := backfunction.UpdateWatchlist(body.Username, body.Watchlist); err != nil {
		log.Printf("error updating watchlist: %v", err)
		http.Error(w, "error updating watchlist", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("Watchlist updated successfully"))
}

func main() {
	http.HandleFunc("/data", handlerData)
	http.HandleFunc("/signup", handlerSignUP)
	http.HandleFunc("/signin", handlerSignIn)
	http.HandleFunc("/watchlist", handlerWatchlist)
	http.HandleFunc("/update_watchlist", handlerUpdateWatchlist)

	go func() {
		log.Println("Starting server on :8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("server error: %v", err)
		}
	}()

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		text := strings.ToLower(strings.TrimSpace(scanner.Text()))
		switch text {
		case "stop":
			log.Println("Shutting down server...")
			return
		case "admin":
			log.Println("Running admin command")
			adminCommand()
		default:
			fmt.Println("Unknown command.")
		}
	}
}
