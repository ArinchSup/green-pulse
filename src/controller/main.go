package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"bufio"

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

	users, err := backfunction.CheckAllUsers()
	if err != nil {
		log.Printf("error checking users: %v", err)
		return
	}
	log.Printf("All users: %v", users)
}


func main() {
	http.HandleFunc("/data", handlerData)
	http.HandleFunc("/signup", handlerSignUP)
	
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
