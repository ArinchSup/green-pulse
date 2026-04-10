package main

import (
	"encoding/json"
	"fmt"
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
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("request (%s) from (%s)\n", r.URL.Path, r.RemoteAddr)
	log.Println("Connecting to database")

	if err := backfunction.Init("../database/data/app.db"); err != nil {
		log.Fatal(err)
	}
	defer backfunction.DB.Close()
	log.Println("DB connected")

	var body User

	err := json.NewDecoder(r.Body).Decode(&body)
	if err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	hashedPassword, err := hashPassword(body.Password)
	if err != nil {
		http.Error(w, "error hashing password", http.StatusInternalServerError)
		return
	}

	log.Printf("Received signup request for user: %s\n", body.Username)
	log.Printf("Hashed password: %s\n", hashedPassword)

	err = backfunction.InsertToDB(body.Username, hashedPassword)
	if err != nil {
		http.Error(w, "error inserting user into database", http.StatusInternalServerError)
		log.Printf("Error inserting user %s: %v\n", body.Username, err)
		return
	} else {
		log.Printf("User %s signed up successfully\n", body.Username)
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("User signed up successfully"))
}

func main() {
	http.HandleFunc("/data", handlerData)
	http.HandleFunc("/signup", handlerSignUP)
	fmt.Println("Starting server on :8080")
	http.ListenAndServe(":8080", nil)
}
