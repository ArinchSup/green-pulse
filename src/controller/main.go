package main

import (
	"log"
	"net/http"
	"os"
	"strings"

	function "github.com/ArinchSup/green-pulse/src/controller/function"
)


func ExitHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("exit server request")
	log.Println("shutdown server")
	os.Exit(0)
}

func startServer() {
	log.Println("Start server")
	mux := http.NewServeMux()

	//handler functions
	mux.HandleFunc("/signin", function.SigninHandler)
	mux.HandleFunc("/signup", function.SignupHandler)
	mux.HandleFunc("/callback", function.CallbackHandler)
	mux.HandleFunc("/exit", ExitHandler)
	//-----------------

	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Printf("Server error due to %v", err)
	}
}

func main() {
	if len(os.Args) < 2 {
		log.Println("Usage: go run main.go <startserver|connectdb>")
		return
	}
	arg := strings.ToLower(os.Args[1])
	switch arg {
	case "startserver":
		function.LoadEnv()
		function.InitConfig()
		db := function.ConnectDB()
		defer db.Close()
		startServer()
	case "connectdb":
		function.LoadEnv()
		db := function.ConnectDB()
		defer db.Close()
	default:
		log.Printf("Unknown command: %s", arg)
	}
}
