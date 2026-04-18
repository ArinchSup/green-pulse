package main

import (
	"log"
	"net/http"
	"os"
	"strings"

	function "github.com/ArinchSup/green-pulse/src/controller/function"
	auth "golang.org/x/oauth2"
	authG "golang.org/x/oauth2/google"
)

var googleConfig *auth.Config

func initGoogleConfig() error {
	clientID := strings.TrimSpace(os.Getenv("ClientID"))
	clientSecret := strings.TrimSpace(os.Getenv("ClientSecret"))

	if clientID == "" {
		return &missingEnvError{key: "ClientID"}
	}
	if clientSecret == "" {
		return &missingEnvError{key: "ClientSecret"}
	}

	googleConfig = &auth.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURL:  "http://localhost:8080/callback",
		Scopes:       []string{"https://www.googleapis.com/auth/userinfo.email"},
		Endpoint:     authG.Endpoint,
	}

	return nil
}

type missingEnvError struct {
	key string
}

func (e *missingEnvError) Error() string {
	return "missing required env variable: " + e.key
}

func SigninHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("user signin")
}

func SignupHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("user signup")
	if googleConfig == nil {
		http.Error(w, "OAuth is not configured", http.StatusInternalServerError)
		return
	}
	url := googleConfig.AuthCodeURL("state-Random") // need to be random for security
	log.Printf("Redirecting to Google OAuth URL")
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

func CallbackHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Callback received")
	log.Printf("State: %s", r.FormValue("state"))
	log.Printf("Code: %s", r.FormValue("code"))
	if r.FormValue("state") != "state-Random" { // should match the state sent in the auth code URL
		http.Error(w, "State is invalid", http.StatusBadRequest)
		return
	}
	code := r.FormValue("code")
	token, err := googleConfig.Exchange(r.Context(), code)
	if err != nil {
		http.Error(w, "Failed to exchange token: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp, err := http.Get("https://www.googleapis.com/oauth2/v2/userinfo?access_token=" + token.AccessToken)
	if err != nil {
		http.Error(w, "Failed to get user info", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	http.Redirect(w, r, "http://127.0.0.1:5500/src/controller/frontendtest/index.html", http.StatusSeeOther)
}

func ExitHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("exit server request")
	log.Println("shutdown server")
	os.Exit(0)
}

func startServer() {
	log.Println("Start server")
	mux := http.NewServeMux()

	//handler functions
	mux.HandleFunc("/signin", SigninHandler)
	mux.HandleFunc("/signup", SignupHandler)
	mux.HandleFunc("/callback", CallbackHandler)
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
		if err := initGoogleConfig(); err != nil {
			log.Printf("OAuth configuration error: %v", err)
			return
		}
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
