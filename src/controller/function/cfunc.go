package function

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"

	auth "golang.org/x/oauth2"
	authG "golang.org/x/oauth2/google"
)

var ConfigAuth *auth.Config

func InitConfig() {
	clientID := strings.TrimSpace(os.Getenv("ClientID"))
	clientSecret := strings.TrimSpace(os.Getenv("ClientSecret"))

	if clientID == "" {
		log.Println("ClientID is missing in environment variables")
	}
	if clientSecret == "" {
		log.Println("ClientSecret is missing in environment variables")
	}
	ConfigAuth = &auth.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURL:  "http://localhost:8080/callback",
		Scopes:       []string{"https://www.googleapis.com/auth/userinfo.email"},
		Endpoint:     authG.Endpoint,
	}
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
	if ConfigAuth == nil {
		http.Error(w, "OAuth is not configured", http.StatusInternalServerError)
		return
	}
	url := ConfigAuth.AuthCodeURL("state-Random") // need to be random for security
	log.Printf("Redirecting to Google OAuth URL")
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

func CallbackHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Callback received")
	if r.FormValue("state") != "state-Random" { // should match the state sent in the auth code URL
		http.Error(w, "State is invalid", http.StatusBadRequest)
		return
	}
	code := r.FormValue("code")
	token, err := ConfigAuth.Exchange(r.Context(), code)
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

	var googleUser struct {
		ID    string `json:"id"`
		Email string `json:"email"`
	}
	json.NewDecoder(resp.Body).Decode(&googleUser)

	query := `
		INSERT INTO users (google_id, email)
		VALUES ($1, $2)
		ON CONFLICT (google_id) DO UPDATE SET email = EXCLUDED.email
		RETURNING id;
	`

	internalID, err := SyncGoogleUser(query, googleUser.ID, googleUser.Email)
	if err != nil {
		log.Printf("database error: %v", err)
		http.Error(w, "Failed to sync user", http.StatusInternalServerError)
		return
	}

	log.Printf("User synced with internal id: %s", internalID)

	http.Redirect(w, r, "http://127.0.0.1:5500/src/controller/frontendtest/index.html", http.StatusSeeOther)
}
