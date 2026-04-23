package function

import (
	"encoding/json"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	auth "golang.org/x/oauth2"
	authG "golang.org/x/oauth2/google"
)

var ConfigAuth *auth.Config

const (
	stateSignin = "state-signin"
	stateSignup = "state-signup"
)

func InitConfig() {
	clientID := strings.TrimSpace(os.Getenv("CLIENTID"))
	clientSecret := strings.TrimSpace(os.Getenv("CLIENTSECRET"))

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
	if ConfigAuth == nil {
		http.Error(w, "OAuth is not configured", http.StatusInternalServerError)
		return
	}
	authURL := ConfigAuth.AuthCodeURL(stateSignin)
	log.Printf("Redirecting to Google OAuth URL for signin")
	http.Redirect(w, r, authURL, http.StatusTemporaryRedirect)
}

func SignupHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("user signup")
	if ConfigAuth == nil {
		http.Error(w, "OAuth is not configured", http.StatusInternalServerError)
		return
	}
	authURL := ConfigAuth.AuthCodeURL(stateSignup)
	log.Printf("Redirecting to Google OAuth URL for signup")
	http.Redirect(w, r, authURL, http.StatusTemporaryRedirect)
}

func CallbackHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Callback received")
	state := r.FormValue("state")
	if state != stateSignin && state != stateSignup {
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
	if err := json.NewDecoder(resp.Body).Decode(&googleUser); err != nil {
		http.Error(w, "Failed to decode user info", http.StatusInternalServerError)
		return
	}

	frontendURL := "http://127.0.0.1:5500/src/controller/frontendtest/index.html"

	if state == stateSignup {
		existingID, err := GetGoogleUserID(googleUser.ID)
		if err != nil {
			log.Printf("database lookup error: %v", err)
			http.Error(w, "Failed to check user", http.StatusInternalServerError)
			return
		}

		if existingID != "" {
			http.Redirect(w, r, frontendURL+"?signup_status=exists&email="+url.QueryEscape(googleUser.Email), http.StatusSeeOther)
			return
		}

		newID, err := CreateGoogleUser(googleUser.ID, googleUser.Email)
		if err != nil {
			log.Printf("database create error: %v", err)
			http.Error(w, "Failed to create user", http.StatusInternalServerError)
			return
		}

		log.Printf("User signed up with internal id: %s", newID)
		http.Redirect(w, r, frontendURL+"?signup_status=complete&email="+url.QueryEscape(googleUser.Email), http.StatusSeeOther)
		return
	}

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
	http.Redirect(w, r, frontendURL+"?token="+url.QueryEscape(token.AccessToken)+"&email="+url.QueryEscape(googleUser.Email)+"&user_id="+url.QueryEscape(internalID), http.StatusSeeOther)
}
