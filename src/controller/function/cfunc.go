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
		RedirectURL:  strings.TrimSpace(os.Getenv("REDIRECT_URL")),
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

	frontendURL := strings.TrimSpace(os.Getenv("FRONTEND_URL"))

	if state == stateSignup {
		existingID, err := GetGoogleUserID(googleUser.ID)
		if err != nil {
			log.Printf("database lookup error: %v", err)
			http.Error(w, "Failed to check user", http.StatusInternalServerError)
			return
		}
		if existingID != "" {
			log.Printf("Signup rejected: account already exists for google_id %s", googleUser.ID)
			http.Redirect(w, r, frontendURL+"?error=account_exists", http.StatusSeeOther)
			return
		}
		newID, err := CreateGoogleUser(googleUser.ID, googleUser.Email)
		if err != nil {
			log.Printf("database create error: %v", err)
			http.Error(w, "Failed to create user", http.StatusInternalServerError)
			return
		}
		log.Printf("New user signed up with internal id: %s", newID)
		http.Redirect(w, r, frontendURL+"?token="+url.QueryEscape(token.AccessToken)+"&email="+url.QueryEscape(googleUser.Email)+"&user_id="+url.QueryEscape(newID), http.StatusSeeOther)
		return
	}

	// Sign in: reject if no account exists
	existingID, err := GetGoogleUserID(googleUser.ID)
	if err != nil {
		log.Printf("database lookup error: %v", err)
		http.Error(w, "Failed to check user", http.StatusInternalServerError)
		return
	}
	if existingID == "" {
		log.Printf("Signin rejected: no account found for google_id %s", googleUser.ID)
		http.Redirect(w, r, frontendURL+"?error=no_account", http.StatusSeeOther)
		return
	}

	log.Printf("User signed in with internal id: %s", existingID)
	http.Redirect(w, r, frontendURL+"?token="+url.QueryEscape(token.AccessToken)+"&email="+url.QueryEscape(googleUser.Email)+"&user_id="+url.QueryEscape(existingID), http.StatusSeeOther)
}
