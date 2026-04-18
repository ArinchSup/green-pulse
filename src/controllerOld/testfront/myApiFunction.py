import requests

server_url = "localhost:8080"

def dosomething():
    ticker = "GOOGL"
    url = f"http://{server_url}/data?ticker={ticker}"

    response = requests.get(url)
    print("Doing something...")
    if response.status_code == requests.codes.ok:
        print(response.headers.get("Content-Disposition"))
        with open("GOOGL.csv", "wb") as f:
            f.write(response.content)
        print(f"Downloaded data for {ticker} and saved to {ticker}.csv")

    else:
        print(f"Fail status code: {response.status_code}")

def rqSignIn(username, password):
    url = f"http://{server_url}/signin"
    data = {"username": username, "password": password}
    response = requests.post(url, json=data)
    if response.status_code == requests.codes.ok:
        print("Sign in successful")
        return 1
    else:
        print(f"Sign in failed with status code: {response.status_code}")
        return 0

def rqSignUp(username, password):
    url = f"http://{server_url}/signup"
    data = {"username": username, "password": password}
    response = requests.post(url, json=data)
    if response.status_code == requests.codes.ok:
        print(response.text)
    else:
        print(f"Sign up failed with status code: {response.status_code}")
        return 0
    
def rqGetWatchlist(username):
    url = f"http://localhost:8080/watchlist?username={username}"
    response = requests.get(url)
    if response.status_code == requests.codes.ok:
        watchlist = response.json().get("watchlist", [])
        return watchlist
    else:
        print(f"Failed to get watchlist with status code: {response.status_code}")
        return []

def rqUpdateWatchlist(username, watchlist):
    url = "http://localhost:8080/watchlist"
    data = {"username": username, "watchlist": watchlist}
    response = requests.post(url, json=data)
    if response.status_code == requests.codes.ok:
        print("Watchlist updated successfully")
    else:
        print(f"Failed to update watchlist with status code: {response.status_code}")