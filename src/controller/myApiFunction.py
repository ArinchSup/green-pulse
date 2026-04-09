import requests

def dosomething():
    ticker = "GOOGL"
    url = f"http://localhost:8080/data?ticker={ticker}"

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
    url = "http://localhost:8080/signin"
    data = {"username": username, "password": password}
    response = requests.post(url, json=data)
    if response.status_code == requests.codes.ok:
        print("Sign in successful")
        return 1
    else:
        print(f"Sign in failed with status code: {response.status_code}")
        return 0

def rqSignUp(username, password):
    url = "http://localhost:8080/signup"
    data = {"username": username, "password": password}
    response = requests.post(url, json=data)
    if response.status_code == requests.codes.ok:
        print("Sign up successful")
        return 1
    else:
        print(f"Sign up failed with status code: {response.status_code}")
        return 0