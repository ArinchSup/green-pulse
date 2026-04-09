import requests

ticker = "GOOGL"
url = f"http://localhost:8080/data?ticker={ticker}"

response = requests.get(url)

if response.status_code == requests.codes.ok:
    print(response.headers.get("Content-Disposition"))
    with open("GOOGL.csv", "wb") as f:
        f.write(response.content)
    print(f"Downloaded data for {ticker} and saved to {ticker}.csv")

else:
    print(f"Fail status code: {response.status_code}")