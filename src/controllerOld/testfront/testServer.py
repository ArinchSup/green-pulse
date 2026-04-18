import myApiFunction
connect = False

def sign_in():
    username = input("Username: ")
    password = input("Password: ")
    print(f"Signing in with username: {username} and password: {'*' * len(password)}")
    status = myApiFunction.rqSignIn(username, password)
    if status == 1:
        global connect
        connect = True
        return username
    else:
        return None

def sign_up():
    username = input("Choose a username: ")
    password = input("Choose a password: ")
    print(f"Signing up with username: {username} and password: {'*' * len(password)}")
    myApiFunction.rqSignUp(username, password)


def start_test_server():
    import time
    print(time.ctime())
    print("User log in...")
    ch = input("sign in/sign up? (x/y)")
    if ch == "x":
        print("Signing in")
        return sign_in()
    elif ch == "y":
        print("Signing up")
        sign_up()
        return None
    else:
        print("Invalid choice. Exiting.")
        return None

if __name__ == "__main__":
    username = start_test_server()
    if connect:
        print("Welcome !")
        watchlist = myApiFunction.rqGetWatchlist(username)
        print("Your watchlist:", watchlist)
        print("Insert your watchlist (comma separated):")
        new_watchlist = input().split(",")
        myApiFunction.rqUpdateWatchlist(username, new_watchlist)

    else:
        print("bye")
    