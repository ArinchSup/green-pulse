import myApiFunction
connect = False

def sign_in():
    username = input("Username: ")
    password = input("Password: ")
    print(f"Signing in with username: {username} and password: {'*' * len(password)}")
    status = myApiFunction.rqSignIn(username, password)
    if status == 1:
        print("Sign in successful")
        global connect
        connect = True
    else:
        print("Sign in failed")

def sign_up():
    username = input("Choose a username: ")
    password = input("Choose a password: ")
    print(f"Signing up with username: {username} and password: {'*' * len(password)}")
    status = myApiFunction.rqSignUp(username, password)
    if status == 1:
        print("Sign up successful, please sign in to continue")
        sign_in()
    else:
        print("Sign up failed")

def start_test_server():
    import time
    print(time.ctime())
    print("User log in...")
    ch = input("sign in/sign up? (x/y)")
    if ch == "x":
        print("Signing in")
        sign_in()
    elif ch == "y":
        print("Signing up")
        sign_up()
    else:
        print("Invalid choice. Exiting.")
        return

if __name__ == "__main__":
    start_test_server()