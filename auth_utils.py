import os
import json
import bcrypt

USERS_FILE = "users.json"

def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(name: str, email: str, password: str, age: str, gender: str, location: str) -> tuple:
    users = _load_users()
    if email in users:
        return False, "Email already registered."
    
    users[email] = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "age": age,
        "gender": gender,
        "location": location
    }
    _save_users(users)
    return True, "Registration successful! Please log in."

def login_user(email: str, password: str) -> tuple:
    users = _load_users()
    if email not in users:
        return False, None, "Invalid email or password."
    
    user = users[email]
    if verify_password(password, user["password_hash"]):
        profile = {k: v for k, v in user.items() if k != "password_hash"}
        return True, profile, "Login successful!"
    return False, None, "Invalid email or password."