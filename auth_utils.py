import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from passlib.context import CryptContext
from jose import jwt, JWTError

# Secret key - in production load from env var
SECRET_KEY = "replace_this_with_a_random_secret_in_prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Replace bcrypt w/ pbkdf2 (safe & Windows-friendly)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ----------------------
# JSON USER STORE
# ----------------------
USERS_FILE = "data/users.json"

def load_users() -> Dict[str, Dict]:
    """Load existing users from users.json"""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def save_users(users: Dict[str, Dict]):
    """Save users to users.json"""
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ----------------------
# Password Helpers
# ----------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# ----------------------
# JWT Helpers
# ----------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ----------------------
# User Operations
# ----------------------
def create_user(email: str, password: str, full_name: str = ""):
    users = load_users()
    if email in users:
        raise ValueError("User already exists")

    hashed = get_password_hash(password)
    user = {"email": email, "hashed_password": hashed, "full_name": full_name}

    users[email] = user
    save_users(users)

    return user


def authenticate_user(email: str, password: str):
    users = load_users()
    user = users.get(email)
    if not user:
        return None

    if not verify_password(password, user["hashed_password"]):
        return None

    return user


def get_user(email: str):
    users = load_users()
    return users.get(email)
