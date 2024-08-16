import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Mansi Dabriwal"]
usernames = ["mdabriwal"]
passwords = ["m123"]

hashed_password = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_password, file)
#This will create the binary file and  store hashed passwords in it.
print("Hashed passwords saved to hashed_pw.pkl")
