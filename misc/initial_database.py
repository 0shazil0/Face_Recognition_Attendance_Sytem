import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "your firebase database url",
        
    },
)

ref = db.reference(
    "Employees"
)  # reference path to our database... will create employee directory in the database

data = {
    "0001": {  # id of employee which is a key
        "id": "0001",
        "name": "Shazil Shaikh",
        "password": "12345",
        "dob": "2004-06-23",
        "address": "Hyderabad, Pakistan",
        "phone": "0000000000",
        "email": "shazilshaikh756@gmail.com",
        "Post": "AI Intern",
        "starting_year": 2020,
        "standing": "G",
        "total_attendance": 0,
        "year": 2,
        "last_attendance_time": "2023-02-21 12:33:10",
        "content": "This section aims to offer essential guidance for employees",
    },
}


for key, value in data.items():
    ref.child(key).set(value)
