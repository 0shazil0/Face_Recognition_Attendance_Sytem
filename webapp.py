from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


app = Flask(__name__)  # initializing


# database credentials
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "Your DataBase URL",
        "storageBucket": "Your Firebase Storage Bucket URL",
    },
)

bucket = storage.bucket()


def dataset(id):
    employeeInfo = db.reference(f"Employees/{id}").get()
    blob = bucket.get_blob(f"static/Files/Images/{id}.png")
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    imgEmployee = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    datetimeObject = datetime.strptime(
        employeeInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
    )
    secondElapsed = (datetime.now() - datetimeObject).total_seconds()
    return employeeInfo, imgEmployee, secondElapsed


already_marked_id_employee = []
already_marked_id_admin = []


def generate_frame():
    # Background and Different Modes

    # video camera
    capture = cv2.VideoCapture(1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    imgBackground = cv2.imread("static/Files/Resources/background.png")

    folderModePath = "static/Files/Resources/Modes/"
    modePathList = os.listdir(folderModePath)
    imgModeList = []

    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

    modeType = 0
    id = -1
    imgEmployee = []
    counter = 0

    # encoding loading ---> to identify if the person is in our database or not.... to detect faces that are known or not

    file = open("EncodeFile.p", "rb")
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodedFaceKnown, employeeIDs = encodeListKnownWithIds

    while True:
        success, img = capture.read()

        if not success:
            break
        else:
            imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            faceCurrentFrame = face_recognition.face_locations(imgSmall)
            encodeCurrentFrame = face_recognition.face_encodings(
                imgSmall, faceCurrentFrame
            )

            imgBackground[162 : 162 + 480, 55 : 55 + 640] = img
            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]

            if faceCurrentFrame:
                for encodeFace, faceLocation in zip(
                    encodeCurrentFrame, faceCurrentFrame
                ):
                    matches = face_recognition.compare_faces(
                        encodedFaceKnown, encodeFace
                    )
                    faceDistance = face_recognition.face_distance(
                        encodedFaceKnown, encodeFace
                    )

                    matchIndex = np.argmin(faceDistance)

                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                    if matches[matchIndex] == True:
                        id = employeeIDs[matchIndex]

                        if counter == 0:
                            cvzone.putTextRect(
                                imgBackground, "Face Detected", (65, 200), thickness=2
                            )
                            cv2.waitKey(1)
                            counter = 1
                            modeType = 1
                    else:
                        cvzone.putTextRect(
                            imgBackground, "Face Detected", (65, 200), thickness=2
                        )
                        cv2.waitKey(3)
                        cvzone.putTextRect(
                            imgBackground, "Face Not Found", (65, 200), thickness=2
                        )
                        modeType = 4
                        counter = 0
                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                if counter != 0:
                    if counter == 1:
                        employeeInfo, imgEmployee, secondElapsed = dataset(id)
                        if secondElapsed > 60:
                            ref = db.reference(f"Employees/{id}")
                            employeeInfo["total_attendance"] += 1
                            ref.child("total_attendance").set(
                                employeeInfo["total_attendance"]
                            )
                            ref.child("last_attendance_time").set(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        else:
                            modeType = 3
                            counter = 0
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

                            already_marked_id_employee.append(id)
                            already_marked_id_admin.append(id)

                    if modeType != 3:
                        if 5 < counter <= 10:
                            modeType = 2

                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                        if counter <= 5:
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["total_attendance"]),
                                (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["Post"]),
                                (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(id),
                                (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["standing"]),
                                (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["year"]),
                                (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["starting_year"]),
                                (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )

                            (w, h), _ = cv2.getTextSize(
                                str(employeeInfo["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1
                            )

                            offset = (414 - w) // 2
                            cv2.putText(
                                imgBackground,
                                str(employeeInfo["name"]),
                                (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (50, 50, 50),
                                1,
                            )

                            imgEmployeeResize = cv2.resize(imgEmployee, (216, 216))

                            imgBackground[
                                175 : 175 + 216, 909 : 909 + 216
                            ] = imgEmployeeResize

                        counter += 1

                        if counter >= 10:
                            counter = 0
                            modeType = 0
                            employeeInfo = []
                            imgEmployee = []
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

            else:
                modeType = 0
                counter = 0

            ret, buffer = cv2.imencode(".jpeg", imgBackground)
            frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg \r\n\r\n" + frame + b"\r\n")


#########################################################################################################################


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


#########################################################################################################################


@app.route("/employee_login", methods=["GET", "POST"])
def employee_login():
    id = request.form.get("id_number", False)
    email = request.form.get("email", False)
    password = request.form.get("password", False)
    employeeIDs, _ = add_image_database()

    if id:
        if id not in employeeIDs:
            return render_template(
                "employee_login.html", data=" ❌ The id is not registered"
            )
        else:
            secret_key = f"{id}{email}{password}anythingyoulike"
            hash_secret_key = str(hash(secret_key))
            if (
                dataset(id)[0]["password"] == password
                and dataset(id)[0]["email"] == email
            ):
                return redirect(url_for("employee", data=id, title=hash_secret_key))
            else:
                id = False
                return render_template(
                    "employee_login.html", data=" ❌ Email/Password Incorrect"
                )
    else:
        return render_template("employee_login.html")


@app.route("/employee/<data>/<title>")
def employee(data, title=None):
    employeeInfo, imgEmployee, secondElapsed = dataset(data)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "employeeInfo": employeeInfo,
        "lastlogin": hoursElapsed,
        "image": imgEmployee,
    }
    return render_template("employee.html", data=info)


@app.route("/employee_attendance_list")
def employee_attendance_list():
    unique_id_employee = list(set(already_marked_id_employee))
    employee_info = []
    for i in unique_id_employee:
        employee_info.append(dataset(i))
    return render_template("employee_attendance_list.html", data=employee_info)

#########################################################################################################################


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    id = request.form.get("id_number", False)
    email = request.form.get("email", False)
    password = request.form.get("password", False)
    employeeIDs, _ = add_image_database()

    if id:
        if id not in employeeIDs:
            return render_template(
                "admin_login.html", data=" ❌ The id is not registered"
            )
        else:
            if (
                dataset(id)[0]["password"] == password
                and dataset(id)[0]["email"] == email
            ):
                return redirect(url_for("admin"))
            else:
                id = False
                return render_template(
                    "admin_login.html", data=" ❌ Email/Password Incorrect"
                )
    else:
        return render_template("admin_login.html")


@app.route("/admin")
def admin():
    all_employee_info = []
    employeeIDs, _ = add_image_database()
    for i in employeeIDs:
        all_employee_info.append(dataset(i))
    return render_template("admin.html", data=all_employee_info)


@app.route("/admin/admin_attendance_list", methods=["GET", "POST"])
def admin_attendance_list():
    if request.method == "POST":
        if request.form.get("button_employee") == "VALUE1":
            already_marked_id_employee.clear()
            return redirect(url_for("admin_attendance_list"))
        else:
            request.form.get("button_admin") == "VALUE2"
            already_marked_id_admin.clear()
            return redirect(url_for("admin_attendance_list"))
    else:
        unique_id_admin = list(set(already_marked_id_admin))
        employee_info = []
        for i in unique_id_admin:
            employee_info.append(dataset(i))
        return render_template("admin_attendance_list.html", data=employee_info)



#########################################################################################################################

def add_image_database():
    folderPath = "static/Files/Images"
    imgPathList = os.listdir(folderPath)
    imgList = []
    employeeIDs = []

    for path in imgPathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        employeeIDs.append(os.path.splitext(path)[0])

        fileName = f"{folderPath}/{path}"
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)

    return employeeIDs, imgList


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


@app.route("/admin/add_user", methods=["GET", "POST"])
def add_user():
    id = request.form.get("id", False)
    name = request.form.get("name", False)
    password = request.form.get("password", False)
    dob = request.form.get("dob", False)
    city = request.form.get("city", False)
    country = request.form.get("country", False)
    phone = request.form.get("phone", False)
    email = request.form.get("email", False)
    Post = request.form.get("Post", False)
    starting_year = request.form.get("starting_year", False)
    standing = request.form.get("standing", False)
    total_attendance = request.form.get("total_attendance", False)
    year = request.form.get("year", False)
    last_attendance_date = request.form.get("last_attendance_date", False)
    last_attendance_time = request.form.get("last_attendance_time", False)
    content = request.form.get("content", False)

    address = f"{city}, {country}"
    last_attendance_datetime = f"{last_attendance_date} {last_attendance_time}:00"
    year = int(year)
    total_attendance = int(total_attendance)
    starting_year = int(starting_year)

    if request.method == "POST":
        image = request.files["image"]
        filename = f"{'static/Files/Images'}/{id}.png"
        image.save(os.path.join(filename))

    employeeIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, employeeIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    if id:
        add_employee = db.reference(f"Employees")

        add_employee.child(id).set(
            {
                "id": id,
                "name": name,
                "password": password,
                "dob": dob,
                "address": address,
                "phone": phone,
                "email": email,
                "Post": Post,
                "starting_year": starting_year,
                "standing": standing,
                "total_attendance": total_attendance,
                "year": year,
                "last_attendance_time": last_attendance_datetime,
                "content": content,
            }
        )

    return render_template("add_user.html")


#########################################################################################################################


@app.route("/admin/edit_user", methods=["POST", "GET"])
def edit_user():
    value = request.form.get("edit_employee")

    employeeInfo, imgEmployee, secondElapsed = dataset(value)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "employeeInfo": employeeInfo,
        "lastlogin": hoursElapsed,
        "image": imgEmployee,
    }

    return render_template("edit_user.html", data=info)


#########################################################################################################################


@app.route("/admin/save_changes", methods=["POST", "GET"])
def save_changes():
    content = request.get_data()

    dic_data = json.loads(content.decode("utf-8"))

    dic_data = {k: v.strip() for k, v in dic_data.items()}

    dic_data["year"] = int(dic_data["year"])
    dic_data["total_attendance"] = int(dic_data["total_attendance"])
    dic_data["starting_year"] = int(dic_data["starting_year"])

    update_employee = db.reference(f"Employees")

    update_employee.child(dic_data["id"]).update(
        {
            "id": dic_data["id"],
            "name": dic_data["name"],
            "dob": dic_data["dob"],
            "address": dic_data["address"],
            "phone": dic_data["phone"],
            "email": dic_data["email"],
            "Post": dic_data["Post"],
            "starting_year": dic_data["starting_year"],
            "standing": dic_data["standing"],
            "total_attendance": dic_data["total_attendance"],
            "year": dic_data["year"],
            "last_attendance_time": dic_data["last_attendance_time"],
            "content": dic_data["content"],
        }
    )

    return "Data received successfully!"


#########################################################################################################################


def delete_image(employee_id):
    filepath = f"static/Files/Images/{employee_id}.jpg"

    os.remove(filepath)

    bucket = storage.bucket()
    blob = bucket.blob(filepath)
    blob.delete()

    return "Successful"


@app.route("/admin/delete_user", methods=["POST", "GET"])
def delete_user():
    content = request.get_data()

    employee_id = json.loads(content.decode("utf-8"))

    delete_employee = db.reference(f"Employees")
    delete_employee.child(employee_id).delete()

    delete_image(employee_id)

    employeeIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, employeeIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    return "Successful"


#########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
