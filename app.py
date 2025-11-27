from flask import Flask, render_template, jsonify, redirect, url_for, request
import subprocess
from threading import Thread
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import uuid
import base64
import io
import json
from PIL import Image
import numpy as np
import face_recognition
from datetime import datetime, time, timedelta, timezone
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import math

app = Flask(__name__)
CORS(app)

# ---------- Constants for Location ----------
OFFICE_LAT = 3.205170
OFFICE_LNG = 101.720107
WFO_RADIUS = 100.0   # meters
WFH_RADIUS = 500.0   # meters

# ---------- Malaysia Time (UTC+8, no ZoneInfo needed) ----------
MALAYSIA_TZ = timezone(timedelta(hours=8))

# ---------- Enhanced Geocoder for Place Names ----------
geolocator = Nominatim(user_agent="attendance_system_v2")


def get_place_name(lat, lng, max_retries=3):
    """
    Returns a human-friendly place name (e.g., 'Setapak Central Mall')
    Uses OpenStreetMap POI tags for best result.
    """
    if lat is None or lng is None:
        return "Location not provided"

    for _ in range(max_retries):
        try:
            location = geolocator.reverse(
                f"{lat}, {lng}",
                language='en',
                zoom=18,
                addressdetails=True
            )
            if not location:
                return "Unknown location"

            addr = location.raw.get('address', {})

            place_name = (
                addr.get('building') or
                addr.get('amenity') or
                addr.get('shop') or
                addr.get('leisure') or
                addr.get('tourism') or
                addr.get('public_building') or
                addr.get('university') or
                addr.get('school') or
                (addr.get('house_number', '') + ' ' + addr.get('road', '')).strip() or
                location.address
            ).strip()

            return place_name if place_name else "Location not identified"
        except GeocoderTimedOut:
            continue
    return "Geocoding failed"


# ---------- Distance & Location Status Helpers ----------

def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Returns distance in metres between two lat/lng points.
    """
    R = 6371e3  # metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_location_status(distance_m, mode):
    """
    Returns location status text, like 'At Office', 'Far from Office', 'At Home', etc.
    """
    if mode == 'wfo':
        if distance_m < 50:
            return "At Office"
        elif distance_m < 500:
            return "Near Office"
        else:
            return "Far from Office"
    elif mode == 'wfh':
        if distance_m < 50:
            return "At Home"
        elif distance_m < 500:
            return "Near Home"
        else:
            return "Far from Home"
    else:
        # Fallback if no mode
        if distance_m < 50:
            return "At Office"
        elif distance_m < 500:
            return "Near Office"
        else:
            return "Far from Office"


# ---------- Firebase ----------
cred_path = "serviceAccountKey.json"
if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Firebase credential file '{cred_path}' not found!")

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(credentials.Certificate(cred_path))

db = firestore.client()

scan_results = {}

# ---------- Configuration ----------
TOLERANCE = 0.45
# On time if before or at 12:30 PM (Malaysia time)
ON_TIME_END = time(12, 30)

ATTENDANCE_COLLECTION = "attendance_test"  # <--- main collection for attendance


# ---------- Frontend Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/attendance")
def attendance():
    return render_template("attendance.html")


@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/admin-dashboard")
def admin_dashboard():
    return render_template("admin-dashboard.html")


@app.route("/admin-attendancelog")
def admin_attendancelog():
    return render_template("admin-attendancelog.html")


@app.route("/leavecalendar")
def leave_calendar():
    return render_template("leavecalendar.html")


@app.route("/admin-leavecalendar")
def admin_leave_calendar():
    return render_template("admin-leavecalendar.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/manage-staff")
def manage_staff():
    return render_template("manage-staff.html")


@app.route("/admin-liveloc")
def admin_liveloc():
    return render_template("admin-liveloc.html")


# ---------- (Optional) API: Legacy get-attendance ----------
@app.route("/get-attendance")
def get_attendance():
    """
    Fetch attendance records from the attendance_test collection.
    (Mostly for debugging; your frontend now reads directly from Firestore.)
    """
    try:
        docs = (
            db.collection(ATTENDANCE_COLLECTION)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
        records = []
        for doc in docs:
            data = doc.to_dict()
            records.append({
                "name": data.get("name", "Unknown"),
                "timestamp": data.get("timestamp", ""),
                "status": data.get("status", "Unknown"),
                "latitude": data.get("checkInLocation", {}).get("latitude"),
                "longitude": data.get("checkInLocation", {}).get("longitude"),
                "address": data.get("address", "N/A"),
                "userId": data.get("userId", "")
            })
        return jsonify(records)
    except Exception as e:
        print("❌ Error fetching attendance:", e)
        return jsonify([])


# ---------- Face recognition runner (if you still use live_recognition.py) ----------
def run_face_recognition(scan_id):
    try:
        subprocess.run(["python", "live_recognition.py"])
        scan_results[scan_id] = {"status": "completed"}
    except Exception as e:
        scan_results[scan_id] = {"status": "failed", "error": str(e)}


@app.route("/scan")
def scan():
    scan_id = str(uuid.uuid4())
    Thread(target=run_face_recognition, args=(scan_id,)).start()
    return redirect(url_for("attendance", scan_id=scan_id))


@app.route("/scan-status/<scan_id>")
def scan_status(scan_id):
    return jsonify(scan_results.get(scan_id, {"status": "running"}))


@app.route("/clear-scan/<scan_id>")
def clear_scan(scan_id):
    scan_results.pop(scan_id, None)
    return jsonify({"status": "cleared"})


# ---------- Helper: Find user in users collection ----------

def find_user_in_users_collection(recognized_label: str):
    """
    recognized_label is what comes from known_faces.json (e.g. 'Syed Omar').

    Your Firestore structure:
      - firstName = 'Syed Omar'
      - lastName  = 'Syed Osman'

    We:
      1) Search users where firstName == recognized_label (exact match).
      2) If found:
         - user_id      = document ID (e.g. VQ4cEU4v3OQH2LcROZEsCHdsJhy1)
         - first_name   = data['firstName']
         - last_name    = data.get('lastName', '')
         - full_name    = first_name + ' ' + last_name
         - homeLocation = data.get('homeLocation', {lat,lng})
    """
    if not recognized_label:
        return None

    label_str = recognized_label.strip()
    if not label_str:
        return None

    try:
        users_ref = db.collection("users")
        q = users_ref.where("firstName", "==", label_str).limit(1)
        docs = list(q.stream())
        if not docs:
            print(f"⚠️ No user found in users collection with firstName == '{label_str}'")
            return None

        doc_ref = docs[0]
        data = doc_ref.to_dict() or {}

        first_name = data.get("firstName", label_str)
        last_name = (data.get("lastName") or "").strip()

        full_name = f"{first_name} {last_name}".strip()

        home_loc = data.get("homeLocation") or {}
        home_lat = home_loc.get("lat")
        home_lng = home_loc.get("lng")

        result = {
            "userId": doc_ref.id,
            "firstName": first_name,
            "lastName": last_name,
            "fullName": full_name,
            "home_lat": home_lat,
            "home_lng": home_lng,
            "raw": data
        }
        print(f"✅ Found user in users: {result['userId']} → {result['fullName']}")
        return result
    except Exception as e:
        print("❌ Error searching users collection:", e)
        return None


# ---------- Recognition + Save Attendance (Malaysia time) ----------

@app.route("/recognize", methods=["POST"])
def recognize():
    """
    1. Decode image and detect faces.
    2. Match against known_faces.json.
    3. For each recognized face:
       - Look up user in users collection by firstName == recognized_label.
       - Build:
           firstName, lastName, fullName, userId
       - Save into ATTENDANCE_COLLECTION with:
           Document ID = firstName_YYYY-MM-DD
           userId = Firestore user doc ID
           full display name = firstName + " " + lastName
       - Use Malaysia timezone for timestamp/check-in.
    """
    try:
        data = request.get_json() or {}

        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Location & work mode
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        work_mode = data.get("work_mode")  # 'wfo' or 'wfh'
        # front-end sends this for WFH
        home_lat_client = data.get("home_lat")
        home_lng_client = data.get("home_lng")

        # Decode image
        try:
            # Handle "data:image/png;base64,...."
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            rgb_image = np.array(image)
        except Exception as e:
            print("❌ Error decoding image:", e)
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        # Load known faces
        if not os.path.exists("known_faces.json"):
            return jsonify({"success": False, "error": "No known faces enrolled"}), 400

        with open("known_faces.json", "r") as f:
            known_data = json.load(f)

        if not known_data.get("encodings") or not known_data.get("names"):
            return jsonify({"success": False, "error": "Known faces database is empty"}), 400

        known_encodings = [np.array(enc) for enc in known_data["encodings"]]
        known_names = known_data["names"]

        # Encode faces in captured image
        unknown_encodings = face_recognition.face_encodings(rgb_image)

        if not unknown_encodings:
            return jsonify({"success": False, "error": "No face detected"}), 400

        # Malaysia time now (fixed UTC+8)
        now_my = datetime.now(MALAYSIA_TZ)
        current_time = now_my.time()
        # Time status: On Time / Late
        check_in_time_status = "On Time" if current_time <= ON_TIME_END else "Late"

        # Text formats
        check_in_time_str = now_my.strftime("%I:%M %p").lower()  # "09:56 am"
        time_str = now_my.strftime("%I:%M %p")                   # "09:56 AM"
        date_iso = now_my.strftime("%Y-%m-%d")                   # "2025-11-24"
        date_slash = now_my.strftime("%d/%m/%Y")                 # "24/11/2025"

        # Get place name once (for this scan)
        place_name = get_place_name(latitude, longitude)

        recognized_entries = []

        for unknown_encoding in unknown_encodings:
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            best_match_index = int(np.argmin(face_distances))
            best_distance = float(face_distances[best_match_index])

            if best_distance > TOLERANCE:
                # Face not recognized; skip saving attendance
                continue

            label_name = known_names[best_match_index]
            print(f"🙂 Recognized face as '{label_name}' with distance {best_distance:.4f}")

            # 🔍 Find user in users collection by firstName == label_name
            user_info = find_user_in_users_collection(label_name)

            if user_info:
                user_id = user_info["userId"]
                first_name = user_info["firstName"]
                last_name = user_info["lastName"]
                full_name = user_info["fullName"]
            else:
                # Fallback: no user record, we still save something
                user_id = ""
                first_name = label_name
                last_name = ""
                full_name = label_name
                print(f"⚠️ No matching user doc found for '{label_name}', saving with empty userId")

            # Distance for location
            dist_m = None
            dist_m_rounded = None
            if latitude is not None and longitude is not None:
                if work_mode == "wfh" and home_lat_client is not None and home_lng_client is not None:
                    dist_m = haversine_distance_m(float(home_lat_client), float(home_lng_client),
                                                  float(latitude), float(longitude))
                else:
                    dist_m = haversine_distance_m(OFFICE_LAT, OFFICE_LNG,
                                                  float(latitude), float(longitude))
                dist_m_rounded = round(dist_m, 1)

            # Location status
            if dist_m is not None:
                check_in_status = get_location_status(dist_m, work_mode)
            else:
                check_in_status = "Location unavailable"

            # Document ID like "Syed Omar_2025-11-25"
            doc_id = f"{first_name}_{date_iso}"

            check_in_location = None
            if latitude is not None and longitude is not None:
                check_in_location = {
                    "latitude": float(latitude),
                    "longitude": float(longitude)
                }

            # Location type: e.g. Office / Home
            if work_mode == "wfo":
                location_type = "Office"
            elif work_mode == "wfh":
                location_type = "Home"
            else:
                location_type = ""

            # Build attendance document
            attendance_doc = {
                "checkIn": check_in_time_str,                    # "03:38 pm"
                "checkInDistance": str(dist_m_rounded) if dist_m_rounded is not None else "",
                "checkInLocation": check_in_location,
                "checkInStatus": check_in_status,                # "At Office", "Far from Office", etc.
                "checkInTimeStatus": check_in_time_status,       # "On Time" / "Late"

                "checkOut": None,
                "checkOutDistance": None,
                "checkOutLocation": None,
                "checkOutStatus": None,
                "checkOutTimeStatus": None,

                "date": date_slash,                              # "24/11/2025"
                "lastUpdated": now_my,                           # Firestore Timestamp
                "locationType": location_type,                   # "Office" / "Home"
                "name": full_name,                               # full display name
                "status": "Check In",                            # e.g. "Check In"
                "time": time_str,                                # e.g. "03:38 PM"
                "userId": user_id,                               # Firestore user doc ID
                "userName": full_name,                           # display name
                "timestamp": now_my.isoformat(),                 # extra for sorting
                "address": place_name
            }

            try:
                db.collection(ATTENDANCE_COLLECTION).document(doc_id).set(attendance_doc, merge=True)
                print(f"✅ Saved attendance for '{full_name}' with docId '{doc_id}' in '{ATTENDANCE_COLLECTION}'")
            except Exception as e:
                print("❌ Error saving attendance:", e)

            recognized_entries.append({
                "name": full_name,
                "userId": user_id,
                "status": "Check In",
                "address": place_name,
                "timestamp": now_my.strftime("%Y-%m-%d %H:%M:%S"),
                "docId": doc_id,
                "distance": best_distance,
            })

        if not recognized_entries:
            return jsonify({"success": False, "error": "Face not recognized"}), 400

        primary = recognized_entries[0]

        return jsonify({
            "success": True,
            "recognized": recognized_entries,
            "name": primary["name"],
            "userId": primary["userId"],
            "status": primary["status"],
            "address": place_name,
            "timestamp": primary["timestamp"],
        })

    except Exception as e:
        print("❌ Error in /recognize:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- Simple CHECKOUT (no face scan) ----------

@app.route("/checkout", methods=["POST"])
def checkout():
    """
    Simple checkout:
      - Frontend sends { userId, latitude, longitude }.
      - We look for today's attendance_test record(s) for that user.
      - NO Firestore composite index required (we do NOT use order_by).
      - We pick the latest document for today (by lastUpdated / timestamp) and update:
          checkOut, checkOutLocation, lastUpdated, status.
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("userId")
        lat = data.get("latitude")
        lng = data.get("longitude")

        if not user_id:
            return jsonify({"success": False, "error": "Missing userId"}), 400

        now_my = datetime.now(MALAYSIA_TZ)
        today_slash = now_my.strftime("%d/%m/%Y")  # "DD/MM/YYYY"

        attendance_ref = db.collection(ATTENDANCE_COLLECTION)

        # 🔍 Query only by userId (single field) – no index needed
        query_ref = attendance_ref.where("userId", "==", user_id)
        docs = list(query_ref.stream())

        if not docs:
            return jsonify({
                "success": False,
                "error": "No check-in record found to check out from."
            }), 404

        # Filter to today's docs and pick the latest via lastUpdated / timestamp
        today_docs = []
        for snap in docs:
            doc_data = snap.to_dict() or {}
            if doc_data.get("date") == today_slash:
                today_docs.append((snap, doc_data))

        if not today_docs:
            return jsonify({
                "success": False,
                "error": "No check-in record found for today."
            }), 404

        def get_doc_datetime(doc_data):
            # Prefer lastUpdated (Firestore timestamp)
            lu = doc_data.get("lastUpdated")
            if lu is not None:
                return lu
            # Fallback to timestamp string
            ts = doc_data.get("timestamp")
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except Exception:
                    return datetime.min.replace(tzinfo=MALAYSIA_TZ)
            return datetime.min.replace(tzinfo=MALAYSIA_TZ)

        # Choose latest
        latest_snap, latest_data = max(today_docs, key=lambda pair: get_doc_datetime(pair[1]))
        doc_ref = latest_snap.reference

        checkout_time_12h = now_my.strftime("%I:%M %p").lower()  # "05:32 pm"

        update_data = {
            "checkOut": checkout_time_12h,
            "lastUpdated": now_my,
            "status": "Checked out"
        }

        # Optional: save checkout location if provided
        if lat is not None and lng is not None:
            try:
                lat_f = float(lat)
                lng_f = float(lng)
                update_data["checkOutLocation"] = {
                    "latitude": lat_f,
                    "longitude": lng_f
                }
            except Exception:
                # ignore invalid lat/lng
                pass

        doc_ref.update(update_data)
        print(f"✅ Checkout updated for userId '{user_id}' on doc '{doc_ref.id}'")

        return jsonify({
            "success": True,
            "message": "Checked out successfully",
            "checkOut": checkout_time_12h
        })

    except Exception as e:
        print("❌ Checkout error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- Staff Live Location API ----------
@app.route("/api/staff-live-locations")
def get_staff_live_locations():
    """Fetch current live locations of staff from 'staff_locations' collection."""
    try:
        docs = db.collection("staff_locations").stream()
        locations = []
        for doc_ref in docs:
            data = doc_ref.to_dict()
            locations.append({
                "userId": doc_ref.id,
                "name": data.get("name", "Unknown"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "lastUpdated": data.get("lastUpdated"),
                "status": data.get("status", "Offline")
            })
        return jsonify(locations)
    except Exception as e:
        print("❌ Error fetching live locations:", e)
        return jsonify([]), 500


# ---------- Optional: For testing location updates ----------
@app.route("/api/update-location", methods=["POST"])
def update_location():
    """Temporary endpoint to simulate staff location updates (for testing)."""
    try:
        data = request.get_json() or {}
        user_id = data.get("userId")
        name = data.get("name")
        lat = data.get("latitude")
        lng = data.get("longitude")

        if not all([user_id, name, lat, lng]):
            return jsonify({"error": "Missing required fields"}), 400

        db.collection("staff_locations").document(user_id).set({
            "name": name,
            "latitude": lat,
            "longitude": lng,
            "lastUpdated": datetime.utcnow().isoformat(),
            "status": "Active"
        }, merge=True)

        return jsonify({"success": True})
    except Exception as e:
        print("❌ Error updating location:", e)
        return jsonify({"error": str(e)}), 500


# ---------- Run ----------
if __name__ == "__main__":
    # For dev: Flask built-in server; for production: use waitress-serve
    app.run(debug=True, host="0.0.0.0", port=5000)
