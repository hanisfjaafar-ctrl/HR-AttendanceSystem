import cv2
import face_recognition
import os
import numpy as np
import json
import sys
import time

# --- Configuration ---
FACES_DIR = "faces"
PREP_DURATION = 3      # seconds before scanning starts
SCAN_DURATION = 10     # seconds to scan for a known face
TOLERANCE = 0.45       # Lower = stricter (0.6 is default; 0.4–0.5 recommended for security)

# --- Load known faces ---
known_encodings = []
known_names = []

if not os.path.exists(FACES_DIR):
    print(json.dumps({"recognized": False, "error": "Faces directory 'faces' not found"}))
    sys.exit(1)

valid_extensions = {'.png', '.jpg', '.jpeg'}
for filename in os.listdir(FACES_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext in valid_extensions:
        img_path = os.path.join(FACES_DIR, filename)
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"⚠️ No face found in {filename}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}", file=sys.stderr)

print(f"✅ Loaded {len(known_names)} known faces: {known_names}", flush=True)

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(json.dumps({"recognized": False, "error": "Cannot access webcam"}))
    sys.exit(1)

# --- Preparation Phase ---
prep_start = time.time()
while time.time() - prep_start < PREP_DURATION:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(
        frame,
        f"Get Ready... Scanning in {max(0, int(PREP_DURATION - (time.time() - prep_start)))}s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        print(json.dumps({"recognized": False, "error": "User cancelled during preparation"}))
        sys.exit(0)

print("⏳ Starting face scan...", flush=True)

# --- Scanning Phase ---
detected = False
result = {"recognized": False}
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enforce scan timeout
    if time.time() - start_time > SCAN_DURATION:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_in_frame = False

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance <= TOLERANCE:
            name = known_names[best_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            result = {
                "recognized": True,
                "name": name,
                "user_id": name
            }
            detected = True
            recognized_in_frame = True
            print(f"✅ Recognized: {name} (distance: {best_distance:.3f})", flush=True)
            break
        else:
            # Unknown person — draw red box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"❌ Unknown face (closest distance: {best_distance:.3f})", flush=True)

    # If no faces detected at all, optionally show message (optional)
    if not face_locations:
        cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if detected:
        cv2.waitKey(2000)  # Show success for 2 seconds
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

# --- Final JSON output ---
if not detected:
    result = {"recognized": False, "error": "No recognized face found within time limit"}

print(json.dumps(result))
sys.stdout.flush()