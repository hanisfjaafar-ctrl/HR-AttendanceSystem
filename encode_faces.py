import face_recognition
import json
import os

# Directory with reference images (name.jpg)
KNOWN_FACES_DIR = "faces"
JSON_FILE = "known_faces.json"

def encode_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]  # Extract name from filename
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0].tolist())  # Convert to list for JSON
                known_names.append(name)
                print(f"Encoded {name}")
            else:
                print(f"No face found in {filename}")

    # Save to JSON
    with open(JSON_FILE, "w") as f:
        json.dump({"names": known_names, "encodings": known_encodings}, f)
    print(f"Saved {len(known_names)} faces to {JSON_FILE}")

if __name__ == "__main__":
    encode_known_faces()