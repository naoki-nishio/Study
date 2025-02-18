from PIL import Image
import torch
from model.face_recognition import FaceRecognizer
from utils.face_database import FaceDatabase

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = FaceRecognizer(device)
    face_db = FaceDatabase()

    enroll_image_path = "data/enroll.jpg"
    enroll_image = Image.open(enroll_image_path)
    enrollment_embedding = recognizer.get_embedding(enroll_image)
    if enrollment_embedding is not None:
        face_db.add_face("Person_A", enrollment_embedding)
        print("Enrolled 'Person_A'")
    else:
        print("Enrollment image has no detected face.")

    test_image_path = "data/test.jpg"
    test_image = Image.open(test_image_path)
    test_embedding = recognizer.get_embedding(test_image)
    if test_embedding is not None:
        name, score = face_db.verify(test_embedding, threshold=0.8)
        if name is not None:
            print(f"Authentication succeeded: {name} (Similarity: {score:.2f})")
        else:
            print(f"Authentication failed: Max similarity {score:.2f}")
    else:
        print("Test image has no detected face.")

if __name__ == '__main__':
    main()
