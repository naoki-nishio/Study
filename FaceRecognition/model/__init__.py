import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

class FaceRecognizer:
    def __init__(self, device):
        self.device = device
        self.yolo = YOLO('yolov8n-face.pt')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, image: Image.Image):
        results = self.yolo(image)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        box = results[0].boxes[0].xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, box)
        face_crop = image.crop((x1, y1, x2, y2))
        face_crop = face_crop.resize((160, 160))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(face_tensor)
        return embedding[0].cpu().numpy()
