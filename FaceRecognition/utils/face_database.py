import numpy as np
from .face_utils import cosine_similarity

class FaceDatabase:
    def __init__(self):
        self.db = {}

    def add_face(self, name: str, embedding: np.ndarray):
        self.db[name] = embedding

    def verify(self, embedding: np.ndarray, threshold: float = 0.8):
        best_match = None
        best_score = -1
        for name, db_embedding in self.db.items():
            score = cosine_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match = name
        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score
