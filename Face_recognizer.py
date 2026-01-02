import numpy as np
from typing import List, Tuple, Dict, Optional
from Database import db


class FaceRecognizer:
    """
    Core facial recognition engine implementing biometric vector comparison.
    Handles template management and similarity calculations using Cosine Similarity.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize the recognizer with a similarity threshold.

        Args:
            threshold: Minimum cosine similarity score (default 0.7).
        """
        self.threshold = threshold
        self.templates = []  # In-memory biometric template cache
        self._load_templates()

        # Hooks for preprocessing and feature extraction modules
        self.preprocess_func = None
        self.extract_feature_func = None

    def _load_templates(self):
        """Loads all biometric templates from the persistent JSON database."""
        self.templates = db.get_all_templates()
        print(f"[INFO] Biometric templates loaded: {len(self.templates)}")

    def refresh_templates(self):
        """Refreshes the internal template cache, typically called after new registration."""
        self._load_templates()
        print("[INFO] Biometric cache synchronized with database.")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the Cosine Similarity between two feature vectors.
        This represents the core comparison algorithm.

        Args:
            vec1: Query embedding vector.
            vec2: Template embedding vector.

        Returns:
            float: Similarity score ranging from [-1.0, 1.0].
        """
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        # Vector dot product
        dot_product = np.dot(v1, v2)

        # L2 Norm (Magnitude)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Divide-by-zero prevention for zero-magnitude vectors
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Clamping result to valid range
        return max(-1.0, min(1.0, similarity))

    def identify(self, query_vector: List[float]) -> Tuple[str, float, bool, Dict]:
        """
        Performs 1:N biometric identification against the loaded database.

        Args:
            query_vector: 512-D feature vector extracted from the query image.

        Returns:
            tuple: (EmployeeID, SimilarityScore, SuccessFlag, EmployeeData)
        """
        if not self.templates:
            return "Unknown", 0.0, False, {}

        best_match = None
        best_similarity = -1.0

        # Iterative linear search for the highest similarity match
        for template in self.templates:
            template_vector = template.get("embedding_vector", [])

            if len(template_vector) != len(query_vector):
                continue

            similarity = self.cosine_similarity(query_vector, template_vector)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = template

        # Decision logic based on pre-defined threshold
        recognized = best_similarity >= self.threshold

        if recognized and best_match:
            emp_id = best_match.get("emp_id", "Unknown")
            # Retrieve detailed metadata from the employee database
            employee_info = db.get_employee_by_id(emp_id) or {}
            return emp_id, best_similarity, True, employee_info
        else:
            return "Unknown", best_similarity, False, {}

    def search_top_k(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        Searches for the Top K most similar identities in the database.

        Args:
            query_vector: Target embedding vector.
            k: Number of candidates to return.

        Returns:
            list: Ranked list of similarity results.
        """
        if not self.templates:
            return []

        results = []
        for template in self.templates:
            template_vector = template.get("embedding_vector", [])

            if len(template_vector) != len(query_vector):
                continue

            similarity = self.cosine_similarity(query_vector, template_vector)

            results.append({
                "emp_id": template.get("emp_id"),
                "similarity": similarity,
                "template_id": template.get("template_id")
            })

        # Rank results by similarity score in descending order
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def set_preprocess_function(self, func):
        """Configures the hook for image preprocessing (Facial Alignment/Cropping)."""
        self.preprocess_func = func
        print("[INFO] Preprocessing hook established.")

    def set_extract_feature_function(self, func):
        """Configures the hook for feature extraction (Deep Neural Inference)."""
        self.extract_feature_func = func
        print("[INFO] Feature extraction hook established.")

    def process_image(self, image_data) -> Tuple[str, float, bool, Dict]:
        """
        Executes the full biometric pipeline: Preprocessing -> Inference -> Matching.

        Args:
            image_data: Raw input image (numpy array or base64 string).

        Returns:
            tuple: Final identification results.
        """
        if self.preprocess_func is None:
            raise ValueError("Dependency Error: Preprocessing hook is missing.")
        if self.extract_feature_func is None:
            raise ValueError("Dependency Error: Feature extraction hook is missing.")

        # Step 1: Detect and align faces
        processed_faces = self.preprocess_func(image_data)

        if not processed_faces:
            return "Unknown", 0.0, False, {}

        # Step 2: Compute feature embeddings (typically 512-D vectors)
        face_vectors = []
        for face in processed_faces:
            vector = self.extract_feature_func(face)
            face_vectors.append(vector)

        # Step 3: Biometric identification (Matching against database)
        if face_vectors:
            return self.identify(face_vectors[0])

        return "Unknown", 0.0, False, {}


# Instantiate global recognizer with tuned confidence threshold
recognizer = FaceRecognizer(threshold=0.7)