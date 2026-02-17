"""OCR module for detecting Japanese text in images using Google Vision API."""

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

from google.cloud import vision
from google.oauth2 import service_account
import google.auth
import numpy as np
from sklearn.cluster import DBSCAN

from config import GOOGLE_CLOUD_API_KEY, MIN_CONFIDENCE


@dataclass
class TextBoundingBox:
    """Represents a detected text bounding box."""
    
    text: str
    vertices: List[Tuple[int, int]]  # List of (x, y) coordinates
    confidence: float
    
    @property
    def min_x(self) -> int:
        return min(v[0] for v in self.vertices)
    
    @property
    def min_y(self) -> int:
        return min(v[1] for v in self.vertices)
    
    @property
    def max_x(self) -> int:
        return max(v[0] for v in self.vertices)
    
    @property
    def max_y(self) -> int:
        return max(v[1] for v in self.vertices)
    
    @property
    def width(self) -> int:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> int:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)


class OCRProcessor:
    """Handles text detection using Google Cloud Vision API."""
    
    def __init__(self, cluster_eps: float = 60.0):
        """
        Initialize the OCR processor with Google Cloud Vision client.
        
        Args:
            cluster_eps: DBSCAN epsilon for clustering words into speech bubbles
        """
        self.client = self._create_client()
        self.cluster_eps = cluster_eps
    
    def _create_client(self) -> vision.ImageAnnotatorClient:
        """Create and configure the Vision API client."""
        if GOOGLE_CLOUD_API_KEY:
            from google.api_core.client_options import ClientOptions
            options = ClientOptions(api_key=GOOGLE_CLOUD_API_KEY)
            return vision.ImageAnnotatorClient(client_options=options)
        else:
            return vision.ImageAnnotatorClient()
    
    def _is_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters (Hiragana, Katakana, or Kanji)."""
        for char in text:
            code = ord(char)
            if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
                return True
        return False
    
    def get_japanese_words(self, image_path: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Extract Japanese words with their bounding boxes using document text detection.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of (text, vertices) tuples for each Japanese word
        """
        with open(image_path, "rb") as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        words = []
        
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    for word in para.words:
                        # Combine symbols into word text
                        text = "".join([symbol.text for symbol in word.symbols])
                        
                        # Filter for Japanese text
                        if self._is_japanese(text):
                            vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                            words.append((text, vertices))
        
        return words
    
    def cluster_words(self, words: List[Tuple[str, List[Tuple[int, int]]]]) -> List[List[Tuple[str, List[Tuple[int, int]]]]]:
        """
        Cluster word boxes into groups (e.g., speech bubbles) using DBSCAN.
        
        Args:
            words: List of (text, vertices) tuples
            
        Returns:
            List of word groups
        """
        if not words:
            return []
        
        # Calculate centers for each word box
        centers = []
        for _, vertices in words:
            xs = [p[0] for p in vertices]
            ys = [p[1] for p in vertices]
            centers.append([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2])
        
        centers = np.array(centers)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=1).fit(centers)
        
        # Group words by cluster label
        groups = {}
        for label, item in zip(clustering.labels_, words):
            groups.setdefault(label, []).append(item)
        
        return list(groups.values())
    
    def merge_boxes(self, group: List[Tuple[str, List[Tuple[int, int]]]]) -> Tuple[int, int, int, int]:
        """
        Merge multiple word boxes into a single bounding box.
        
        Args:
            group: List of (text, vertices) tuples in a cluster
            
        Returns:
            Merged bounding box as (x1, y1, x2, y2)
        """
        all_xs = []
        all_ys = []
        for _, vertices in group:
            all_xs.extend([p[0] for p in vertices])
            all_ys.extend([p[1] for p in vertices])
        
        return (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
    
    def detect_text(self, image_path: str) -> List[TextBoundingBox]:
        """
        Detect Japanese text in an image with clustering into speech bubbles.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of TextBoundingBox objects (clustered and merged)
        """
        # Get word-level detections
        words = self.get_japanese_words(image_path)
        
        if not words:
            return []
        
        # Cluster words into groups
        groups = self.cluster_words(words)
        
        # Create merged bounding boxes
        text_boxes = []
        for group in groups:
            # Combine text from all words in group
            combined_text = "".join([text for text, _ in group])
            
            # Merge bounding boxes
            x1, y1, x2, y2 = self.merge_boxes(group)
            vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            text_box = TextBoundingBox(
                text=combined_text,
                vertices=vertices,
                confidence=1.0
            )
            text_boxes.append(text_box)
        
        # Sort by y-coordinate for reading order
        text_boxes.sort(key=lambda b: b.min_y)
        
        return text_boxes
    
    def detect_text_from_bytes(self, image_bytes: bytes) -> List[TextBoundingBox]:
        """
        Detect text in image bytes (fallback method without clustering).
        
        Args:
            image_bytes: Image content as bytes
            
        Returns:
            List of TextBoundingBox objects
        """
        image = vision.Image(content=image_bytes)
        response = self.client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        text_boxes = []
        for annotation in response.text_annotations[1:]:
            text = annotation.description.strip()
            
            if self._is_japanese(text) and len(text) > 1:
                vertices = [(v.x, v.y) for v in annotation.bounding_poly.vertices]
                confidence = getattr(annotation, 'confidence', 1.0) or 1.0
                
                text_box = TextBoundingBox(
                    text=text,
                    vertices=vertices,
                    confidence=confidence
                )
                text_boxes.append(text_box)
        
        text_boxes.sort(key=lambda b: b.min_y)
        return text_boxes