"""Image processing utilities for text inpainting and rendering."""

import os
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FONT_COLOR, PADDING_RATIO, BASE_DIR
from .ocr import TextBoundingBox


class ImageProcessor:
    """Handles image processing operations for text replacement."""
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize the image processor.
        
        Args:
            font_path: Optional path to font file
        """
        self.font_path = font_path or self._find_font()
    
    def _find_font(self) -> Optional[str]:
        """Find a suitable font for rendering text."""
        # Common font paths on macOS/Linux
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        
        return None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array (BGR format)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def make_mask(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        padding: int = 8
    ) -> np.ndarray:
        """
        Create a combined mask for all text regions.
        
        Args:
            image: Input image as numpy array
            regions: List of (x1, y1, x2, y2) bounding boxes
            padding: Padding in pixels around each box
            
        Returns:
            Mask as numpy array
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for (x1, y1, x2, y2) in regions:
            cv2.rectangle(mask, (x1 - padding, y1 - padding), (x2 + padding, y2 + padding), 255, -1)
        
        return mask
    
    def inpaint_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        padding: int = 8
    ) -> np.ndarray:
        """
        Inpaint (remove) text from multiple regions in the image.
        
        Args:
            image: Input image as numpy array
            regions: List of (x1, y1, x2, y2) bounding boxes
            padding: Padding in pixels around each box
            
        Returns:
            Image with text removed
        """
        mask = self.make_mask(image, regions, padding)
        
        # Apply inpainting with TELEA method
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted
    
    def draw_text_with_wrapping(
        self,
        pil_image: Image.Image,
        regions_text: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> Image.Image:
        """
        Draw text with word wrapping fitted to bounding boxes.
        
        Args:
            pil_image: PIL Image to draw on
            regions_text: List of ((x1, y1, x2, y2), text) tuples
            
        Returns:
            PIL Image with rendered text
        """
        draw = ImageDraw.Draw(pil_image)
        
        for (x1, y1, x2, y2), text in regions_text:
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate font size based on box height
            font_size = max(16, int(box_height * 0.22))
            
            # Load font
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            # Word wrap the text
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
                
                if text_width < box_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Calculate total text height
            total_height = len(lines) * (font_size + 3)
            
            # Center text vertically
            y = y1 + (box_height - total_height) // 2
            
            # Draw each line
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                
                # Center text horizontally
                x = x1 + (box_width - text_width) // 2
                
                # Draw text in black
                draw.text((x, y), line, fill="black", font=font)
                
                y += font_size + 3
        
        return pil_image
    
    def process_image(
        self,
        image: np.ndarray,
        regions_text: List[Tuple[Tuple[int, int, int, int], str]]
    ) -> np.ndarray:
        """
        Process an image by inpainting and rendering translated text.
        
        Args:
            image: Input image as numpy array (BGR)
            regions_text: List of ((x1, y1, x2, y2), translated_text) tuples
            
        Returns:
            Processed image with translated text (BGR)
        """
        if not regions_text:
            return image
        
        # Extract regions for inpainting
        regions = [box for box, _ in regions_text]
        
        # Inpaint all regions
        clean = self.inpaint_regions(image, regions, padding=8)
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        
        # Draw text with wrapping
        pil_image = self.draw_text_with_wrapping(pil_image, regions_text)
        
        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return result
    
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save an image to file.
        
        Args:
            image: Image as numpy array
            output_path: Path to save the image
        """
        cv2.imwrite(output_path, image)