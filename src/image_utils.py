"""Image processing utilities for text inpainting and rendering."""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FONT_COLOR


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

    def _load_font(self, font_size: int) -> ImageFont.ImageFont:
        """
        Load the configured font at the requested size.
        
        Args:
            font_size: Font size in pixels
            
        Returns:
            Pillow font object
        """
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, font_size)
        except Exception:
            pass
        
        return ImageFont.load_default()

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int
    ) -> List[str]:
        """
        Wrap text into multiple lines that fit within a maximum width.
        
        Args:
            draw: Pillow drawing context
            text: Text to wrap
            font: Font used for measuring
            max_width: Maximum allowed line width
            
        Returns:
            Wrapped lines
        """
        words = text.split()
        if not words:
            return [text]
        
        lines = []
        current_line = ""

        def split_word(word: str) -> List[str]:
            """Split a single overlong word into chunks that fit the box width."""
            chunks = []
            current_chunk = ""
            
            for char in word:
                test_chunk = f"{current_chunk}{char}"
                bbox = draw.textbbox((0, 0), test_chunk, font=font)
                chunk_width = bbox[2] - bbox[0]
                
                if chunk_width <= max_width or not current_chunk:
                    current_chunk = test_chunk
                    continue
                
                chunks.append(current_chunk)
                current_chunk = char
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=font)
            word_width = bbox[2] - bbox[0]
            word_parts = split_word(word) if word_width > max_width else [word]
            
            for word_part in word_parts:
                test_line = f"{current_line} {word_part}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= max_width or not current_line:
                    current_line = test_line
                    continue
                
                lines.append(current_line)
                current_line = word_part
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def _measure_wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        lines: List[str],
        font: ImageFont.ImageFont,
        line_spacing: int
    ) -> Tuple[int, int]:
        """
        Measure the rendered size of wrapped text.
        
        Args:
            draw: Pillow drawing context
            lines: Wrapped text lines
            font: Font used for measuring
            line_spacing: Extra pixels between lines
            
        Returns:
            Tuple of (max_width, total_height)
        """
        if not lines:
            return 0, 0
        
        widths = []
        heights = []
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        
        total_height = sum(heights) + max(0, len(lines) - 1) * line_spacing
        return max(widths, default=0), total_height

    def _fit_text_to_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box_width: int,
        box_height: int
    ) -> Tuple[ImageFont.ImageFont, List[str], int]:
        """
        Find the largest font size that allows wrapped text to fit the box.
        
        Args:
            draw: Pillow drawing context
            text: Text to render
            box_width: Width of target box
            box_height: Height of target box
            
        Returns:
            Tuple of (font, wrapped_lines, line_spacing)
        """
        safe_width = max(1, box_width - 4)
        safe_height = max(1, box_height - 4)
        start_size = max(12, min(int(box_height * 0.35), 48))
        
        best_font = self._load_font(12)
        best_lines = [text]
        best_spacing = 2
        
        for font_size in range(start_size, 7, -1):
            font = self._load_font(font_size)
            line_spacing = max(1, font_size // 8)
            lines = self._wrap_text(draw, text, font, safe_width)
            text_width, text_height = self._measure_wrapped_text(
                draw, lines, font, line_spacing
            )
            
            if text_width <= safe_width and text_height <= safe_height:
                return font, lines, line_spacing
            
            best_font = font
            best_lines = lines
            best_spacing = line_spacing
        
        return best_font, best_lines, best_spacing
    
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
            font, lines, line_spacing = self._fit_text_to_box(
                draw, text, box_width, box_height
            )
            _, total_height = self._measure_wrapped_text(draw, lines, font, line_spacing)
            y = y1 + max(0, (box_height - total_height) // 2)
            
            # Draw each line
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center text horizontally
                x = x1 + max(0, (box_width - text_width) // 2)
                
                # Draw text in black
                draw.text((x, y), line, fill=FONT_COLOR, font=font)
                
                y += text_height + line_spacing
        
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
