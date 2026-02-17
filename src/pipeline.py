"""Main translation pipeline orchestrator."""

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import INPUT_DIR, OUTPUT_DIR
from .ocr import OCRProcessor, TextBoundingBox
from .translator import Translator
from .image_utils import ImageProcessor


@dataclass
class TranslationResult:
    """Result of processing a single text region."""
    original_text: str
    translated_text: str
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    success: bool
    error: Optional[str] = None


class TranslationPipeline:
    """
    Main pipeline for translating Japanese text in images.
    
    Coordinates OCR detection, translation, and image processing.
    """
    
    def __init__(self, translation_model: str = "gpt-4o-mini", cluster_eps: float = 60.0):
        """
        Initialize the translation pipeline.
        
        Args:
            translation_model: OpenAI model to use for translation
            cluster_eps: DBSCAN epsilon for clustering words into speech bubbles
        """
        self.ocr = OCRProcessor(cluster_eps=cluster_eps)
        self.translator = Translator(model=translation_model)
        self.image_processor = ImageProcessor()
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> List[TranslationResult]:
        """
        Process a single image: detect text, translate, and render.
        
        Args:
            image_path: Path to the input image
            output_path: Path for the output image (auto-generated if not provided)
            
        Returns:
            List of TranslationResult objects for each detected text region
        """
        print(f"\nProcessing: {image_path}")
        
        # Generate output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(OUTPUT_DIR / input_path.name)
        
        # Step 1: Detect and cluster Japanese text
        print("  [1/3] Detecting Japanese text...")
        start_time = time.time()
        
        try:
            text_boxes = self.ocr.detect_text(image_path)
            detection_time = time.time() - start_time
            print(f"       Found {len(text_boxes)} text regions in {detection_time:.2f}s")
        except Exception as e:
            print(f"       Error detecting text: {e}")
            return []
        
        if not text_boxes:
            print("       No text detected in image.")
            # Copy original image to output
            import shutil
            shutil.copy(image_path, output_path)
            return []
        
        # Step 2: Translate text
        print("  [2/3] Translating text to English...")
        start_time = time.time()
        
        # Extract texts for batch translation
        texts = [bbox.text for bbox in text_boxes]
        
        try:
            translations = self.translator.translate_batch(texts)
        except Exception as e:
            print(f"       Batch translation failed, falling back to individual: {e}")
            translations = [self.translator.translate(t) for t in texts]
        
        # Build results
        results = []
        regions_text = []
        
        for bbox, translation in zip(text_boxes, translations):
            print(f"       '{bbox.text[:30]}{'...' if len(bbox.text) > 30 else ''}' → '{translation[:40]}{'...' if len(translation) > 40 else ''}'")
            
            box = (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
            
            result = TranslationResult(
                original_text=bbox.text,
                translated_text=translation,
                bounding_box=box,
                success=True
            )
            results.append(result)
            regions_text.append((box, translation))
        
        translation_time = time.time() - start_time
        print(f"       Completed translations in {translation_time:.2f}s")
        
        # Step 3: Process image (inpaint + render text)
        print("  [3/3] Rendering translated text...")
        start_time = time.time()
        
        try:
            # Load original image
            image = self.image_processor.load_image(image_path)
            
            # Process image (inpaint and render with word wrapping)
            processed_image = self.image_processor.process_image(image, regions_text)
            
            # Save result
            self.image_processor.save_image(processed_image, output_path)
            processing_time = time.time() - start_time
            print(f"       Saved to: {output_path}")
            print(f"       Processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            print(f"       Error processing image: {e}")
            import shutil
            shutil.copy(image_path, output_path)
            return results
        
        print(f"       ✓ Done! Processed {len(results)} text regions")
        
        return results
    
    def process_directory(
        self, 
        input_dir: Optional[str] = None, 
        output_dir: Optional[str] = None,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.webp')
    ) -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path (defaults to INPUT_DIR)
            output_dir: Output directory path (defaults to OUTPUT_DIR)
            extensions: Tuple of file extensions to process
            
        Returns:
            Dictionary mapping input paths to their results
        """
        input_path = Path(input_dir) if input_dir else INPUT_DIR
        output_path = Path(output_dir) if output_dir else OUTPUT_DIR
        
        # Ensure output directory exists
        output_path.mkdir(exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        
        print(f"\n{'='*60}")
        print(f"Found {len(image_files)} images to process")
        print(f"{'='*60}")
        
        all_results = {}
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            
            output_file = output_path / image_file.name
            results = self.process_image(str(image_file), str(output_file))
            all_results[str(image_file)] = results
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        total_regions = sum(len(r) for r in all_results.values())
        successful = sum(1 for results in all_results.values() for r in results if r.success)
        print(f"Total images processed: {len(all_results)}")
        print(f"Total text regions: {total_regions}")
        print(f"Successfully translated: {successful}")
        print(f"Failed: {total_regions - successful}")
        print('='*60)
        
        return all_results