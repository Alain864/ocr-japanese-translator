#!/usr/bin/env python3
"""
Japanese Image Translator - Main Entry Point

This script processes images containing Japanese text, detects the text,
translates it to English, and renders the translated text back onto the image.

Usage:
    python main.py                    # Process all images in input/ directory
    python main.py --image path.png   # Process a specific image
    python main.py --preview          # Generate detection preview only
    python main.py --help             # Show help message
"""

import argparse
import sys
from pathlib import Path

from config import INPUT_DIR, OUTPUT_DIR
from src import TranslationPipeline


def main():
    """Main entry point for the Japanese image translator."""
    parser = argparse.ArgumentParser(
        description="Translate Japanese text in images to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          Process all images in input/
    python main.py --image input/manga.png  Process a specific image
    python main.py --preview --image test.png  Show text detection preview
    python main.py --model gpt-4            Use a different OpenAI model
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to a specific image to process'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the translated image (only used with --image)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing images to process'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for translated images'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use for translation (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Generate a preview showing detected text regions (no translation)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           Japanese Image Translator (OCR + GPT)               ║
║                   日本語 → English                             ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    try:
        pipeline = TranslationPipeline(translation_model=args.model)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Handle preview mode
    if args.preview:
        if not args.image:
            print("Error: --preview requires --image to specify the image")
            sys.exit(1)
        
        print(f"Generating detection preview for: {args.image}")
        text_boxes = pipeline.get_detection_preview(args.image, args.output)
        print(f"\nDetected {len(text_boxes)} text regions:")
        for i, bbox in enumerate(text_boxes, 1):
            print(f"  {i}. '{bbox.text}' at ({bbox.min_x}, {bbox.min_y})")
        return
    
    # Process single image
    if args.image:
        results = pipeline.process_image(args.image, args.output)
        
        if results:
            print("\nTranslation Results:")
            print("-" * 40)
            for i, result in enumerate(results, 1):
                status = "✓" if result.success else "✗"
                print(f"{i}. {status} '{result.original_text[:30]}...'")
                print(f"      → '{result.translated_text[:50]}...'")
        return
    
    # Process directory
    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    all_results = pipeline.process_directory(str(input_dir), str(output_dir))
    
    if not all_results:
        print("\nNo images were processed. Check the input directory.")
        sys.exit(0)
    
    print("\n✓ All images processed successfully!")


if __name__ == "__main__":
    main()