# OCR Japanese Translator - AI Coding Guidelines

## Project Overview

This tool automatically detects Japanese text in images, translates it to English, and replaces the original text while preserving visual style. The pipeline coordinates three main tasks: **text detection via clustering**, **translation via LLM**, and **smart text removal and replacement** through inpainting and rendering.

## Code Style

- **Language**: Python 3.8+, PEP 8 compliant
- **Type Hints**: Use throughout for function signatures and complex operations
- **Dataclasses**: Prefer for data containers (e.g., [TextBoundingBox](src/ocr.py#L14) in `ocr.py`)
- **NumPy/OpenCV**: Use numpy arrays for image data (BGR format); OpenCV for image ops
- **Error Handling**: Graceful fallbacks (e.g., return original text if translation fails in [translator.py](src/translator.py#L48))

## Architecture

Three independent modules orchestrated via [pipeline.py](src/pipeline.py):

1. **OCR Detection** ([ocr.py](src/ocr.py)): 
   - Google Cloud Vision API extracts words with bounding boxes
   - **DBSCAN Clustering** (eps=60 pixels default) groups spatially-near words into speech bubbles
   - Japanese character filtering validates text (Hiragana: `0x3040–0x309F`, Katakana: `0x30A0–0x30FF`, Kanji: `0x4E00–0x9FFF`)

2. **Translation** ([translator.py](src/translator.py)):
   - OpenAI GPT models with manga-optimized prompts (temperature=0.3 for consistency)
   - Batch translation sends multiple texts with context labels
   - Fallback to original text on API errors

3. **Image Processing** ([image_utils.py](src/image_utils.py)):
   - **MASK**: Binary mask creation for all text regions with configurable padding
   - **INPAINTING**: OpenCV morphological operations to naturally fill masked areas
   - **TYPESETTING**: Word-wrapped text rendering with dynamic font sizing to fit bounding boxes

## Build and Test

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key"
export GOOGLE_CLOUD_API_KEY="your_key"  # or use Application Default Credentials

# Process single image with detection preview
python main.py --image input/sample.png --preview

# Full translation (OCR → Translate → Inpaint → Render)
python main.py --image input/sample.png --output output/translated.png

# Batch process directory
python main.py --input-dir ./images --output-dir ./translated

# Custom model
python main.py --image input/sample.png --model gpt-4
```

## Project Conventions

### CLUSTERING Strategy
- DBSCAN parameters in [ocr.py](src/ocr.py#L54): `eps=60` (word distance threshold)
- Centers of word bounding boxes determine cluster membership
- Isolated words become single-element clusters

### MERGING Bounding Boxes
- After clustering, each cluster's boxes merged via `min/max` vertices in [ocr.py](src/ocr.py#L155)
- Result: one box per text region (speech bubble)

### MASK Creation
- Binary 8-bit grayscale, 255 for text regions in [image_utils.py](src/image_utils.py#L65)
- Rectangular padding added via `PADDING_RATIO` config (10% default)
- Used directly by inpainting algorithm

### INPAINTING Approach
- OpenCV's morphological closing + median blur for natural fill
- Kernel size 5×5, closure iterations=2 in [image_utils.py](src/image_utils.py#L91)
- Avoids sharp artifacts in clean backgrounds

### TYPESETTING System
- PIL ImageFont for CPU-based rendering
- Binary search for max font size that fits box width/height in [image_utils.py](src/image_utils.py#L115)
- Text centered both horizontally and vertically
- `FONT_COLOR` always (0, 0, 0) black; adjust in [config.py](config.py#L27) if needed

## Integration Points

- **Google Cloud Vision**: Requires API key or Application Default Credentials; uses `document_text_detection` method
- **OpenAI API**: `chat.completions.create` with configurable model (default: gpt-4o-mini)
- **File I/O**: Input images from [INPUT_DIR](config.py#L10), output to [OUTPUT_DIR](config.py#L11)
- **CLI**: [main.py](main.py) provides argument parsing; delegates to [TranslationPipeline](src/pipeline.py#L28)

## Security

- API keys managed via `.env` file (loaded in [config.py](config.py#L8) via `python-dotenv`)
- **Never commit `.env`** — add to `.gitignore`
- Google Cloud credentials can use Application Default Credentials for GCP environments
- Translation requests contain no sensitive data beyond user images

## Key Files Reference

- [pipeline.py](src/pipeline.py) — Orchestrator: calls OCR → Translator → ImageProcessor in sequence
- [ocr.py](src/ocr.py) — DBSCAN clustering logic and Japanese text filtering
- [translator.py](src/translator.py) — OpenAI integration with fallback handling
- [image_utils.py](src/image_utils.py) — Mask, inpaint, and typesetting implementations
- [config.py](config.py) — Centralized settings: paths, API keys, image parameters
