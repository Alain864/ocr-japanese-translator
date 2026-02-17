# Japanese Image Translator

A Python tool that automatically detects Japanese text in images, translates it to English, and replaces the original text with the translation while preserving the image's visual style.

## Features

- 🔍 **Smart OCR Detection**: Uses Google Cloud Vision Document Text Detection for word-level extraction with DBSCAN clustering to group words into speech bubbles
- 🌐 **AI Translation**: Leverages OpenAI GPT models with manga-optimized prompts for natural, emotional translations
- 🎨 **Smart Inpainting**: Removes original text and fills the area naturally using OpenCV inpainting
- 📝 **Word-Wrapped Text Rendering**: Automatically wraps translated text to fit within bounding boxes with dynamic font sizing
- 📁 **Batch Processing**: Process entire directories of images at once

## Project Structure

```
ocr-japanese-translator/
├── .env                    # API keys configuration
├── config.py               # Project configuration settings
├── main.py                 # Main entry point with CLI
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── ocr.py              # Google Vision OCR with DBSCAN clustering
│   ├── translator.py       # OpenAI translation module
│   ├── image_utils.py      # Image processing & text rendering
│   └── pipeline.py         # Main translation pipeline orchestrator
├── input/                  # Place images here for processing
│   └── *.png
└── output/                 # Translated images are saved here
    └── *.png
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.\.venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_CLOUD_API_KEY=your_google_cloud_api_key
```

#### Getting API Keys:

- **OpenAI API Key**: Go to [OpenAI API Keys](https://platform.openai.com/api-keys) to create one
- **Google Cloud API Key**: 
  1. Go to [Google Cloud Console](https://console.cloud.google.com/)
  2. Create a project and enable the Cloud Vision API
  3. Create an API key with Vision API access

## Usage

### Process All Images in Input Directory

```bash
python main.py
```

### Process a Single Image

```bash
python main.py --image input/manga.png
```

### Specify Output Path

```bash
python main.py --image input/manga.png --output output/translated.png
```

### Generate Detection Preview (No Translation)

```bash
python main.py --preview --image input/test.png
```

### Use a Different OpenAI Model

```bash
python main.py --model gpt-4
```

### Custom Input/Output Directories

```bash
python main.py --input-dir ./manga --output-dir ./translated
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--image` | `-i` | Path to a specific image to process |
| `--output` | `-o` | Output path for the translated image |
| `--input-dir` | | Input directory containing images |
| `--output-dir` | | Output directory for translated images |
| `--model` | `-m` | OpenAI model to use (default: gpt-4o-mini) |
| `--preview` | `-p` | Generate detection preview without translation |
| `--verbose` | `-v` | Enable verbose output |
| `--help` | `-h` | Show help message |

## How It Works

### 1. Text Detection & Clustering

The pipeline uses a sophisticated approach to detect and group text:

- **Document Text Detection**: Google Cloud Vision API extracts individual words with precise bounding boxes
- **Japanese Filtering**: Only text containing Japanese characters (Hiragana, Katakana, Kanji) is processed
- **DBSCAN Clustering**: Words are clustered into groups using distance-based clustering (eps=60 pixels) to identify speech bubbles and text blocks
- **Box Merging**: Each cluster's bounding boxes are merged into a single region for translation

### 2. Translation

Each detected text group is sent to OpenAI's GPT model with a specialized manga prompt:

```
Translate this Japanese manga dialogue to natural English.
Keep tone emotional and short.
Return only translation.
```

Batch translation is attempted first for context, with fallback to individual translations.

### 3. Image Processing

- **Inpainting**: Original text is removed using OpenCV's TELEA inpainting algorithm with 8px padding
- **Word Wrapping**: Translated text is automatically wrapped to fit within the bounding box width
- **Dynamic Font Sizing**: Font size is calculated based on box height (22% of height)
- **Centered Rendering**: Text is centered both horizontally and vertically within each region

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| `google-cloud-vision` | OCR and text detection |
| `openai` | GPT translation API |
| `opencv-python` | Image processing and inpainting |
| `Pillow` | Text rendering with fonts |
| `scikit-learn` | DBSCAN clustering algorithm |
| `numpy` | Array operations |
| `python-dotenv` | Environment variable management |

### Text Detection Pipeline

```
Image → Document Text Detection → Word Extraction
                                          ↓
                              Japanese Character Filter
                                          ↓
                              DBSCAN Clustering (eps=60)
                                          ↓
                              Bounding Box Merging
                                          ↓
                              Translation Groups
```

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- WebP (.webp)

## Troubleshooting

### "Could not load image"
- Ensure the image file exists and is not corrupted
- Check the file extension matches the actual format

### "Vision API error"
- Verify your Google Cloud API key is valid
- Ensure the Vision API is enabled in your Google Cloud project
- Check your API quota and billing

### "Translation error"
- Verify your OpenAI API key is valid
- Check your API credits/quota
- Try a different model with `--model gpt-4`

### Text not fitting in bounding box
- The system automatically wraps text and sizes fonts
- Very long translations may need manual adjustment
- Consider using a more concise translation model

### Single characters being translated oddly
- The system filters for Japanese text but single characters (like の, を, た) may produce odd translations
- This is expected behavior for isolated particles - they work better when clustered with surrounding text

## Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║           Japanese Image Translator (OCR + GPT)               ║
║                   日本語 → English                             ║
╚═══════════════════════════════════════════════════════════════╝

Processing: input/manga.png
  [1/3] Detecting Japanese text...
       Found 5 text regions in 0.86s
  [2/3] Translating text to English...
       'こんにちは' → 'Hello!'
       'ありがとう' → 'Thank you!'
       Completed translations in 2.34s
  [3/3] Rendering translated text...
       Saved to: output/manga.png
       Processing completed in 0.15s
       ✓ Done! Processed 5 text regions
```

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.