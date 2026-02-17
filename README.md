# Japanese Image Translator

A Python tool that automatically detects Japanese text in images, translates it to English, and replaces the original text with the translation while preserving the image's visual style.

## Features

- 🔍 **OCR Detection**: Uses Google Cloud Vision API to accurately detect Japanese text and its bounding boxes
- 🌐 **AI Translation**: Leverages OpenAI GPT models for natural, context-aware translations
- 🎨 **Smart Inpainting**: Removes original text and fills the area naturally using OpenCV inpainting
- 📝 **Text Rendering**: Places translated text in the original text's location with appropriate sizing
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
│   ├── ocr.py              # Google Vision OCR module
│   ├── translator.py       # OpenAI translation module
│   ├── image_utils.py      # Image processing utilities
│   └── pipeline.py         # Main translation pipeline
├── input/                  # Place images here for processing
│   └── *.png
└── output/                 # Translated images are saved here
    └── *_translated.png
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

1. **Text Detection**: The pipeline uses Google Cloud Vision API to detect all Japanese text in the image, along with precise bounding box coordinates for each text region.

2. **Translation**: Each detected Japanese text is sent to OpenAI's GPT model with a specialized prompt for manga/Japanese content translation.

3. **Image Inpainting**: Original text is removed using OpenCV's inpainting algorithm, which intelligently fills in the removed area based on surrounding pixels.

4. **Text Rendering**: Translated English text is rendered in the original text's location, with automatic font sizing to fit within the bounding box.

## Configuration

Edit `config.py` to customize:

```python
# Translation settings
TARGET_LANGUAGE = "English"
SOURCE_LANGUAGE = "Japanese"

# Font settings
DEFAULT_FONT_SIZE = 20
FONT_COLOR = (0, 0, 0)  # Black text

# Image processing
MIN_CONFIDENCE = 0.5     # Minimum OCR confidence
PADDING_RATIO = 0.1     # Padding around text boxes
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
- Adjust `DEFAULT_FONT_SIZE` in config.py
- The system will automatically try to fit text, but very long translations may need manual adjustment

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.