import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from google.cloud import vision
from openai import OpenAI

INPUT_DIR = "input"
OUTPUT_DIR = "output"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

vision_client = vision.ImageAnnotatorClient()
llm = OpenAI()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# OCR WORD EXTRACTION
# -----------------------------
def get_japanese_words(path):
    with open(path, "rb") as f:
        img = vision.Image(content=f.read())

    resp = vision_client.document_text_detection(image=img)
    words = []

    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    text = "".join([s.text for s in word.symbols])

                    if any(ord(c) > 0x3000 for c in text):
                        box = [(v.x, v.y) for v in word.bounding_box.vertices]
                        words.append((text, box))

    return words


# -----------------------------
# BOX CLUSTERING (speech bubble)
# -----------------------------
def cluster_boxes(words):
    centers = []
    for _, box in words:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        centers.append([(min(xs)+max(xs))/2, (min(ys)+max(ys))/2])

    centers = np.array(centers)

    clustering = DBSCAN(eps=60, min_samples=1).fit(centers)
    groups = {}

    for label, item in zip(clustering.labels_, words):
        groups.setdefault(label, []).append(item)

    return list(groups.values())


# -----------------------------
# MERGE BOXES
# -----------------------------
def merge_boxes(group):
    xs, ys = [], []
    for _, box in group:
        xs += [p[0] for p in box]
        ys += [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


# -----------------------------
# TRANSLATION (context prompt)
# -----------------------------
def translate_sentence(text):
    prompt = f"""
Translate this Japanese manga dialogue to natural English.
Keep tone emotional and short.
Return only translation.

{text}
"""
    r = llm.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return r.output_text.strip()


# -----------------------------
# INPAINT MASK
# -----------------------------
def make_mask(img, regions):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (x1,y1,x2,y2) in regions:
        cv2.rectangle(mask, (x1-8,y1-8), (x2+8,y2+8), 255, -1)
    return mask


# -----------------------------
# DRAW TEXT
# -----------------------------
def draw_text(img, regions_text):
    draw = ImageDraw.Draw(img)

    for (x1,y1,x2,y2), text in regions_text:
        w = x2-x1
        h = y2-y1

        font_size = max(16, int(h*0.22))
        font = ImageFont.truetype(FONT_PATH, font_size)

        words = text.split()
        lines = []
        cur = ""

        for wword in words:
            test = cur+" "+wword if cur else wword
            if draw.textbbox((0,0), test, font=font)[2] < w:
                cur = test
            else:
                lines.append(cur)
                cur = wword
        if cur:
            lines.append(cur)

        total_h = len(lines)*(font_size+3)
        y = y1 + (h-total_h)//2

        for line in lines:
            tw = draw.textbbox((0,0), line, font=font)[2]
            x = x1 + (w-tw)//2
            draw.text((x,y), line, fill="black", font=font)
            y += font_size+3

    return img


# -----------------------------
# MAIN
# -----------------------------
def process(path):
    print("Processing:", path)

    words = get_japanese_words(path)
    groups = cluster_boxes(words)

    regions = []
    regions_text = []

    for g in groups:
        text = "".join([t for t,_ in g])
        box = merge_boxes(g)
        en = translate_sentence(text)

        regions.append(box)
        regions_text.append((box, en))

    img = cv2.imread(path)
    mask = make_mask(img, regions)
    clean = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    pil = draw_text(pil, regions_text)

    out = os.path.join(OUTPUT_DIR, os.path.basename(path))
    pil.save(out)
    print("Saved:", out)


for f in os.listdir(INPUT_DIR):
    if f.lower().endswith((".png",".jpg",".jpeg",".webp")):
        process(os.path.join(INPUT_DIR, f))
