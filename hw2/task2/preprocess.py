import os
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# Setup
pdf_path = "AI.pdf"
img_dir = "images"
ocr_output = "ocr.csv"
os.makedirs(img_dir, exist_ok=True)

# Read PDF
reader = PdfReader(pdf_path)
total_pages = len(reader.pages)

# Convert PDF pages to images in chunks
chunk_size = 15
for first_page in range(1, total_pages + 1, chunk_size):
    last_page = min(first_page + chunk_size - 1, total_pages)
    images = convert_from_path(
        pdf_path,
        dpi=150,
        first_page=first_page,
        last_page=last_page
    )
    for i, image in enumerate(images):
        page_num = first_page + i
        image.save(os.path.join(img_dir, f"{page_num}.png"), "PNG")

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False, gpu_mem=8096)

# Run OCR
results = []

image_files = sorted(
    (f for f in os.listdir(img_dir) if f.endswith(".png")),
    key=lambda x: int(os.path.splitext(x)[0])
)

for filename in tqdm(image_files, desc="Running OCR"):
    image_path = os.path.join(img_dir, filename)
    result = ocr.ocr(image_path, cls=True)

    lines = [line[1][0].strip() for line in result[0]] if result and result[0] else []

    # Extract important lines
    first_line = lines[0] if lines else ""
    cues = ["keyword", "definition", "example", "summary"]
    cue_lines = [l for l in lines if any(cue in l.lower() for cue in cues)]

    # Compose final text
    text = " ".join(cue_lines + [first_line] + lines)

    page_number = int(os.path.splitext(filename)[0])
    results.append((page_number, text))

# Save results
df = pd.DataFrame(results, columns=["Page", "Text"])
df.to_csv(ocr_output, index=False)

print(f"OCR extraction completed. Results saved to {ocr_output}")

# Next Step:
# Concatenate OCR results with Phi-4 Image Captions

ocr_df = df
caption_df = pd.read_csv('caption.csv')

# Fill missing Page values in caption_df with empty string
caption_df['Page'] = caption_df['Page'].fillna('')

# Merge the two dataframes on 'Page'
merged_df = pd.merge(ocr_df, caption_df, on='Page', how='left')

# Save or use the merged dataframe
merged_df.to_csv('merged_results.csv', index=False)
