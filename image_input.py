from pathlib import Path
import os
import shutil
from PIL import Image
import pytesseract

# On Windows, set TESSERACT_CMD if Tesseract is installed in a nonstandard path.
# On Linux hosts, install the tesseract-ocr system package and let PATH locate it.
_tesseract_cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def extract_image_text(file_path=None):
    if file_path is None:
        file_path = input("Enter image path: ").strip().strip('"')
    path = Path(file_path)
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print("Please provide an existing supported image file.")
        return None
    try:
        with Image.open(path) as image:
            # Try English first; Hindi requires the corresponding traineddata to be installed.
            try:
                text = pytesseract.image_to_string(image, lang="eng+hin").strip()
            except pytesseract.TesseractError:
                text = pytesseract.image_to_string(image, lang="eng").strip()
        if not text:
            print("No readable text found in image.")
            return None
        return text
    except Exception as error:
        print(f"Image extraction failed: {error}")
        return None

if __name__ == "__main__":
    text = extract_image_text()
    if text:
        print("\n--- Extracted Image Text ---\n" + text)
