# src/rag/ingest_pdf.py
import os
import sys
import json
import base64
import pathlib
import argparse
import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from ollama import Client

# ----------------------------
# CONFIG & ENV
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "Documents")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")

client = Client(host=OLLAMA_HOST)

# ----------------------------
# IMAGE CAPTIONING
# ----------------------------
def caption_image(image_path: str, prompt="Describe this figure briefly for a report."):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = client.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_bytes]
        }]
    )
    return resp["message"]["content"]

# ----------------------------
# HELPER — Normalize Table
# ----------------------------
def table_to_markdown(table):
    if not table:
        return ""
    md_lines = []
    for i, row in enumerate(table):
        safe_row = [cell if cell is not None else "" for cell in row]
        md_lines.append("| " + " | ".join(safe_row) + " |")
        if i == 0:  # header separator
            md_lines.append("| " + " | ".join(["---"] * len(safe_row)) + " |")
    return "\n".join(md_lines)

# ----------------------------
# PDF PARSER (Reading Order)
# ----------------------------
def parse_pdf(pdf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    bundle = {"file_name": os.path.basename(pdf_path), "pages": []}

    # ---- Open with both PyMuPDF & pdfplumber ----
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    for page_index, page in enumerate(doc):
        plumber_page = plumber_doc.pages[page_index]
        page_elements = []

        # 1. TEXT BLOCKS (with coordinates)
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                text_content = []
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content.append(span["text"])
                full_text = " ".join(text_content).strip()
                if full_text:
                    page_elements.append({
                        "y": block["bbox"][1],
                        "type": "text",
                        "content": full_text
                    })

        # 2. TABLES (coordinates from pdfplumber)
        tables = plumber_page.find_tables()
        for t_idx, table in enumerate(tables):
            table_data = table.extract()
            csv_path = os.path.join(tables_dir, f"page{page_index+1}_table{t_idx+1}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                for row in table_data:
                    f.write(",".join([cell if cell else "" for cell in row]) + "\n")

            md_table = table_to_markdown(table_data)
            page_elements.append({
                "y": table.bbox[1],
                "type": "table",
                "content": md_table,
                "path": csv_path
            })

        # 3. IMAGES (coordinates + captions)
        img_list = page.get_images(full=True)
        for img_idx, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_path = os.path.join(images_dir, f"page{page_index+1}_img{img_idx+1}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            caption = caption_image(img_path)
            # Estimate Y position (PyMuPDF stores in page.get_images? no direct)
            y_pos = 0
            try:
                y_pos = page.get_image_bbox(img).y0
            except:
                pass
            page_elements.append({
                "y": y_pos,
                "type": "image",
                "content": f"![Image]({img_path})\n\n*Caption:* {caption}",
                "path": img_path
            })

        # Sort by Y position (top to bottom)
        page_elements.sort(key=lambda x: x["y"])

        # Add to bundle
        bundle["pages"].append({
            "page_number": page_index + 1,
            "elements": page_elements
        })

    plumber_doc.close()

    # ---- Save JSON ----
    json_path = os.path.join(output_dir, "bundle.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    # ---- Save Markdown ----
    md_path = os.path.join(output_dir, "bundle.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {bundle['file_name']}\n\n")
        for p in bundle["pages"]:
            f.write(f"## Page {p['page_number']}\n\n")
            for el in p["elements"]:
                f.write(el["content"] + "\n\n")

    print(f"✅ Parsed {pdf_path} -> {output_dir}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recursive", action="store_true", help="Recursively search for PDFs in Documents/")
    args = parser.parse_args()

    search_pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files = list(pathlib.Path(DOCUMENTS_DIR).glob(search_pattern))
    if not pdf_files:
        print(f"❌ No PDFs found in {DOCUMENTS_DIR}")
        sys.exit(1)

    for pdf in pdf_files:
        pdf_name = pathlib.Path(pdf).stem
        output_dir = os.path.join(OUTPUT_ROOT, pdf_name)
        parse_pdf(str(pdf), output_dir)

    print("🎯 All PDFs processed.")
