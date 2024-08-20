import os
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
dependencies = [
    "minio",
    "pymupdf"
]

for dep in dependencies:
    install(dep)
    
import fitz  # PyMuPDF
import re
from minio import Minio
from minio.error import S3Error
    

#Remove Watermark
def remove_watermark_advanced(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    
    for page in doc:
        # Iterate over all images in the page
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            # Remove the image (possible watermark)
            page.delete_image(xref)
        
        # Iterate over all text annotations (potential text-based watermarks)
        annots = page.annots()
        if annots:
            for annot in annots:
                annot_info = annot.info
                if "Watermark" in annot_info.get("title", ""):
                    annot.set_flags(fitz.ANNOT_HIDDEN)

        # If there are rectangles or other objects you suspect to be watermarks,
        # you might need to identify them and remove them programmatically here.
        # But we'll skip the empty search that caused the error.

        page.apply_redactions()

    doc.save(output_path)
    print(f"Watermark removed: {output_path}")

# List of PDFs to process
pdf_files = [
    "./2024-bernardo.pdf"
]

# Process each PDF
for pdf in pdf_files:
    output_pdf = pdf.replace(".pdf", "_no_watermark_advanced.pdf")
    remove_watermark_advanced(pdf, output_pdf)
    print(f"Processed: {output_pdf}")