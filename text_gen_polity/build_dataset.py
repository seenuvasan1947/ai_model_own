import os
import sys
from pypdf import PdfReader

# Update this to match your actual PDF filename
pdf_filename = "polity_book.pdf"  
txt_filename = "indian_polity.txt"

if not os.path.exists(pdf_filename):
    print(f"Error: '{pdf_filename}' not found!")
    print(f"Please place your PDF file in this folder and rename it to '{pdf_filename}'.")
    sys.exit(1)

print(f"📖 Opening '{pdf_filename}'...")
reader = PdfReader(pdf_filename)
total_pages = len(reader.pages)
print(f"Found {total_pages} pages. Extracting text...\n")

text_content = []
for i, page in enumerate(reader.pages):
    extracted = page.extract_text()
    if extracted:
        text_content.append(extracted)
    
    # Progress feedback
    print(f"\rProcessing page {i+1}/{total_pages} ({(i+1)/total_pages*100:.1f}%)", end="", flush=True)

full_text = "\n".join(text_content)

# Clean up redundant spaces and empty lines
cleaned_lines = [line.strip() for line in full_text.split("\n") if line.strip()]
final_text = "\n".join(cleaned_lines)

with open(txt_filename, "w", encoding="utf-8") as f:
    f.write(final_text)

file_size_mb = os.path.getsize(txt_filename) / (1024 * 1024)
print(f"\n\n🎉 Success! Extracted text saved to '{txt_filename}' ({file_size_mb:.2f} MB)")