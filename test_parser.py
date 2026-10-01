import fitz

pdf_path = "sample_resume.pdf"

document = fitz.open(pdf_path)

print("Number of pages:", len(document))

for page in document:
    text = page.get_text()
    print("\n--- PAGE ---")
    print(text)

document.close()