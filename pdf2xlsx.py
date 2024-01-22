import pdfplumber
pdf = pdfplumber.open("UNECE 贸易便利化术语 TRADE Facilitation Terms v3 cn en ru.pdf")
page = pdf.pages[29]
print(page.chars)
for i in page.chars:
    print(i["fontname"],i["text"])

for i in page.lines:
    print(i)
print(page.width,page.height)
page = page.crop((272,0,419,595))
im = page.to_image(resolution=150)
im.draw_rects(page.extract_words())
im
text = page.extract_text()
print(text)
text = page.extract_table(table_settings={"vertical_strategy":"text"})
print(text)