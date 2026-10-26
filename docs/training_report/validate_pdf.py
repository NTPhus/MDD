"""Inspect and render the report PDF without executing any experiment."""
from pathlib import Path
import hashlib
import json
import re
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"_build_deps"))
import pymupdf
from PIL import Image, ImageDraw, ImageFont

pdf=HERE/"MDD_Bao_cao_huan_luyen_Wav2Vec2_Qwen.pdf"
doc=pymupdf.open(pdf)
qa=HERE/"qa"
qa.mkdir(exist_ok=True)
pages=[]
alltext=[]
images=set()
for i,page in enumerate(doc):
    text=page.get_text()
    alltext.append(text)
    for image in page.get_images(full=True):images.add(image[0])
    blocks=page.get_text("dict")["blocks"]
    bad=[]
    for block in blocks:
        if block["type"]!=0:continue
        x0,y0,x1,y1=block["bbox"]
        if x0<20 or x1>page.rect.width-20 or y0<5 or y1>page.rect.height-8:
            bad.append(block["bbox"])
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    pages.append({"page":i+1,"width":page.rect.width,"height":page.rect.height,
                  "characters":len(text),"image_count":len(page.get_images()),
                  "suspect_text_bounds":bad,"first_lines":lines[:5],
                  "last_lines":lines[-4:]})
    pix=page.get_pixmap(matrix=pymupdf.Matrix(0.8,0.8),alpha=False)
    pix.save(qa/f"page_{i+1:03d}.png")

joined="\n".join(alltext)
(qa/"pdf_text.txt").write_text(joined,encoding="utf-8")
captions=[int(x) for x in re.findall(r"Hình\s+(\d{2})\.",joined)]
reference_urls={item['url'] for item in json.loads((HERE.parent/'references_20261023.json').read_text(encoding='utf-8'))['references']}
external_links=[link.get('uri') for page in doc for link in page.get_links() if link.get('uri')]
result={"pdf":pdf.name,"sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages":len(doc),"unique_image_xrefs":len(images),
        "captions":captions,"missing_captions":sorted(set(range(1,36))-set(captions)),
        "replacement_characters":joined.count("\ufffd"),
        "suspect_text_bounds":sum(len(p["suspect_text_bounds"]) for p in pages),
        "outline_entries":len(doc.get_toc()),
        "additional_chart_labels":re.findall(r"Biểu đồ\s+(A\d+)\.",joined),
        "external_link_annotations":len(external_links),
        "missing_primary_reference_links":sorted(reference_urls-set(external_links)),
        "page_details":pages}
(qa/"pdf_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",17)
for start in range(0,len(doc),12):
    sheet=Image.new("RGB",(1200,1720),"#dbe3eb")
    draw=ImageDraw.Draw(sheet)
    for slot,i in enumerate(range(start,min(start+12,len(doc)))):
        pic=Image.open(qa/f"page_{i+1:03d}.png").convert("RGB")
        pic.thumbnail((280,385))
        col,row=slot%4,slot//4
        x=col*300+(300-pic.width)//2
        y=row*560+40
        sheet.paste(pic,(x,y))
        draw.text((col*300+12,row*560+8),f"Trang {i+1}",font=font,fill="#172c43")
    sheet.save(qa/f"pdf_contact_{start//12+1:02d}.png")
print(json.dumps({k:v for k,v in result.items() if k!="page_details"},ensure_ascii=False,indent=2))
print("\n".join(f"p{x['page']:02d}  chars={x['characters']:4d} images={x['image_count']}  "+" / ".join(x['first_lines'][2:5]) for x in pages))
