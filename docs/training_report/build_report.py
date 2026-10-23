"""Build the Vietnamese MDD training report from a reviewable JSON source.

Only reads saved experiment evidence and extracted notebook images. No models,
notebook cells or training scripts are executed.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "_build_deps"))

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, NextPageTemplate, CondPageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents

rl_config.warnOnMissingFontGlyphs = 1
for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"),
                   ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

INK = colors.HexColor("#172C43")
TEAL = colors.HexColor("#087E8B")
MUTED = colors.HexColor("#526479")
PALE = colors.HexColor("#EAF4F5")
LINE = colors.HexColor("#D4E0E8")
AMBER = colors.HexColor("#8C5415")
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
WIDTH = PAGE_W - 2 * MARGIN

styles = {
    "body": ParagraphStyle("Body", fontName="Arial", fontSize=9.8, leading=15,
        textColor=INK, spaceAfter=8, allowWidows=0, allowOrphans=0),
    "small": ParagraphStyle("Small", fontName="Arial", fontSize=8, leading=11.5,
        textColor=MUTED, spaceAfter=6, wordWrap="CJK"),
    "source": ParagraphStyle("Source", fontName="Arial", fontSize=7.1, leading=10,
        textColor=MUTED, spaceAfter=6, wordWrap="CJK"),
    "h1": ParagraphStyle("H1", fontName="Arial-Bold", fontSize=20, leading=26,
        textColor=INK, spaceAfter=15, keepWithNext=True),
    "h2": ParagraphStyle("H2", fontName="Arial-Bold", fontSize=12.5, leading=18,
        textColor=TEAL, spaceBefore=12, spaceAfter=8, keepWithNext=True),
    "h3": ParagraphStyle("H3", fontName="Arial-Bold", fontSize=10.5, leading=15,
        textColor=INK, spaceBefore=8, spaceAfter=5, keepWithNext=True),
    "caption": ParagraphStyle("Caption", fontName="Arial", fontSize=8.4, leading=12,
        textColor=INK, spaceBefore=5, spaceAfter=3),
    "table": ParagraphStyle("Cell", fontName="Arial", fontSize=8.1, leading=11.7,
        textColor=INK, wordWrap="CJK"),
    "th": ParagraphStyle("HeaderCell", fontName="Arial-Bold", fontSize=8.1, leading=11.7,
        textColor=colors.white, wordWrap="CJK"),
    "code": ParagraphStyle("Code", fontName="Arial", fontSize=8, leading=12,
        textColor=INK, backColor=colors.HexColor("#F0F3F7"), borderPadding=9,
        spaceAfter=9, wordWrap="CJK"),
    "bullet": ParagraphStyle("Bullet", fontName="Arial", fontSize=9.8, leading=15,
        textColor=INK, leftIndent=12, firstLineIndent=-10, spaceAfter=5),
}


def clean(text):
    return html.escape(str(text)).replace("\n", "<br/>")


def rich(text):
    # The source is controlled report prose. Support a small, explicit subset.
    t = clean(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font color="#087E8B">\1</font>', t)
    return t


def para(text, style="body"):
    return Paragraph(rich(text), styles[style])


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=21*mm, bottomMargin=19*mm, **kwargs)
        self.heading_entries = []
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[Frame(MARGIN, 19*mm, WIDTH, PAGE_H-38*mm,
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.cover_bg),
            PageTemplate(id="body", frames=[Frame(MARGIN, 19*mm, WIDTH, PAGE_H-40*mm,
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.page_bg),
            PageTemplate(id="wide", pagesize=landscape(A4), frames=[Frame(MARGIN,19*mm,
                PAGE_H-2*MARGIN,PAGE_W-40*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)], onPage=self.page_bg),
        ])

    def cover_bg(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(INK)
        canvas.rect(0, PAGE_H-18*mm, PAGE_W, 18*mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(MARGIN, 21*mm, WIDTH, 1.2*mm, fill=1, stroke=0)
        canvas.restoreState()

    def page_bg(self, canvas, doc):
        w, h = canvas._pagesize
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN,h-15*mm,w-MARGIN,h-15*mm)
        canvas.setFont("Arial-Bold",7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(MARGIN,h-12*mm,"MDD  /  wav2vec2 & Qwen")
        canvas.setFont("Arial",7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(w-MARGIN,h-12*mm,"Báo cáo thực nghiệm • 23/10/2026")
        canvas.line(MARGIN,14*mm,w-MARGIN,14*mm)
        canvas.drawString(MARGIN,10*mm,"Nguồn: notebook, cấu hình và artifact cục bộ của dự án MDD")
        canvas.drawRightString(w-MARGIN,10*mm,str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable,"report_heading"):
            level, title, key = flowable.report_heading
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=level, closed=False)
            self.notify("TOCEntry", (level,title,self.page,key))


def table(block, width=WIDTH):
    header = [para(x,"th") for x in block["headers"]]
    rows = [[para("—" if x is None else str(x),"table") for x in row] for row in block["rows"]]
    ratios = block.get("widths",[1]*len(header))
    widths = [width*x/sum(ratios) for x in ratios]
    t = Table([header]+rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),INK),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),7),
        ("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F2F6F9")]),
        ("LINEBELOW",(0,0),(-1,0),.6,INK),
        ("LINEBELOW",(0,1),(-1,-1),.3,LINE),
    ]))
    return t


def main():
    source = json.loads((HERE/"report_content.json").read_text(encoding="utf-8"))
    figures = json.loads((HERE/"figures_manifest.json").read_text(encoding="utf-8"))["figures"]
    figmap = {x["figure_id"]:x for x in figures}
    out = HERE / "MDD_Bao_cao_huan_luyen_Wav2Vec2_Qwen.pdf"
    doc = ReportDoc(str(out), title="MDD – Báo cáo huấn luyện wav2vec2 và Qwen",
        author="Dự án MDD", subject="Quy trình, thông số, kết quả và hình ảnh notebook")
    story = [Spacer(1,26*mm)]
    cover_label=ParagraphStyle("CoverLabel",fontName="Arial-Bold",fontSize=11,leading=17,textColor=TEAL,spaceAfter=18)
    cover_title=ParagraphStyle("CoverTitle",fontName="Arial-Bold",fontSize=32,leading=42,textColor=INK,spaceAfter=20)
    cover_sub=ParagraphStyle("CoverSub",fontName="Arial",fontSize=15,leading=23,textColor=MUTED,spaceAfter=18)
    story += [Paragraph("BÁO CÁO TỔNG HỢP THỰC NGHIỆM",cover_label),
        Paragraph("Huấn luyện wav2vec2<br/>và Qwen trong MDD",cover_title),
        Paragraph("Quy trình · Cấu hình · Kết quả<br/>Đánh giá và đối chiếu bằng chứng",cover_sub),
        Spacer(1,10*mm),para("**MDD – Mispronunciation Detection and Diagnosis**\nPhát hiện và chẩn đoán lỗi phát âm tiếng Anh cho người Việt."),
        Spacer(1,7*mm), table({"headers":["Phạm vi","Tài liệu"],"rows":[
            ["Mô hình","wav2vec2-base, wav2vec2-base-960h, Qwen3-0.6B và Projector"],
            ["Bằng chứng","14 notebook; 35 hình gốc; log, checkpoint metadata và metrics đã lưu"],
            ["Ngày tổng hợp","23/10/2026 · Giờ Bangkok (UTC+7)"],
            ["Phiên bản","1.0 · Tổng hợp trạng thái cục bộ, không chạy lại huấn luyện"],
        ],"widths":[1,3]}), Spacer(1,10*mm),
        para("Số liệu của từng protocol được trình bày riêng. Ảnh được lấy trực tiếp từ output đã lưu của notebook; các ảnh chưa cập nhật được chú thích rõ.","small"),
        NextPageTemplate("body"),PageBreak(),para("Mục lục","h1")]
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle("TOC0",fontName="Arial-Bold",fontSize=10,leading=15,spaceBefore=8,textColor=INK,rightIndent=20),
        ParagraphStyle("TOC1",fontName="Arial",fontSize=9,leading=13,leftIndent=14,textColor=MUTED)]
    story += [toc,Spacer(1,9*mm),para("Quy ước đọc tài liệu","h2"),
        para("Các tỷ lệ được đổi sang %; chênh lệch giữa hai tỷ lệ dùng **điểm phần trăm (đpt)**. Loss/NLL, perplexity, BLEU và ROUGE giữ đúng thang ghi trong nguồn. Dấu — là không có hoặc không xác định, không phải 0."),
        para("Nguồn `[Sxx]` được giải thích ở danh mục cuối tài liệu. Cell được đánh số từ 1 theo vị trí trong file notebook, không theo execution count. Mỗi hình có thêm output index để truy ngược chính xác."),
        para("Ảnh notebook có thể phản ánh thời điểm lưu cũ. Nếu có khác biệt, bảng phân tích dùng artifact thực tế mới hơn và chỉ rõ khác biệt. Không xem mẫu smoke, pilot hay validation như kết quả test toàn bộ.")]
    used_figures=[]
    for ci,chapter in enumerate(source["chapters"],1):
        story.append(PageBreak())
        heading=para(chapter["title"],"h1")
        heading.report_heading=(0,chapter["title"],f"chapter-{ci}")
        story.append(heading)
        for bi,b in enumerate(chapter["blocks"]):
            kind=b["type"]
            if kind in ("p","small","source","code"):
                story.append(para(b["text"], {"p":"body"}.get(kind,kind)))
            elif kind in ("h2","h3"):
                story.append(para(b["text"],kind))
            elif kind=="bullets":
                story.extend(para("• "+x,"bullet") for x in b["items"])
                story.append(Spacer(1,4))
            elif kind=="table":
                story.extend([table(b),Spacer(1,8)])
                if b.get("source"):story.append(para(b["source"],"source"))
            elif kind=="note":
                box=Table([[para(b["text"])]],colWidths=[WIDTH])
                box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE),
                    ("BOX",(0,0),(-1,-1),.6,TEAL),
                    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
                story += [box,Spacer(1,10)]
            elif kind=="pagebreak":story.append(PageBreak())
            elif kind=="figure":
                f=figmap[b["id"]]
                used_figures.append(b["id"])
                dims=f["dimensions"]
                w=WIDTH
                maxh=b.get("max_height_mm",100)*mm
                scale=min(w/dims["width"],maxh/dims["height"])
                img=Image(str(HERE/f["file"]),width=dims["width"]*scale,height=dims["height"]*scale)
                cap=para(f"**Hình {len(used_figures):02d}.** "+b["caption"],"caption")
                src=para(f"Nguồn: {f['notebook']} · cell {f['cell_number_1']} · output {f['output_index_0']+1}." + (" "+b["note"] if b.get("note") else ""),"source")
                story += [KeepTogether([Spacer(1,6),img,cap,src]),Spacer(1,5)]
            else:raise ValueError(kind)
    missing=sorted(set(figmap)-set(used_figures))
    if missing:raise ValueError(f"Notebook figures not included: {missing}")
    if len(used_figures)!=len(set(used_figures)):raise ValueError("Duplicate figure placements")
    doc.multiBuild(story)
    manifest={
        "built_at_bangkok":datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "pdf":out.name,"pdf_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "figure_count":len(used_figures),"figure_ids":used_figures,
        "figure_number_to_extraction_id":{str(n):f for n,f in enumerate(used_figures,1)},
        "source": "report_content.json", "chapters":len(source["chapters"]),
        "method":"ReportLab; Arial TrueType embedded; saved notebook PNG images unchanged",
    }
    (HERE/"report_build_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
