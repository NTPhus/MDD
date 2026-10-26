"""Bundle the two authored reports, retaining links and nested PDF bookmarks."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'training_report' / '_build_deps'))
sys.path.insert(0, str(HERE / 'training_report'))
import pymupdf
import build_report as layout
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm

parts = [
    ('Phần I · Huấn luyện và đánh giá', HERE / 'training_report/MDD_Bao_cao_huan_luyen_Wav2Vec2_Qwen.pdf'),
    ('Phần II · Pipeline và các model local', HERE / 'pipeline_report/MDD_Pipeline_hien_tai_Kien_truc_Model_Workflow.pdf'),
]
inputs = [(title, path, pymupdf.open(path)) for title, path in parts]
cover_bytes = io.BytesIO()
c = canvas.Canvas(cover_bytes, pagesize=layout.A4)
c.setTitle('MDD – Báo cáo tổng hợp: huấn luyện, pipeline và đánh giá')
c.setAuthor('Dự án MDD')
width, height = layout.A4
c.setFillColor(layout.INK); c.rect(0, height-100*mm, width, 100*mm, fill=1, stroke=0)
c.setFillColorRGB(1, 1, 1); c.setFont('Arial-Bold', 12)
c.drawString(22*mm, height-28*mm, 'MDD  /  BÁO CÁO TỔNG HỢP')
style = ParagraphStyle('BundleTitle', fontName='Arial-Bold', fontSize=29, leading=39, textColor='white')
title = Paragraph('Huấn luyện, pipeline<br/>và đánh giá hệ thống', style)
_, title_height = title.wrap(166*mm, 200)
title.drawOn(c, 22*mm, height-45*mm-title_height)

def paragraph(text, y, size=11, bold=False):
    style = ParagraphStyle('BundleBody', fontName='Arial-Bold' if bold else 'Arial',
                           fontSize=size, leading=size*1.6, textColor=layout.INK)
    item = Paragraph(text, style)
    _, used = item.wrap(166*mm, height)
    item.drawOn(c, 22*mm, y-used)
    return y-used-7*mm

y = height-117*mm
y = paragraph('Snapshot ngày 23/10/2026 · Từ mã nguồn, checkpoint và kết quả đã lưu', y, 11, True)
y = paragraph('Whisper small.en · wav2vec2 B · Qwen3-0.6B + LoRA · CROTTC-IF thử nghiệm · Kokoro local', y)
offset = 1
toc = []
for part_title, path, document in inputs:
    y = paragraph(f'{part_title}<br/><font size="10">Trang {offset+1}–{offset+len(document)} · {len(document)} trang</font>', y, 12, True)
    toc.append([1, part_title, offset+1])
    toc.extend([[level+1, title, page+offset] for level, title, page in document.get_toc()])
    offset += len(document)
y = paragraph('Có cấu hình và flow train, sơ đồ runtime, kết quả trước/sau train, đối chứng ASR/MDD, giới hạn và 21 nguồn sơ cấp có liên kết.', y, 10)
y = paragraph('Kết quả đo trên máy được tách khỏi số công bố trong bài báo. Không chạy thêm training, inference hoặc API để biên soạn tài liệu.', y, 10)
c.setFont('Arial', 8); c.setFillColor(layout.MUTED)
c.drawString(22*mm, 15*mm, 'Các phần giữ mục lục và số trang riêng; bookmark PDF dẫn trực tiếp đến từng chương.')
c.save()

combined = pymupdf.open(stream=cover_bytes.getvalue(), filetype='pdf')
for _, _, document in inputs: combined.insert_pdf(document)
combined.set_toc(toc)
combined.set_metadata({'title': 'MDD – Báo cáo tổng hợp huấn luyện, pipeline và đánh giá',
                       'author': 'Dự án MDD', 'subject': 'Snapshot 23/10/2026; bằng chứng local và 21 nguồn sơ cấp'})
out = HERE / 'MDD_Bao_cao_tong_hop_20261023.pdf'
combined.save(out, garbage=4, deflate=True)
combined.close()
for _, _, document in inputs: document.close()

with pymupdf.open(out) as doc:
    assert len(doc) == offset
    assert len(doc.get_toc()) == len(toc)
    text = '\n'.join(page.get_text() for page in doc)
    assert '\ufffd' not in text
    urls = {link.get('uri') for page in doc for link in page.get_links() if link.get('uri')}
    references = json.loads((HERE / 'references_20261023.json').read_text(encoding='utf-8'))['references']
    assert {row['url'] for row in references} <= urls
    doc[0].get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False).save(HERE / 'pipeline_report/qa/bundle_cover.png')
    manifest = {'built_at_bangkok': datetime.now(timezone(timedelta(hours=7))).isoformat(),
                'file': out.name, 'pages': len(doc), 'bookmarks': len(toc),
                'unique_primary_sources': len(references), 'replacement_characters': text.count('\ufffd'),
                'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                'inputs': [{'file': str(path.relative_to(HERE)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for _, path in parts], 'training_or_inference_executed': False}
(HERE / 'report_bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
