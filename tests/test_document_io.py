from pathlib import Path
from sda.utils.document_io import extract_text, document_to_image

def test_extract_text_and_render(tmp_path):
    p = tmp_path/'doc.txt'; p.write_text('INVOICE test USD 10.00', encoding='utf-8')
    assert 'INVOICE' in extract_text(p)
    img = document_to_image(p, cache_dir=tmp_path)
    assert img.exists()
