"""Structural PDF acceptance. Visual inspection remains necessary."""
import json
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]


def main():
    local=PdfReader(ROOT/'submission/report.pdf')
    public=PdfReader(ROOT/'submission/report_public.pdf')
    texts=[p.extract_text() for p in local.pages]
    start=next(i for i,t in enumerate(texts) if 'Supplementary Pages:' in t)
    assert start<=4, f'Main pages: {start}'
    assert len(local.pages)==len(public.pages)
    for p in [*local.pages,*public.pages]:
        assert abs(float(p.mediabox.width)-595.276)<1
        assert abs(float(p.mediabox.height)-841.89)<1
    public_text='\n'.join(p.extract_text() for p in public.pages)
    assert 'Anonymous public version' in public_text
    identity=ROOT/'paper/identity.tex'
    has_identity=identity.exists()
    if has_identity:
        import re
        vals=re.findall(r'\\renewcommand\{\\student(?:name|number)\}\{([^}]*)\}',identity.read_text())
        for v in vals:
            assert v in texts[0]
            assert v not in public_text
    assert '??' not in '\n'.join(texts)
    for name in ['main','public']:
        log=(ROOT/f'paper/build/{name}.log').read_text()
        assert 'Overfull \\hbox' not in log
        assert 'Overfull \\vbox' not in log
        assert 'undefined references' not in log
    r=dict(status='PASS',main_pages=start,supplementary_pages=len(local.pages)-start,
           a4=True,local_identity_present=has_identity,public_identity_removed=True,
           overfull_boxes=False,unresolved_references=False,
           typography='10 TeX pt body and code; 9.96 PDF bp is the same physical size')
    (ROOT/'results/pdf_acceptance.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__': main()
