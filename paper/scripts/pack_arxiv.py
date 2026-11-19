"""Pack paper/arxiv/ into a self-contained zip for upload to arXiv.

    python paper/scripts/make_arxiv.py      # refresh the arXiv source from the ICLR source
    (cd paper/arxiv && latexmk -pdf gentrack_arxiv.tex)   # builds the .bbl
    python paper/scripts/pack_arxiv.py      # -> paper/arxiv/gentrack_arxiv_upload.zip

arXiv does not run BibTeX, so the compiled .bbl is included. The shared figures/ and tables/
are copied in and the three ../ paths are rewritten to point at the copies. The zip is
test-compiled in a temporary folder with pdflatex only, as arXiv does.
"""
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

A = Path("paper/arxiv")
OUT = A / "gentrack_arxiv_upload.zip"
FILES = ["gentrack_arxiv.bbl", "iclr2027_conference.sty", "iclr2027_conference.bst",
         "natbib.sty", "fancyhdr.sty", "math_commands.tex"]

tex = (A / "gentrack_arxiv.tex").read_text(encoding="utf-8")
for old, new in [(r"\graphicspath{{../figures/}}", r"\graphicspath{{figures/}}"),
                 (r"\input{../tables/", r"\input{tables/"),
                 (r"\bibliography{../references}", r"\bibliography{references}")]:
    assert old in tex, old
    tex = tex.replace(old, new)
assert "../" not in tex, "unrewritten relative path"
figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
tables = sorted(set(re.findall(r"\\input\{tables/([^}]+)\}", tex)))

with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    (t / "figures").mkdir()
    (t / "tables").mkdir()
    (t / "gentrack_arxiv.tex").write_text(tex, encoding="utf-8", newline="")
    for f in FILES:
        shutil.copy(A / f, t / f)
    for f in figs:
        name = f if f.endswith(".pdf") else f + ".pdf"
        shutil.copy(Path("paper/figures") / name, t / "figures" / name)
    for f in tables:
        shutil.copy(Path("paper/tables") / f, t / "tables" / f)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(t.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(t).as_posix())
    for _ in range(3):  # arXiv-style build: pdflatex only, no BibTeX
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "gentrack_arxiv.tex"],
                       cwd=t, capture_output=True, text=True, errors="replace")
    log = (t / "gentrack_arxiv.log").read_text(errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages", log, re.S)
    assert pages and "undefined" not in log, "test build failed or has undefined references"
    assert "Rerun to get cross-references" not in log, "cross-references did not settle"
print(f"{OUT}: {len(figs)} figures, {len(tables)} tables, test build {pages.group(1)} pages")
