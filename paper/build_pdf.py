"""Build paper/manuscript.pdf from paper/manuscript.md (styled HTML, printed by headless Edge).

uv run --with markdown python paper/build_pdf.py
"""

import subprocess
import tempfile
import time
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CSS = """
@page { size: Letter; margin: 2.2cm 2.2cm 2.4cm 2.2cm;
        @bottom-center { content: counter(page); font: 9pt "Times New Roman", serif;
                         color: #555; } }
body { font-family: "Times New Roman", Georgia, serif; font-size: 11pt; line-height: 1.45;
       color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 10pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 7pt 18pt; padding: 0; }
li { margin: 0 0 2pt; }
hr { border: 0; border-top: 0.6pt solid #999; margin: 12pt 0; }
table { border-collapse: collapse; margin: 4pt 0 12pt; font-size: 8.6pt; width: 100%;
        break-inside: avoid; }
th, td { border-top: 0.5pt solid #bbb; border-bottom: 0.5pt solid #bbb; padding: 2.5pt 4pt;
         text-align: left; }
th { background: #f1f1f1; }
img { width: 100%; margin: 4pt 0 2pt; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
/* captions stay whole and with their tables and figures */
p:has(+ table), p:has(> img) + p { break-inside: avoid; }
p:has(+ table), p:has(+ p > img), p:has(> img) { break-after: avoid; }
"""


def main() -> None:
    md = (HERE / "manuscript.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    title = md.splitlines()[0].lstrip("# ").strip()
    html = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    page = HERE / "manuscript.html"
    page.write_text(html, encoding="utf-8")
    out = HERE / "manuscript.pdf"
    out.unlink(missing_ok=True)
    # a separate profile, so a running Edge window can't take the job and drop it
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(  # noqa: S603 - a fixed browser path and our own file, no outside input
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                f"--user-data-dir={profile}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out}",
                page.as_uri(),
            ],
            check=True,
            timeout=120,
            capture_output=True,  # waits until Edge's printing child closes the pipe
        )
    for _ in range(60):  # and, in case it hands off once more, until the file appears
        if out.exists() and out.stat().st_size > 0:
            break
        time.sleep(1)
    else:
        raise SystemExit("Edge did not write the PDF")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
