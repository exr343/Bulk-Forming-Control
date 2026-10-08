"""Render report.md (with its figure) to report.pdf in this folder.

Needs PyMuPDF (in jax-fem-env) and the `markdown` package
(`pip install markdown`). Run from anywhere:
    python applications/Agility_Forge/progress/control_reports/2026-10-08_mpc_briefing/make_pdf.py
"""
import os

import markdown
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1b1b1b; }
h1 { font-size: 17pt; color: #0b0b0b; margin: 0 0 6pt 0; }
h2 { font-size: 13.5pt; color: #0b0b0b; margin: 16pt 0 5pt 0; border-bottom: 1px solid #cfcfca; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; color: #1b1b1b; margin: 12pt 0 4pt 0; }
p { margin: 0 0 7pt 0; }
li { margin: 0 0 3pt 0; }
code { font-family: monospace; font-size: 9pt; color: #3a3935; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0; }
th { font-size: 9pt; font-weight: bold; text-align: left; padding: 3pt 4pt; border: 1px solid #c9c8c2; border-bottom: 2px solid #8a8984; }
td { font-size: 9pt; padding: 3pt 4pt; border: 1px solid #d9d8d2; vertical-align: top; }
img { width: 100%; }
.meta { font-size: 9pt; color: #52514e; margin-bottom: 10pt; }
"""


def main():
    md = open(os.path.join(HERE, "report.md"), encoding="utf-8").read()
    lines = md.split("\n")
    # The line under the title (date, code, jobs) is shown smaller as a meta block.
    title, rest = lines[0], "\n".join(lines[1:]).lstrip("\n")
    meta, _, body = rest.partition("\n\n")
    html = (markdown.markdown(title, extensions=["tables"])
            + f'<p class="meta">{markdown.markdown(meta)[3:-4]}</p>'
            + markdown.markdown(body, extensions=["tables"]))
    story = pymupdf.Story(html=f"<body>{html}</body>", user_css=CSS, archive=HERE)
    out = os.path.join(HERE, "report.pdf")
    writer = pymupdf.DocumentWriter(out)
    page = pymupdf.paper_rect("letter")
    where = page + (54, 54, -54, -54)                     # 0.75 inch margins
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # Page numbers.
    doc = pymupdf.open(out)
    for i, pg in enumerate(doc):
        pg.insert_text((page.width / 2 - 10, page.height - 28), f"{i + 1} / {doc.page_count}", fontsize=8,
                       color=(0.45, 0.45, 0.43))
    doc.saveIncr()
    print("wrote", out, f"({doc.page_count} pages)")


if __name__ == "__main__":
    main()
