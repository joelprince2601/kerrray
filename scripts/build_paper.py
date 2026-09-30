"""Build research.md into paper/build/research.html and research.pdf.

A small, dependency-free Markdown converter for the subset used by the
manuscript (headings, paragraphs, bold, italics, inline code, code blocks,
bullet and numbered lists, tables, images), followed by printing to PDF with
headless Chrome or Edge. Unicode mathematics is rendered by the browser.

Usage:
    python scripts/build_paper.py            # HTML and PDF
    python scripts/build_paper.py --html     # HTML only

The PDF is a preview for review; a journal or arXiv version can be typeset
from the same source.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research.md"
OUT = ROOT / "paper" / "build"
BROWSERS = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }
body { font-family: "Cambria", "STIX Two Text", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 170mm; margin: 0 auto; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 8pt; text-align: center; }
h1 + p, h1 + p + p { text-align: center; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; border-bottom: 0.5pt solid #999; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; hyphens: auto; }
code { font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 8.8pt; }
pre { background: #f5f5f5; border: 0.5pt solid #ddd; padding: 6pt; font-size: 8.3pt; overflow-x: auto; break-inside: avoid; }
table { border-collapse: collapse; margin: 6pt auto 10pt; font-size: 8.8pt; break-inside: avoid; }
th, td { border-top: 0.5pt solid #999; border-bottom: 0.5pt solid #999; padding: 2.5pt 6pt; text-align: left; }
th { background: #f0f0f0; }
img { display: block; max-width: 100%; max-height: 95mm; margin: 8pt auto 2pt; }
ul, ol { margin: 4pt 0 6pt 16pt; padding-left: 10pt; }
li { margin: 2pt 0; }
hr { border: none; border-top: 0.5pt solid #bbb; margin: 12pt 0; }
"""


def inline(text: str) -> str:
    """Escape HTML, then apply inline code, bold, italics and links."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        out.append(s)
    return "".join(out)


def convert(md: str, base: Path) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(block))}</code></pre>")
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if s == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m:
            flush()
            src = (base / m.group(2)).resolve().as_uri()
            out.append(f'<img src="{src}" alt="{html.escape(m.group(1))}">')
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush()
            header = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in header)
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush()
            ordered = m.group(2)[0].isdigit()
            items: list[str] = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if mm and (mm.group(2)[0].isdigit()) == ordered:
                    items.append(mm.group(3))
                    i += 1
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            continue
        if not s:
            flush()
            i += 1
            continue
        para.append(s)
        i += 1
    flush()
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="write HTML only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    body = convert(SRC.read_text(encoding="utf-8"), ROOT)
    title = SRC.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    page = (f"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = OUT / "research.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"wrote {html_path}")
    if args.html:
        return
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("chrome") or shutil.which("chromium")
    if not browser:
        raise SystemExit("no Chrome/Edge found; open research.html and print to PDF instead")
    pdf_path = OUT / "research.pdf"
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
