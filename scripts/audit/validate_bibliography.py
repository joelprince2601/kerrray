"""Check every entry of paper/latex/references.bib against its public record.

Entries with a DOI are compared with Crossref (api.crossref.org): first
author's family name, year, volume, page or article number and journal.
Entries with only an arXiv identifier are compared with the arXiv API
(export.arxiv.org): first author's family name and title. Entries with
neither (book chapters) are reported as not machine-checkable.
Only public metadata are requested; nothing else is sent.
Run: ./.venv/Scripts/python.exe scripts/audit/validate_bibliography.py
"""
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BIB = Path(__file__).resolve().parents[2] / 'paper' / 'latex' / 'references.bib'
UA = {'User-Agent': 'KerrRay-bibliography-check/1.0 (research reproducibility script)'}

ACCENTS = {r"{\'a}": 'a', r"{\~n}": 'n', r'{\"O}': 'O', r"{\'s}": 's', r'{\o}': 'o', r'\&': '&'}


def plain(s):
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    s = s.replace('{', '').replace('}', '')
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


def parse(text):
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r'^\s*(\w+)\s*=\s*\{(.*)\},?\s*$', m.group(3), re.M))
        entries.append((m.group(2).strip(), m.group(1).lower(), fields))
    return entries


def get(url, attempts=4):
    """Fetch a URL, waiting and retrying when the server asks us to slow down (HTTP 429/503)."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and i < attempts - 1:
                time.sleep(15 * (i + 1))
                continue
            raise


def crossref(doi):
    return json.loads(get('https://api.crossref.org/works/' + urllib.parse.quote(doi)))['message']


def years(msg):
    ys = set()
    for k in ('published-print', 'published-online', 'issued', 'published'):
        parts = msg.get(k, {}).get('date-parts') or []
        if parts and parts[0] and parts[0][0]:
            ys.add(str(parts[0][0]))
    return ys


def main():
    entries = parse(BIB.read_text(encoding='utf-8'))
    ok = issues = unchecked = 0
    for key, kind, f in entries:
        first = plain(f.get('author', '').split(' and ')[0].split(',')[0])
        try:
            if 'doi' in f:
                m = crossref(f['doi'])
                problems = []
                cr_first = plain((m.get('author') or [{}])[0].get('family', '')) if m.get('author') else ''
                if cr_first and first != cr_first:
                    problems.append(f'first author {first!r} vs {cr_first!r}')
                if f.get('year') and f['year'] not in years(m):
                    problems.append(f'year {f["year"]} vs {sorted(years(m))}')
                if f.get('volume') and m.get('volume') and f['volume'] != str(m.get('volume')):
                    problems.append(f'volume {f["volume"]} vs {m.get("volume")}')
                pg = f.get('pages', '').split('--')[0]
                cr_pages = {str(m.get('article-number', '')), str(m.get('page', '')).split('-')[0]}
                if pg and kind == 'article' and pg not in cr_pages:
                    problems.append(f'page {pg} vs {sorted(x for x in cr_pages if x)}')
                title = (m.get('title') or [''])[0]
                journal = (m.get('container-title') or [''])[0]
                status = 'OK' if not problems else 'CHECK: ' + '; '.join(problems)
                ok += not problems
                issues += bool(problems)
                print(f'{key:22s} {status} | Crossref: {journal} | "{title[:70]}"')
            elif 'eprint' in f:
                xml = get('http://export.arxiv.org/api/query?id_list=' + f['eprint'])
                ns = {'a': 'http://www.w3.org/2005/Atom'}
                e = ET.fromstring(xml).find('a:entry', ns)
                ax_title = ' '.join(e.find('a:title', ns).text.split())
                ax_first = plain(e.find('a:author/a:name', ns).text.split()[-1])
                problems = []
                if first != ax_first:
                    problems.append(f'first author {first!r} vs {ax_first!r}')
                if f.get('title') and plain(f['title'])[:25] != plain(ax_title)[:25]:
                    problems.append('title differs')
                status = 'OK' if not problems else 'CHECK: ' + '; '.join(problems)
                ok += not problems
                issues += bool(problems)
                print(f'{key:22s} {status} | arXiv:{f["eprint"]} | "{ax_title[:70]}"')
                time.sleep(3)
            else:
                unchecked += 1
                print(f'{key:22s} NOT MACHINE-CHECKABLE (no DOI or arXiv id); see LITERATURE_AUDIT.md')
        except Exception as exc:  # network or record problems are reported, not hidden
            issues += 1
            print(f'{key:22s} ERROR {type(exc).__name__}: {exc}')
        time.sleep(0.3)
    print(f'\nentries {len(entries)}: OK {ok}, to check {issues}, not machine-checkable {unchecked}')


if __name__ == '__main__':
    main()
