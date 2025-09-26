#!/usr/bin/env python3
"""Clean Jupyter notebooks by removing top-level metadata.widgets and ensuring nbformat metadata.

Usage: python tools/clean_notebooks.py <notebook1.ipynb> [notebook2.ipynb ...]
This script is safe to run as a pre-commit hook to avoid GitHub rendering issues.
"""
import sys
import json
from pathlib import Path


def clean_nb(path: Path) -> bool:
    changed = False
    nb = json.loads(path.read_text(encoding='utf-8'))
    # Ensure nbformat fields
    if 'nbformat' not in nb:
        nb['nbformat'] = 4
        nb['nbformat_minor'] = 5
        changed = True
    if 'metadata' not in nb:
        nb['metadata'] = {}
        changed = True
    # Remove problematic top-level widgets mapping if present
    if 'widgets' in nb['metadata']:
        nb['metadata'].pop('widgets', None)
        changed = True
    # Optionally ensure kernelspec/language_info exist (non-destructive)
    meta = nb['metadata']
    if 'kernelspec' not in meta:
        meta['kernelspec'] = {"name": "python3", "display_name": "Python 3"}
        changed = True
    if 'language_info' not in meta:
        meta['language_info'] = {"name": "python"}
        changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding='utf-8')
    return changed


def main(argv):
    if len(argv) < 2:
        print('Usage: clean_notebooks.py <notebook1.ipynb> [...]')
        return 1
    any_changed = False
    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            continue
        if clean_nb(path):
            print('Cleaned', p)
            any_changed = True
    return 0 if any_changed else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
