#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=6,<7"]
# ///
"""Make a Unicode-attacked copy of the report without changing its appearance.

Run from the project folder:
    uv run --script output/attack_report.py
    uv run --script output/attack_report.py path/to/report.pdf -o attacked.pdf
    uv run --script output/attack_report.py --family invisible

Or install pypdf once (python -m pip install 'pypdf>=6,<7') and use python.
The default input is output/pdf/main.pdf, relative to this script, so running
from another folder works too. The default output is <input>_attacked.pdf.

Like the earlier attacked PDFs, this edits the fonts' ToUnicode maps. The page
still looks the same; copying or extracting its text exposes the perturbations.
OCR of the rendered pages can recover the original visible text.
The References heading and everything after it keep their original Unicode,
so bibliography entries and URLs remain copyable. The heading is found anew
on each run, even if edits move it to another page or column.

By default, every eligible mapping gets a homoglyph (or a fullwidth fallback),
canonical decomposition where available, and an invisible character after each
letter or number. --family selects just one attack instead. The character sets
come from notebook 02 and config/experiment.yaml; this file is self-contained.

This is a saturated PDF demonstration, not the notebook's position-sampled
strength experiment: a glyph mapping is shared by all occurrences in its font.
The original PDF is never overwritten. Rerunning replaces the attacked copy.
"""

import argparse
import hashlib
from pathlib import Path
import random
import re
import tempfile
import unicodedata

try:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject
except ImportError:
    raise SystemExit(
        "Missing pypdf. Run: uv run --script output/attack_report.py\n"
        "Or install it: python -m pip install 'pypdf>=6,<7'"
    )


HOMOGLYPHS = {
    'A': ('Α', 'А'), 'B': ('Β', 'В'), 'C': ('Ϲ', 'С'), 'E': ('Ε', 'Е'),
    'H': ('Η', 'Н'), 'I': ('Ι', 'І'), 'J': ('Ј',), 'K': ('Κ', 'К'),
    'M': ('Μ', 'М'), 'N': ('Ν',), 'O': ('Ο', 'О'), 'P': ('Ρ', 'Р'),
    'S': ('Ѕ',), 'T': ('Τ', 'Т'), 'X': ('Χ', 'Х'), 'Y': ('Υ', 'У'), 'Z': ('Ζ',),
    'a': ('а',), 'c': ('с',), 'e': ('е',), 'h': ('һ',), 'i': ('і',),
    'j': ('ј',), 'k': ('κ',), 'o': ('ο', 'о'), 'p': ('р',), 's': ('ѕ',),
    'x': ('х',), 'y': ('у',),
}
INVISIBLES = tuple(chr(n) for n in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))
FAMILIES = ('all', 'homoglyph', 'invisible', 'compatibility', 'composition', 'mixed')


def perturb(text, family, seed):
    """Keep choices reproducible, including when fonts appear in a new order."""
    digest = hashlib.sha256(f'{seed}|{family}|{text}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(digest, 'big'))
    result = []
    for ch in text:
        fullwidth = chr(ord(ch) + 0xFEE0) if 0x21 <= ord(ch) <= 0x7E else None
        decomposed = unicodedata.normalize('NFD', ch)
        options = {
            'homoglyph': list(HOMOGLYPHS.get(ch, ())),
            'invisible': [ch + x for x in INVISIBLES] if ch.isalnum() else [],
            'compatibility': [fullwidth] if fullwidth else [],
            'composition': [decomposed] if decomposed != ch else [],
        }
        if family == 'all':
            replacement = rng.choice(options['homoglyph']) if options['homoglyph'] else (fullwidth or ch)
            replacement = unicodedata.normalize('NFD', replacement)
            if ch.isalnum():
                replacement += rng.choice(INVISIBLES)
        else:
            choices = sum(options.values(), []) if family == 'mixed' else options[family]
            replacement = rng.choice(choices) if choices else ch
        result.append(replacement)
    return ''.join(result)


# A ToUnicode map can list individual glyphs or compact ranges. Expand ranges
# before editing, since attacked replacements can contain several code points.
HEX = r'<([0-9A-Fa-f\s]+)>'
BLOCK = re.compile(r'\d+\s+beginbf(char|range)\b(.*?)endbf\1\b', re.S)


def hex_bytes(value):
    return bytes.fromhex(value)


def read_mapping_block(kind, body):
    body = re.sub(r'%[^\r\n]*', '', body)
    pattern = re.compile(HEX + r'\s*' + HEX) if kind == 'char' else re.compile(
        HEX + r'\s*' + HEX + r'\s*(<[^>]+>|\[[^\]]*\])', re.S
    )
    if pattern.sub('', body).strip():
        raise ValueError('Unsupported ToUnicode mapping syntax; no output written.')
    for entry in pattern.finditer(body):
        if kind == 'char':
            yield hex_bytes(entry[1]), hex_bytes(entry[2])
            continue
        start, end = hex_bytes(entry[1]), hex_bytes(entry[2])
        first, last = int.from_bytes(start, 'big'), int.from_bytes(end, 'big')
        if len(start) != len(end) or not 0 <= last - first <= 65535:
            raise ValueError('Unsupported ToUnicode range; no output written.')
        values = [hex_bytes(x) for x in re.findall(HEX, entry[3])]
        array = entry[3].startswith('[')
        if (array and len(values) != last - first + 1) or not values:
            raise ValueError('Invalid ToUnicode range; no output written.')
        for offset in range(last - first + 1):
            destination = values[offset] if array else (
                int.from_bytes(values[0], 'big') + offset
            ).to_bytes(len(values[0]), 'big')
            yield (first + offset).to_bytes(len(start), 'big'), destination


def attack_cmap(data, family, seed):
    cmap = data.decode('ascii')
    if 'usecmap' in cmap:
        raise ValueError('Inherited ToUnicode maps are not supported by this script.')
    changed = 0

    def replace_block(match):
        nonlocal changed
        entries = []
        for source, destination in read_mapping_block(match[1], match[2]):
            before = destination.decode('utf-16-be')
            after = perturb(before, family, seed)
            changed += before != after
            entries.append(f'<{source.hex().upper()}> <{after.encode("utf-16-be").hex().upper()}>')
        # PDF CMap blocks allow at most 100 entries.
        return '\n'.join(
            f'{len(entries[i:i + 100])} beginbfchar\n'
            + '\n'.join(entries[i:i + 100]) + '\nendbfchar'
            for i in range(0, len(entries), 100)
        )

    return BLOCK.sub(replace_block, cmap).encode('ascii'), changed


def find_references(reader):
    """Find the heading's text operation, including on a partly body-text page."""
    for page_number, page in enumerate(reader.pages):
        operation = -1
        depth = 0
        pending = []
        heading = None

        def before(op, args, cm, tm):
            nonlocal operation, depth
            if depth == 0:
                operation += 1
                if op in (b'Tj', b'TJ', b"'", b'"'):
                    pending.append(operation)
            if op == b'Do':
                depth += 1  # Imported figures have their own content operations.

        def after(op, args, cm, tm):
            nonlocal depth
            if op == b'Do':
                depth -= 1

        def text(value, cm, tm, font, size):
            nonlocal heading
            if depth == 0 and value.strip():
                if value.strip() == 'References' and pending:
                    heading = pending[0]
                pending.clear()

        page.extract_text(visitor_operand_before=before, visitor_operand_after=after, visitor_text=text)
        if heading is not None:
            return page_number, heading
    raise ValueError('No References heading found; stopped to avoid attacking the bibliography.')


def preserve_references(writer, boundary):
    """Give the bibliography clean font maps before attacking the shared originals."""
    protected = set()
    clean_fonts = {}
    first_page, first_operation = boundary
    for page_number in range(first_page, len(writer.pages)):
        page = writer.pages[page_number]
        content = page.get_contents()
        if content is None:
            continue
        resources = DictionaryObject(page['/Resources'])
        font_resources = resources.get('/Font')
        fonts = DictionaryObject(font_resources.get_object() if font_resources else {})
        resources[NameObject('/Font')] = fonts
        page[NameObject('/Resources')] = resources
        aliases = {}
        for name, reference in list(fonts.items()):
            font = reference.get_object()
            if id(font) not in clean_fonts:
                copy = DictionaryObject(font)
                if '/ToUnicode' in font:
                    stream = DecodedStreamObject()
                    stream.set_data(font['/ToUnicode'].get_data())
                    protected.add(id(stream))
                    copy[NameObject('/ToUnicode')] = writer._add_object(stream)
                clean_fonts[id(font)] = writer._add_object(copy)
            alias = NameObject(f'{name}_References')
            while alias in fonts:
                alias = NameObject(f'{alias}_')
            aliases[name] = alias
            fonts[alias] = clean_fonts[id(font)]
        operations = []
        active_font = None
        start = first_operation if page_number == first_page else 0
        for index, (args, op) in enumerate(content.operations):
            # The heading can reuse a font selected before the boundary.
            if index == start and active_font is not None:
                operations.append(([aliases[active_font[0]], active_font[1]], b'Tf'))
            if op == b'Tf':
                active_font = args
                if index >= start:
                    args = [aliases[args[0]], args[1]]
            operations.append((args, op))
        content.operations = operations
        page.replace_contents(content)
    return protected


def attack_pdf(source, destination, family='all', seed=13):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or (destination.exists() and source.samefile(destination)):
        raise ValueError('Choose a different output path; the original must be preserved.')
    reader = PdfReader(source)
    if reader.is_encrypted:
        raise ValueError('Use an unencrypted report PDF.')
    boundary = find_references(reader)
    writer = PdfWriter(clone_from=reader)
    protected = preserve_references(writer, boundary)
    seen, processed = set(), set()
    counts = {'maps': 0, 'changed_mappings': 0, 'fonts_without_map': 0,
              'references_page': boundary[0] + 1}

    def visit(value):
        obj = value.get_object()
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, DictionaryObject):
            if obj.get('/Type') == '/Font' and obj.get('/Subtype') not in ('/CIDFontType0', '/CIDFontType2'):
                stream = obj.get('/ToUnicode')
                if stream is None:
                    counts['fonts_without_map'] += 1
                else:
                    stream = stream.get_object()
                    if id(stream) not in processed and id(stream) not in protected:
                        processed.add(id(stream))
                        data, changed = attack_cmap(stream.get_data(), family, seed)
                        if changed:
                            stream.set_data(data)
                        counts['maps'] += 1
                        counts['changed_mappings'] += changed
            for child in obj.values():
                visit(child)
        elif isinstance(obj, ArrayObject):
            for child in obj:
                visit(child)

    # Walking the document also reaches the fonts inside imported figure PDFs.
    visit(writer.root_object)
    if not counts['maps']:
        raise ValueError('No ToUnicode maps found. This script needs a text-based PDF.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.pdf', dir=destination.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        writer.write(temporary)
        check = PdfReader(temporary)
        if len(check.pages) != len(reader.pages):
            raise ValueError('Page-count verification failed.')
        # Do not replace an earlier output if even one bibliography character changed.
        for index in range(boundary[0], len(reader.pages)):
            before = reader.pages[index].extract_text()
            after = check.pages[index].extract_text()
            if index == boundary[0]:
                before = before[before.index('References'):]
                after = after[after.index('References'):]
            if before != after:
                raise ValueError('Bibliography verification failed; no output written.')
        body_before, body_after = [], []
        for index in range(boundary[0] + 1):
            before = reader.pages[index].extract_text()
            after = check.pages[index].extract_text()
            if index == boundary[0]:
                before = before[:before.index('References')]
                after = after[:after.index('References')]
            body_before.append(before)
            body_after.append(after)
        counts['body_text_changed'] = body_before != body_after
        # A font can contain eligible glyphs used only in the bibliography.
        # In that case (e.g. accents for composition), an unchanged body is valid.
        eligible_body = any(perturb(text, family, seed) != text for text in body_before)
        if eligible_body and not counts['body_text_changed']:
            raise ValueError('Text extraction did not change; this PDF may override its font maps.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', type=Path, default=Path(__file__).resolve().parent / 'pdf/report.pdf')
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--family', choices=FAMILIES, default='all')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    output = args.output or args.input.with_name(f'{args.input.stem}_alt.pdf')
    try:
        counts = attack_pdf(args.input, output, args.family, args.seed)
    except Exception as exc:
        parser.exit(1, f'Could not create attacked PDF: {exc}\n')
    print(f'Created {output.resolve()}')
    print(f"Changed {counts['changed_mappings']} glyph mappings across {counts['maps']} Unicode maps.")
    print(f"References preserved from page {counts['references_page']} onward.")
    if counts['fonts_without_map']:
        print(f"Note: {counts['fonts_without_map']} fonts have no Unicode map and were left unchanged.")
    if not counts['body_text_changed']:
        print('No eligible body text for this family; the copy contains no text changes.')
    print('Pages look the same; copy/paste or text extraction shows the Unicode attacks.')


if __name__ == '__main__':
    main()
