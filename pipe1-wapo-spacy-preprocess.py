#!/usr/bin/env python

import argparse
import gzip
import spacy
import json
import re
import sys
import unidecode
import unicodedata
import numpy as np

parser = argparse.ArgumentParser(description='Preprocess WashingtonPost docs to text')
parser.add_argument('bundle', help='path to WaPo bundle to process: "./data/TREC_Washington_Post_collection.v3.jl.gz" or "-" to read from stdin')

args = parser.parse_args()

if args.bundle == '-':
    args.bundle = sys.stdin
else:
    if args.bundle.endswith('.gz'):
        args.bundle = gzip.open(args.bundle, 'rt', encoding='utf-8')
    else:
        args.bundle = open(args.bundle, 'r', encoding='utf-8')

nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")

tokenizer = nlp.tokenizer


def char_filter(string):
    latin = re.compile('[a-zA-Z]+')
    for char in unicodedata.normalize('NFC', string):
        decoded = unidecode.unidecode(char)
        if latin.match(decoded):
            yield char
        else:
            yield decoded


def clean_string(string):
    return "".join(char_filter(string))


def unicode_character_name(char):
    try:
        return unicodedata.name(char)
    except ValueError:
        return None


# Generate all Unicode characters with their names
all_unicode_characters = []
for n in range(0, 0x10ffff):  # Unicode planes 0-16
    char = chr(n)  # Python 3
    # char = unichr(n)           # Python 2
    name = unicode_character_name(char)
    if name:
        all_unicode_characters.append((char, name))

# Find all Unicode quotation marks
unicode_quotations = ''.join([char for char, name in all_unicode_characters if 'QUOTATION MARK' in name])
# " « » ‘ ’ ‚ ‛ “ ” „ ‟ ‹ › ❛ ❜ ❝ ❞ ❟ ❠ ❮ ❯ ⹂ 〝 〞 〟 ＂ 🙶 🙷 🙸

# Find all Unicode hyphens
unicode_hyphens = ''.join([char for char, name in all_unicode_characters if 'HYPHEN' in name])
# - ­ ֊ ᐀ ᠆ ‐ ‑ ‧ ⁃ ⸗ ⸚ ⹀ ゠ ﹣ － 󠀭

# Find all Unicode dashes
unicode_dashes = ''.join([char for char, name in all_unicode_characters if 'DASH' in name and 'DASHED' not in name])
# ‒ – — ⁓ ⊝ ⑈ ┄ ┅ ┆ ┇ ┈ ┉ ┊ ┋ ╌ ╍ ╎ ╏ ⤌ ⤍ ⤎ ⤏ ⤐ ⥪ ⥫ ⥬ ⥭ ⩜ ⩝ ⫘ ⫦ ⬷ ⸺ ⸻ ⹃ 〜 〰 ︱ ︲ ﹘ 💨

char_mapper = {}
for c in unicode_quotations:
    char_mapper[c] = '"'
for c in f'{unicode_dashes}{unicode_hyphens}"':
    char_mapper[c] = '-'
for c in f"‘’‛❛❜'":
    char_mapper[c] = "'"


def get_first_content_by_type(jsarr, t):
    res = None
    for block in jsarr:
        if block is not None and block['type'] == t:
            res = block['content']
            break
    return res


def get_all_content_by_type(jsarr, t, field='content'):
    strings = [c[field] for c in jsarr if c is not None and c['type'] == t and field in c and c[field] is not None]
    if strings:
        return ' '.join(strings)
    else:
        return None


def get_all_content_by_type_and_field(jsarr, t, field=None):
    assert t is not None
    assert field is not None
    if type(t) is not list:
        t = [t]
    if type(field) is not list:
        field = [field]

    strings = [
        "\n".join(c[cc]) if type(c[cc]) is list
        else c[cc]['text'] if type(c[cc]) is dict and 'text' in c[cc]
        else c[cc]['title'] if type(c[cc]) is dict and 'title' in c[cc]
        else c[cc]

        for c in jsarr
        if c and c['type'] in t
        for cc in c
        if cc in field and c[cc]
    ]

    if strings:
        res = '\n'.join(strings)
        res = ''.join([char_mapper[c] if c in char_mapper else c for c in res])
        return res
    else:
        return None


def unique_heads(entry):
    items = set()
    if type(entry) is list:
        for x in entry:
            items.add(x[0])
        return list(items)
    else:
        return entry


def prepare_document(js):
    title = get_first_content_by_type(js['contents'], 'title')
    text = f"{title}.\n{get_all_content_by_type_and_field(js['contents'], ['sanitized_html', 'list', 'image', 'tweet', 'video'], ['content', 'fullcaption', 'text'])}"
    links = []
    if text:
        links = re.findall('href="([^"]*)"', text)
        text = re.sub('<.*?>', ' ', text)
        text = clean_string(text)

    pubdate = 0
    if 'published_date' in js:
        pubdate = js['published_date'] or 0

    text = text or ''
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    sentences = [sent for sent in sentences if len(sent) > 3]

    source_block = {
        "id": js['id'],
        "title": title,
        "pubdate": pubdate,
        "date": get_first_content_by_type(js['contents'], 'date'),
        "kicker": get_first_content_by_type(js['contents'], 'kicker'),
        "author": js['author'],

        "text": text,
        "sentences": sentences,

        "tlen_chars": len(text),
        "tlen_tokens": len(tokenizer(text)),
        "tlen_sent_len_min": float(np.min([len(s) for s in sentences])) if len(sentences) > 0 else 0,
        "tlen_sent_len_max": float(np.max([len(s) for s in sentences])) if len(sentences) > 0 else 0,
        "tlen_sent_len_mean": float(np.mean([len(s) for s in sentences])) if len(sentences) > 0 else 0,

        "captions": get_all_content_by_type(js['contents'], 'image', field='fullcaption'),
        "links": links or [],
        "url": js['article_url'],
    }

    return source_block


for line in args.bundle:
    js = json.loads(line)
    doc = prepare_document(js)
    print(json.dumps(doc))
