#!/usr/bin/env python
import json
from collections import defaultdict
from itertools import islice

import argparse
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser(description='converting text to embedding vector')
parser.add_argument('--text_field', default='text', choices=['text'], help='which field from json document should be used for embedding.')
parser.add_argument('--window_size', default=250, choices=[250, 500, 750, 1000], type=int, help='split document on chunks of this many tokens (feel free to extend)')
parser.add_argument('--window_overlap', default=64, choices=[32, 64, 128], help='chunks should overlap with this many tokens (feel free to extend)')
parser.add_argument('--batch_size', default=10, help='process this many documents in one batch')
parser.add_argument('--device', default='gpu', choices=['cuda|gpu|cpu|auto'.split('|')], help='run torch on this device')
parser.add_argument('--model', default='gpt2', choices=[
    'openai-gpt',
    'transfo-xl-wt103',
    'xlnet-base-cased',
    'xlnet-large-cased',
    'roberta-base',

    'bert-base-uncased',
    'bert-base-cased',
    'bert-large-uncased',
    'bert-large-cased',
    'bert-base-multilingual-cased',
    'bert-base-multilingual-uncased',

    'google/pegasus-multi_news',
    'google/pegasus-newsroom',

    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',

    'xlnet-base-cased',
    'xlnet-large-cased',

    'roberta-base',
    'roberta-large',
    'distilroberta-base',
    'roberta-base-openai-detector',
    'roberta-large-openai-detector',

], help='which model to use for embedding')

args = parser.parse_args()

args.window_size = int(args.window_size)
args.window_overlap = int(args.window_overlap)
args.model = args.model

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model)

d = "cuda" if args.device in ['cuda', 'gpu', 'auto'] and torch.cuda.is_available() else "cpu"
device = torch.device(d)
if args.model in [
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'openai-gpt',

    'roberta-base',
    'roberta-large',
    'distilroberta-base',
    'roberta-base-openai-detector',
    'roberta-large-openai-detector',
]:
    model.eval()
    tokenizer.add_special_tokens({'pad_token': '.'})

model.to(device=device)


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def overlap(array, chunk_size, overlap_size, add_special=False, padding=False):
    assert overlap_size < chunk_size
    _chunk_size = chunk_size if not add_special else chunk_size - 2
    result = []
    if len(array) > _chunk_size:
        indices = range(0, len(array), _chunk_size - overlap_size)
    else:
        indices = [0]
    for ichunk, start_position in enumerate(indices):
        if start_position != 0 and start_position >= len(array):
            continue
        if add_special:
            chunk = ['[CLS]'] + array[start_position:(start_position + _chunk_size)] + ['[SEP]']
        else:
            chunk = array[start_position:(start_position + _chunk_size)]
        if padding:
            chunk += ['[PAD]'] * (chunk_size - len(chunk))
        result.append(chunk)
    return result


def doc_generator(chunk):
    texts = []
    docids = []
    docs = {}
    for line in chunk:
        doc = json.loads(line)
        text = doc[args.text_field].strip() if doc[args.text_field] else None
        texts.append(text)
        docids.append(doc['id'])
        docs[doc['id']] = doc

    if len(texts) == 0:
        return

    chunks = [overlap(tokenizer.tokenize(text), args.window_size, args.window_overlap) for text in texts]
    batches = split_every(args.batch_size, [tokenizer.convert_tokens_to_string(chunk) for dchunk in chunks for chunk in dchunk])
    rebatches = [docids[docindex] for docindex, dchunk in enumerate(chunks) for chunkindex, _ in enumerate(dchunk)]
    features = [tokenizer(batch, padding=True, return_tensors='pt') for batch in batches]

    state = defaultdict(list)

    for feature in features:
        with torch.no_grad():
            if args.model in [
                'google/pegasus-multi_news',
                'google/pegasus-newsroom'
            ]:
                consider = ['last_hidden_state']
                model_states = model.get_encoder()(**feature.to(device=device))
            elif args.model in [
                'gpt2',
                'gpt2-medium',
                'gpt2-large',
                'gpt2-xl',
                'openai-gpt',
                'xlnet-base-cased',
                'xlnet-large-cased'
            ]:
                consider = ['last_hidden_state']
                model_states = model(**feature.to(device=device))
            else:
                consider = ['last_hidden_state', 'pooler_output']
                model_states = model(**feature.to(device=device))

            for k in consider:
                state[k] += model_states[k].cpu()
            del model_states
            del feature
            torch.cuda.empty_cache()

    doc_embeds = defaultdict(lambda: defaultdict(list))

    for sampe_index, doc_id in enumerate(rebatches):
        for k in consider:
            doc_embeds[doc_id][k].append(state[k][sampe_index])

    for doc_id in doc_embeds.keys():
        embeds = doc_embeds[doc_id]
        doc_out = dict(docs[doc_id])
        for key in consider:
            embeds[key] = np.vstack(embeds[key])
            embeds[key] = np.reshape(embeds[key], (-1, embeds[key].shape[-1]))
            doc_out[f'embedding_{key}_mean'] = embeds[key].mean(axis=0).tolist()
            doc_out[f'embedding_{key}_max'] = embeds[key].max(axis=0).tolist()
            doc_out[f'embedding_{key}_min'] = embeds[key].min(axis=0).tolist()

        yield doc_out


for out_doc in doc_generator(sys.stdin):
    if out_doc:
        print(json.dumps(out_doc), flush=True)
        pass
