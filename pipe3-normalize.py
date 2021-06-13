#!/usr/bin/env python
import json
from collections import defaultdict

import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')

parser.add_argument('--repeat_delimiter', default="\0", help='stream data this many loops')
parser.add_argument('--size', type=int, help='expected number of documents in first block')
parser.add_argument('--type', default='sigmoid', choices=['amplitude', 'sigmoid', 'none'], help='normalisation method to be used')
args = parser.parse_args()

fields = defaultdict(dict)
for line in sys.stdin:
    if line.startswith(args.repeat_delimiter):
        break
    doc = json.loads(line)
    print(len(line), line[0:10], line[-10:-1])
    for field_name, field_value in doc.items():
        if 'embedding' in field_name and len(field_value):
            if 'stacked' not in fields[field_name]:
                fields[field_name]['stacked'] = np.zeros((args.size, len(field_value)))
                fields[field_name]['stacked'][0, :] = field_value
                fields[field_name]['size'] = 1
            else:
                fields[field_name]['stacked'][fields[field_name]['size'], :] = field_value
                fields[field_name]['size'] += 1

for field in fields.keys():
    print(f"{field} min:{fields[field]['stacked'].min(axis=0)[0:10]} mean:{fields[field]['stacked'].mean(axis=0)[0:10]} max:{fields[field]['stacked'].max(axis=0)[0:10]} std:{fields[field]['stacked'].std(axis=0)[0:10]}", file=sys.stderr)

    fields[field]['min'] = fields[field]['stacked'].min(axis=0)
    fields[field]['mean'] = fields[field]['stacked'].mean(axis=0)
    fields[field]['max'] = fields[field]['stacked'].max(axis=0)
    fields[field]['std'] = fields[field]['stacked'].std(axis=0)
    del fields[field]['stacked']

for line in sys.stdin:
    if line.startswith(args.repeat_delimiter):
        break
    doc = json.loads(line)
    for field_name, field_value in doc.items():
        if 'embedding' in field_name and len(field_value):
            if args.type == 'amplitude':
                doc[field_name] = (
                        (np.array(doc[field_name]) - np.array(fields[field_name]['min'])) /
                        (np.array(fields[field_name]['max']) - np.array(fields[field_name]['min']))
                ).tolist()
            elif args.type == 'sigmoid':
                doc[field_name] = (1 / (1 + np.exp(-(np.array(doc[field_name]) - fields[field_name]['mean']) / fields[field_name]['std'] * 2))
                                   ).tolist()
            elif args.type == 'none':
                doc[field_name] = doc[field_name]
            else:
                raise ValueError()
    line_out = json.dumps(doc)
    print(line_out, flush=True)
