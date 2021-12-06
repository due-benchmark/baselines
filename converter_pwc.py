#!/usr/bin/env python3
"""Scripts that convert T5-model outputs to format that can be directly compared with `documents.jsonl`"""

from glob import glob
import json
import os
from collections import defaultdict
import fire
import re


def main(test_generation, reference_path, outpath):
    data = defaultdict(list)
    seen = set()

    with open(test_generation) as raw_out:
        for line in raw_out:
            if line in seen:
                continue
            seen.add(line)
            line = json.loads(line)
            doc_id = line['doc_id']
            column = re.search(r'the (\w+) column\?$', line['label_name']).group(1)
            values = re.sub(' \|$', '', line['preds']).split(' | ')
            col_values = [(column, val) for val in values]
            data[doc_id].append(col_values)

    with open(reference_path) as expected, open(outpath, 'w+') as output:
        for line in expected:
            line = json.loads(line)
            col_values_list = data[line['name']]
            if len(line['annotations']) > 1:
                print('assumes only one table annotation per document')
                raise
            max_len = max(len(l) for l in col_values_list)
            leaderboard_entries = {'key': 'leaderboard_entry', 'values': []}
            for i in range(max_len):
                leaderboard_entry = {'value': '', 'children': []}
                for col_values in col_values_list:
                    if i >= len(col_values):
                        continue
                    column, value = col_values[i]
                    entry_value = {'key': column, 'values': [{'value': value}]}
                    leaderboard_entry['children'].append(entry_value)
                leaderboard_entries['values'].append(leaderboard_entry)
            # for col_valueskey, val in

            ans_doc = {'name': line['name'], 'annotations': [leaderboard_entries]}

            output.write(json.dumps(ans_doc) + '\n')


if __name__ == "__main__":
    fire.Fire(main)

