"""Reproduce EXP-01..06 from read-only cached Arrow data; audit saved results.

Run with .venv/Scripts/python.exe scripts/audit_experiments.py from the repo root.
Does not download data or train EXP-07. Writes refreshed notebook outputs and CSVs.
"""
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback

os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
NOTEBOOKS = ROOT / 'notebooks'


def cached_dataset():
    folder = Path.home() / '.cache/huggingface/datasets/guynich___librispeech_asr_test_vad'
    files = sorted(folder.glob('**/*-test.clean-*.arrow'))
    versions = {p.parent for p in files}
    if len(versions) != 1 or not files:
        raise RuntimeError('Expected exactly one cached test.clean Arrow version')
    return concatenate_datasets([Dataset.from_file(str(p)) for p in files])


def display_outputs(capture):
    outputs = []
    if capture.stdout:
        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': capture.stdout.splitlines(True)})
    if capture.stderr:
        outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': capture.stderr.splitlines(True)})
    for out in capture.outputs:
        outputs.append({'output_type': 'display_data', 'data': out.data, 'metadata': out.metadata})
    return outputs


def run_notebook(path, ds):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    shell = InteractiveShell.instance()
    shell.user_ns.clear()
    shell.user_ns.update({'__name__': '__main__', '_audit_dataset': ds})
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    def show_figures(*args, **kwargs):
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
        plt.close('all')
    plt.show = show_figures
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code' or not ''.join(cell['source']).strip():
            continue
        source = ''.join(cell['source'])
        # Read cached data directly, avoiding writes/locks in the global HF cache.
        source = re.sub(r'load_dataset\("guynich/librispeech_asr_test_vad", split="test.clean"(?:, streaming=True)?\)',
                        '_audit_dataset', source)
        # Avoid Windows named-pipe permissions needed by joblib's parallel pool.
        # This changes execution concurrency, not the estimator or seed.
        source = source.replace('n_jobs=-1', 'n_jobs=1')
        with capture_output() as captured:
            result = shell.run_cell(source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path.name}, cell {i}: {captured.stdout}\n{captured.stderr}')
        cell['outputs'] = display_outputs(captured)
        cell['execution_count'] = i + 1
        print(f'{path.name}: cell {i} OK', flush=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def check_csvs():
    findings = []
    for path in sorted(RESULTS.glob('*.csv')):
        # The first two rows are hierarchical column headers in these exports.
        header = [0, 1] if path.name in {'exp02_noise_robustness.csv', 'exp03_adaptive_threshold.csv', 'exp03_summary.csv'} else 0
        frame = pd.read_csv(path, header=header)
        findings.append({'file': path.name, 'rows': len(frame), 'columns': len(frame.columns),
                         'exact_duplicate_rows': int(frame.duplicated().sum())})
        if {'TP', 'TN', 'FP', 'FN'}.issubset(frame.columns):
            pos = frame.TP + frame.FN
            neg = frame.TN + frame.FP
            expected = {'miss_rate': frame.FN / pos.replace(0, np.nan),
                        'false_alarm_rate': frame.FP / neg.replace(0, np.nan),
                        'precision': frame.TP / (frame.TP + frame.FP).replace(0, np.nan),
                        'recall': frame.TP / pos.replace(0, np.nan),
                        'Accuracy': (frame.TP + frame.TN) / (pos + neg).replace(0, np.nan),
                        'balanced_accuracy': 1 - (frame.FN / pos.replace(0, np.nan) + frame.FP / neg.replace(0, np.nan)) / 2,
                        'balanced_acc': 1 - (frame.FN / pos.replace(0, np.nan) + frame.FP / neg.replace(0, np.nan)) / 2}
            for name, values in expected.items():
                if name in frame:
                    valid = values.notna()
                    assert np.allclose(frame.loc[valid, name], values[valid], rtol=1e-8, atol=1e-10), (path.name, name)
            if 'balanced_acc' in frame and (pos == 0).any():
                findings[-1]['single_class_balanced_acc_rows'] = int((pos == 0).sum())
    pd.DataFrame(findings).to_csv(RESULTS / 'audit_inventory.csv', index=False)
    return findings


def main():
    # Keep old exports in memory so reproduction discrepancies cannot go unnoticed.
    before = {p.name: p.read_bytes() for p in RESULTS.glob('*.csv')}
    ds = cached_dataset()
    print('Read-only cached dataset:', len(ds), 'rows', flush=True)
    os.chdir(NOTEBOOKS)
    selected = set(sys.argv[1:]) or {'01', '02', '03', '04', '05', '06'}
    for path in sorted(NOTEBOOKS.glob('*.ipynb')):
        if path.name[:2] in selected:
            run_notebook(path, ds)
    comparison = []
    for name, content in before.items():
        path = RESULTS / name
        header = [0, 1] if name in {'exp02_noise_robustness.csv', 'exp03_adaptive_threshold.csv', 'exp03_summary.csv'} else 0
        old = pd.read_csv(io.BytesIO(content), header=header)
        new = pd.read_csv(path, header=header)
        same_shape = old.shape == new.shape and old.columns.equals(new.columns)
        max_difference = 0.0
        labels_equal = True
        if same_shape:
            for column in old:
                if pd.api.types.is_numeric_dtype(old[column]) and pd.api.types.is_numeric_dtype(new[column]):
                    if not old[column].isna().equals(new[column].isna()):
                        labels_equal = False
                    differences = (old[column] - new[column]).abs().dropna()
                    if len(differences):
                        max_difference = max(max_difference, float(differences.max()))
                else:
                    labels_equal &= old[column].fillna('').equals(new[column].fillna(''))
        comparison.append({'file': name, 'unchanged_bytes': path.read_bytes() == content,
                           'same_shape_and_labels': same_shape and labels_equal,
                           'max_numeric_difference': max_difference if same_shape else np.nan})
    pd.DataFrame(comparison).to_csv(RESULTS / 'audit_reproduction.csv', index=False)
    findings = check_csvs()
    print('AUDIT COMPLETE', len(findings), 'CSV files; count-based metrics validated', flush=True)


if __name__ == '__main__':
    main()
