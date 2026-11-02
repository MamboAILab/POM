"""Small paired-checkpoint comparison regression; outputs are temporary.

Run: python -m test.test_compare_bbob
"""
import csv
from collections import defaultdict
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from test.compare_bbob import ROOT, run_comparison
from train.train import load_checkpoint


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check_cuda_preflight(reference, root):
    class StopBeforeCudaLoad(Exception):
        pass

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        for requested, index in (('auto', 2), ('cuda', 2), ('cuda:1', 1)):
            output = root / f'cuda_preflight_{requested.replace(":", "_")}'
            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'current_device', return_value=2), \
                    patch.object(torch.cuda, 'set_device', side_effect=torch.cuda._utils._get_device_index) as set_device, \
                    patch('test.compare_bbob.load_model', side_effect=StopBeforeCudaLoad) as load_model:
                try:
                    run_comparison(reference, dims=(5,), functions=[1], benchmark='local',
                                   device=requested, runs=1, popsize=8, generations=0,
                                   output=output)
                except StopBeforeCudaLoad:
                    pass
                else:
                    raise AssertionError('CUDA preflight did not reach model loading')
                explicit = torch.device('cuda', index)
                set_device.assert_called_once_with(explicit)
                assert load_model.call_args.args[2] == explicit
                assert not output.exists()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def check_results(output, report, identical=True):
    rows = read_csv(output / 'per_run.csv')
    models = [item['id'] for item in report['models']]
    count = len(report['dims']) * len(report['functions']) * report['runs'] * len(models)
    budget = report['popsize'] * (report['generations'] + 1)
    assert len(rows) == count
    assert report['evaluations_per_trial'] == budget
    assert report['evaluations_total'] == count * budget
    grouped, finals = defaultdict(dict), defaultdict(list)
    for row in rows:
        key = (int(row['dim']), row['function'], int(row['run']))
        grouped[key][row['model']] = row
        finals[(key[0], key[1], row['model'])].append(float(row['final_best']))
        assert int(row['evaluations']) == budget
        assert len(row['initial_population_sha256']) == 64
        assert float(row['final_best']) <= float(row['initial_best'])
        np.testing.assert_allclose(float(row['fitness_minus_fopt']),
                                   float(row['final_best']) - float(row['fopt']), rtol=0, atol=0)
    for pair in grouped.values():
        assert set(pair) == set(models)
        reference = pair['reference']
        for row in pair.values():
            assert row['initial_population_sha256'] == reference['initial_population_sha256']
            assert row['initial_best'] == reference['initial_best']
            assert row['seed'] == reference['seed']
            if identical:
                assert row['final_best'] == reference['final_best']
    summaries = read_csv(output / 'comparison.csv')
    assert len(summaries) == len(report['dims']) * len(report['functions']) * max(1, len(models) - 1)
    for row in summaries:
        key = (int(row['dim']), row['function'])
        for model, prefix in [('reference', 'reference'), *([(row['candidate'], 'candidate')] if row['candidate'] else [])]:
            values = np.asarray(finals[(*key, model)])
            np.testing.assert_allclose(float(row[f'{prefix}_mean']), values.mean(), rtol=1e-14)
            np.testing.assert_allclose(float(row[f'{prefix}_std']), values.std(ddof=0), rtol=1e-14)
        if identical and row['candidate']:
            assert float(row['mean_difference']) == 0
            assert int(row['paired_ties']) == report['runs']
            assert int(row['paired_wins']) == int(row['paired_losses']) == 0
    trace_rows = read_csv(output / 'traces.csv')
    assert len(trace_rows) == count * (report['generations'] + 1)
    for row in trace_rows:
        assert int(row['evaluations']) == report['popsize'] * (int(row['generation']) + 1)
    with np.load(output / 'curves.npz') as curves, np.load(output / 'best_solutions.npz') as solutions:
        for dim in report['dims']:
            for fid in report['functions']:
                reference_key = f'D{dim}_F{fid:02d}_reference'
                reference = curves[reference_key]
                assert reference.shape == (report['runs'], report['generations'] + 1)
                assert (reference[:, 1:] <= reference[:, :-1]).all()
                for model in models:
                    key = f'D{dim}_F{fid:02d}_{model}'
                    assert solutions[key].shape == (report['runs'], dim)
                    np.testing.assert_array_equal(curves[key][:, -1], finals[(dim, f'F{fid}', model)])
                    if identical:
                        np.testing.assert_array_equal(curves[key], reference)
                        np.testing.assert_array_equal(solutions[key], solutions[reference_key])
    return rows


def main():
    official = ROOT / 'ckpt' / 'pom_m_release.pth'
    common = dict(reference=official, device='cpu', popsize=8, generations=2,
                  runs=2, threads=1)
    with tempfile.TemporaryDirectory(prefix='pom-comparison-check-', dir=ROOT) as directory:
        root = Path(directory)
        check_cuda_preflight(official, root)
        output = root / 'all_author'
        report = run_comparison(**common, candidates=[official], dims=(30, 100), output=output)
        whole = check_results(output, report)
        assert report['benchmark'] == 'author'
        assert set(report['offset_sha256']) == {'30', '100'}
        assert len(report['source_sha256']) >= 4

        # F20 consumes objective RNG; selecting it alone must reproduce the
        # same populations, fitness and solver trajectories as the full suite.
        subset_output = root / 'f20_subset'
        subset = run_comparison(**common, candidates=[official], dims=(30, 100),
                                functions=[20], output=subset_output)
        subset_rows = check_results(subset_output, subset)
        assert subset_rows == [row for row in whole if row['function'] == 'F20']
        with np.load(output / 'curves.npz') as full_curves, np.load(subset_output / 'curves.npz') as subset_curves:
            for key in subset_curves.files:
                np.testing.assert_array_equal(subset_curves[key], full_curves[key])

        # Save the same tensor values as a double raw state_dict, last.pt,
        # and best.pt. All three must evaluate exactly like the float32 source.
        weights64 = {name: value.double() for name, value in load_checkpoint(official).items()}
        raw64, last64, best64 = [root / name for name in ('weights64.pth', 'last.pt', 'best.pt')]
        torch.save(weights64, raw64)
        torch.save(dict(model=weights64, optimizer={}, epoch=17,
                        config=dict(dtype='float64')), last64)
        torch.save(dict(model=weights64, epoch=11, validation={}), best64)
        wrapper_output = root / 'checkpoint_formats'
        wrappers = run_comparison(**common, candidates=[raw64, last64, best64],
                                  dims=(30,), functions=[1, 20, 24], output=wrapper_output)
        check_results(wrapper_output, wrappers)
        assert [model['saved_dtype'] for model in wrappers['models']] == ['float32', 'float64', 'float64', 'float64']
        assert all(model['evaluation_dtype'] == 'float32' for model in wrappers['models'])
        assert [model['epoch'] for model in wrappers['models']] == [None, None, 17, 11]

        baseline_output = root / 'reference_only'
        baseline = run_comparison(**common, dims=(30,), functions=[20], output=baseline_output)
        baseline_rows = check_results(baseline_output, baseline)
        assert len(baseline['models']) == 1
        assert baseline_rows == [row for row in whole if row['dim'] == '30'
                                 and row['function'] == 'F20' and row['model'] == 'reference']
        saved = (baseline_output / 'results.json').read_bytes()
        try:
            run_comparison(**common, dims=(30,), functions=[20], output=baseline_output)
        except ValueError as error:
            assert 'not empty' in str(error)
        else:
            raise AssertionError('Comparison overwrote existing results')
        assert saved == (baseline_output / 'results.json').read_bytes()

        # An actual parameter change must affect a result. Identical-model
        # checks alone would miss accidentally evaluating the reference twice.
        altered = {name: value.float().clone() for name, value in weights64.items()}
        for name, value in altered.items():
            if name.startswith('agen.'):
                value.zero_()
        altered_path = root / 'altered.pth'
        torch.save(altered, altered_path)
        local_output = root / 'local'
        local = run_comparison(**common, candidates=[official, altered_path], benchmark='local',
                               dims=(5,), functions=[1, 24], output=local_output)
        check_results(local_output, local, identical=False)
        with np.load(local_output / 'curves.npz') as curves:
            for fid in (1, 24):
                np.testing.assert_array_equal(curves[f'D5_F{fid:02d}_reference'],
                                              curves[f'D5_F{fid:02d}_candidate_01'])
            assert any(not np.array_equal(curves[f'D5_F{fid:02d}_reference'],
                                          curves[f'D5_F{fid:02d}_candidate_02']) for fid in (1, 24))
        assert local['benchmark'] == 'local' and not local['offset_sha256']
    print('PASS: author F1--F24 D30/D100 identical-weight paired trajectories, '
          'initial hashes, mean/std/budgets, F20 subset RNG reproducibility, '
          'raw/last/best double-checkpoint float32 evaluation, reference-only '
          'and overwrite protection, local F1/F24 smoke, changed-weight sensitivity '
          'and explicit CUDA device preflight')


if __name__ == '__main__':
    main()
