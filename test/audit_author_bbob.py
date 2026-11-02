"""Independent full-horizon comparison with the frozen author model AND task.

Runs the two disputed cases and three controls using the published weights.
The reference imports the original GLHFMODEL.py and original main.taskProblem;
it does not call our repaired model or our objective adapter.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from test.compare_bbob import (AUTHOR_ROOT, AUTHOR_REVISION, AuthorBBOB, capture_rng,
                          file_hash, load_model, load_offsets, move_offsets,
                          reset_seed, restore_rng, trial_seed)
from train.train import ROOT, load_checkpoint

CASES = ((30, 8), (100, 3), (30, 1), (100, 1), (30, 9))


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_modules(device):
    package_path = str(AUTHOR_ROOT / 'BBOB_pkg')
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    import BBOB.bbobfunctions as functions
    import BBOB.utils as utilities
    if Path(functions.__file__).resolve() != (AUTHOR_ROOT / 'BBOB_pkg/BBOB/bbobfunctions.py').resolve():
        raise RuntimeError('Wrong BBOB module imported; run this audit in a fresh process')
    for module in (functions, utilities):
        module.DEVICE = device
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value):
                setattr(module, name, value.to(device))
    source = AUTHOR_ROOT / 'GLHF_pkg/GLHF'
    imports = load_source('_audit_original_imports', source / 'imports.py')
    imports.DEVICE = device
    problem = load_source('_audit_original_problem', source / 'problem.py')
    root_utils = load_source('_audit_original_root_utils', AUTHOR_ROOT / 'utils.py')
    package = load_source('_audit_original_package', source / '__init__.py')
    with patch.dict(sys.modules, {'GLHF': package, 'GLHF.imports': imports}):
        utils = load_source('_audit_original_utils', source / 'utils.py')
        with patch.dict(sys.modules, {'GLHF.utils': utils}):
            network = load_source('_audit_original_network', source / 'GLHFMODEL.py')
            with patch.dict(sys.modules, {'GLHF.GLHFMODEL': network, 'GLHF.problem': problem,
                                          'imports': imports, 'utils': root_utils}):
                main = load_source('_audit_original_main', AUTHOR_ROOT / 'main.py')
    return network, main, functions, utilities


def tensor_difference(actual, expected, rtol, atol):
    actual_finite = bool(torch.isfinite(actual).all())
    expected_finite = bool(torch.isfinite(expected).all())
    finite = actual_finite and expected_finite
    exact = torch.equal(actual, expected)
    close = finite and torch.allclose(actual, expected, rtol=rtol, atol=atol)
    delta = (actual - expected).abs()
    maximum = float(delta.max()) if finite else None
    return dict(exact=exact, within_tolerance=bool(close), finite=finite,
                current_finite=actual_finite, original_finite=expected_finite, max_abs=maximum)


def input_summary(population):
    fitness = population[..., 0]
    deviations = fitness.std(-1)
    return dict(finite=bool(torch.isfinite(population).all()),
                fitness_min=best_value(population),
                fitness_std=deviations.cpu().tolist(),
                zero_fitness_std=bool((deviations == 0).any()))


def best_value(population):
    value = float(population[..., 0].min())
    return value if math.isfinite(value) else None


@torch.no_grad()
def audit(device='auto', runs=3, seed=2026, popsize=100, generations=100,
          output='outputs/author_source_audit', rtol=2e-5, atol=2e-6, all_functions=False):
    if runs < 1 or popsize < 2 or generations < 1 or rtol < 0 or atol < 0:
        raise ValueError('Invalid audit budget or tolerance')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    elif device.type != 'cpu':
        raise ValueError('Use cpu or cuda:N')
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; use a new audit output')
    checkpoint = ROOT / 'ckpt/pom_m_release.pth'
    current, info = load_model(checkpoint, 'current', device, popsize)
    network, main, functions, utilities = original_modules(device)
    original = network.GB_GLHF(popsize=popsize).to(device).eval()
    original.load_state_dict(load_checkpoint(checkpoint), strict=True)
    offsets = {d: load_offsets(ROOT / 'test/data' / f'bbobOffsets_dim{d}.pkl') for d in (30, 100)}
    rank_difference = tensor_difference(current.ranks, original.ranks, rtol, atol)
    report = dict(author_revision=AUTHOR_REVISION, device=str(device), torch_version=str(torch.__version__),
                  runs=runs, popsize=popsize, generations=generations, seed=seed, rtol=rtol, atol=atol,
                  checkpoint=info, reference_network_source=network.__file__, reference_task_source=main.__file__,
                  rank_difference=rank_difference,
                  source_sha256={str(path.relative_to(ROOT)): file_hash(path) for path in
                                 (ROOT / 'model/pom.py', AUTHOR_ROOT / 'GLHF_pkg/GLHF/GLHFMODEL.py',
                                  AUTHOR_ROOT / 'main.py', ROOT / 'test/compare_bbob.py', ROOT / 'test/audit_author_bbob.py')},
                  cases=[], all_within_tolerance=rank_difference['within_tolerance'],
                  all_exact=rank_difference['exact'])
    output.mkdir(parents=True, exist_ok=True)
    cases = tuple((dim, fid) for dim in (30, 100) for fid in range(1, 25)) if all_functions else CASES
    report['all_functions'] = all_functions
    for dim, fid in cases:
        for run in range(runs):
            current_task = AuthorBBOB(fid, dim, offsets[dim], device)
            fun = dict(functions.FUNCTIONS[fid], xlb=-10, xub=10)
            utilities.setOffset(fun, move_offsets(offsets[dim][fid], device))
            original_task = main.taskProblem(fun=fun, repaire=True, dim=dim)
            run_seed = trial_seed(seed, dim, fid, run)
            reset_seed(run_seed)
            start_rng = capture_rng(device)
            actual = current_task.random_population(1, popsize)
            current_rng = capture_rng(device)
            restore_rng(start_rng, device)
            expected = original_task.calfitness(original_task.genRandomPop((1, popsize, dim)))[0]
            original_rng = capture_rng(device)
            initial = tensor_difference(actual, expected, rtol, atol)
            row = dict(dim=dim, function=f'F{fid}', run=run + 1, seed=run_seed,
                       initial=initial, first_exact_difference=None, first_tolerance_failure=None,
                       max_abs=dict(population=0.0, mutation=0.0, crossover=0.0),
                       current_trace=[best_value(actual)], original_trace=[best_value(expected)])
            if not initial['exact']:
                row['first_exact_difference'] = dict(generation=0, tensor='population', **initial)
            if not initial['within_tolerance']:
                row['first_tolerance_failure'] = dict(generation=0, tensor='population', **initial)
            for generation in range(1, generations + 1):
                previous_actual, previous_expected = actual, expected
                restore_rng(current_rng, device)
                current_values = current(actual, current_task)
                current_rng = capture_rng(device)
                restore_rng(original_rng, device)
                original_values = original(expected, original_task)
                original_rng = capture_rng(device)
                for label, left, right in zip(('population', 'mutation', 'crossover'), current_values, original_values):
                    difference = tensor_difference(left, right, rtol, atol)
                    if difference['max_abs'] is None:
                        row['max_abs'][label] = None
                    elif row['max_abs'][label] is not None:
                        row['max_abs'][label] = max(row['max_abs'][label], difference['max_abs'])
                    if not difference['exact'] and row['first_exact_difference'] is None:
                        row['first_exact_difference'] = dict(generation=generation, tensor=label, **difference)
                    if not difference['within_tolerance'] and row['first_tolerance_failure'] is None:
                        row['first_tolerance_failure'] = dict(
                            generation=generation, tensor=label, **difference,
                            current_input=input_summary(previous_actual),
                            original_input=input_summary(previous_expected))
                actual, expected = current_values[0], original_values[0]
                row['current_trace'].append(best_value(actual))
                row['original_trace'].append(best_value(expected))
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    break
            row.update(current_final=row['current_trace'][-1], original_final=row['original_trace'][-1],
                       completed_generations=len(row['current_trace']) - 1)
            report['cases'].append(row)
            report['all_exact'] &= row['first_exact_difference'] is None
            report['all_within_tolerance'] &= row['first_tolerance_failure'] is None
            (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(f'D{dim} F{fid:02d} run={run + 1}: current={row["current_final"]} '
                  f'original={row["original_final"]} max_abs={row["max_abs"]}', flush=True)
    print(f'All exact: {report["all_exact"]}; all within tolerance: {report["all_within_tolerance"]}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--popsize', type=int, default=100)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--all-functions', action='store_true', help='Audit all 24 functions at D30/D100 instead of five focus/control cases')
    parser.add_argument('--output', default='outputs/author_source_audit')
    args = parser.parse_args()
    result = audit(**vars(args))
    raise SystemExit(0 if result['all_within_tolerance'] else 1)
