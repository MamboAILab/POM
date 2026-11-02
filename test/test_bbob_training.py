"""Small CPU check for direct, independent BBOB training (no pytest needed).

Run: python -m test.test_bbob_training
All trial outputs live in an automatically cleaned temporary directory.
"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from test.evaluate_platemo import PlatEMOBBOB
from train.problems import BBOBTrainingProblem, TorchBBOB
from train.train import build_model, load_checkpoint, pom_loss, prepare
from train.train_bbob import DEFAULTS, run_baseline


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def same_model(first, second):
    assert first['model'].keys() == second['model'].keys()
    for key, value in first['model'].items():
        torch.testing.assert_close(second['model'][key], value, rtol=0, atol=0)
    torch.testing.assert_close(first['rng_cpu'], second['rng_cpu'], rtol=0, atol=0)


def main():
    config = dict(DEFAULTS, epochs=2, generations=2, batch_size=2,
                  popsize=12, dim=5, modelsize='vs', device='cpu', threads=1,
                  functions=[1, 9], runs=1, eval_generations=3,
                  eval_batch_size=2, eval_every=1, log_every=100)
    prepare(config)
    # Match the float32 training interface to the original NumPy formulas.
    for fid in range(1, 25):
        reference = PlatEMOBBOB(fid, config['dim'])
        objective = TorchBBOB(fid, config['dim'], 'cpu')
        x = torch.rand(2, 7, config['dim']) * 14 - 7
        clipped = np.clip(x.numpy().astype(np.float64), reference.problem.lower, reference.problem.upper)
        expected = reference.problem.cal_obj(clipped.reshape(-1, config['dim'])).reshape(2, 7)
        actual = objective.calfitness(x)
        assert actual.shape == (2, 7), f'Wrong fitness shape for F{fid}'
        tolerance = dict(rtol=3e-3, atol=2e-3) if fid in (16, 19, 23) else dict(rtol=3e-5, atol=2e-4)
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=x.dtype), **tolerance,
                                   msg=lambda message: f'F{fid}: {message}')
        x = x.requires_grad_(True)
        objective.calfitness(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), f'F{fid} objective backward'
        # Exercise the real mutation/crossover/selection graph, not just f(x).
        problem = BBOBTrainingProblem(fid, config['dim'], config['batch_size'], 'cpu')
        pop = problem.random_population(config['popsize'])
        assert pop.shape == (2, 12, 6)
        model = build_model(config)
        offspring, _, _ = model(pop.detach(), problem)
        loss = pom_loss(pop, offspring, config['diversity_weight'])[0].mean()
        loss.backward()
        assert torch.isfinite(offspring).all() and torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()), f'F{fid} POM backward'
        repaired, fitness = problem.calfitness(pop[..., 1:] * 100)
        assert repaired[..., 1:].abs().max() <= 5
        torch.testing.assert_close(problem.calfitness(repaired[..., 1:])[1], fitness)

    with tempfile.TemporaryDirectory(prefix='pom-bbob-check-', dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        full_config = dict(config, output=str(root / 'full'))
        report = run_baseline(full_config)
        assert isinstance(report, dict) and 'results' in report
        assert (root / 'full' / 'summary.csv').is_file()
        assert len(read_csv(root / 'full' / 'summary.csv')) == 2
        whole = {}
        for fid in config['functions']:
            run_dir = root / 'full' / f'F{fid:02d}' / 'run_01'
            assert all((run_dir / name).is_file() for name in
                       ('config.json', 'train.jsonl', 'fitness.csv', 'last.pt', 'weights.pth'))
            state = whole[fid] = load_checkpoint(run_dir / 'last.pt')
            assert state['fid'] == fid and state['run_index'] == 1
            assert state['epoch'] == 2 and state['updates'] == 4
            assert all(torch.isfinite(value).all() for value in state['model'].values())
            rows = read_csv(run_dir / 'fitness.csv')
            assert [int(row['epoch']) for row in rows] == [0, 1, 2]
            assert all(float(row['fitness_best_of_batch']) <= float(row['fitness_mean_best'])
                       for row in rows)
            records = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().splitlines()]
            assert len(records) == 4
        # The optimizer must change both learned components on an actual BBOB task.
        prepare(config)
        initial = build_model(config).state_dict()
        for prefix in ('agen.', 'crgen.'):
            assert any(not torch.equal(value, whole[1]['model'][key])
                       for key, value in initial.items() if key.startswith(prefix)), prefix
        try:
            run_baseline(full_config)
        except (FileExistsError, ValueError):
            pass
        else:
            raise AssertionError('Existing training outputs must not be overwritten')
        same_model(whole[1], load_checkpoint(root / 'full' / 'F01' / 'run_01' / 'last.pt'))

        # F9 is unchanged when trained by itself: no weights/RNG leak from F1.
        subset_config = dict(config, functions=[9], output=str(root / 'subset'))
        run_baseline(subset_config)
        same_model(whole[9], load_checkpoint(root / 'subset' / 'F09' / 'run_01' / 'last.pt'))
        assert read_csv(root / 'full' / 'F09' / 'run_01' / 'fitness.csv') == read_csv(
            root / 'subset' / 'F09' / 'run_01' / 'fitness.csv')

        # More frequent fixed evaluation must not consume training randomness.
        run_baseline(dict(config, functions=[1], eval_every=2, output=str(root / 'sparse_eval')))
        same_model(whole[1], load_checkpoint(root / 'sparse_eval' / 'F01' / 'run_01' / 'last.pt'))

        # Epoch-boundary resume reproduces full training exactly on CPU.
        split_config = dict(config, functions=[1], epochs=1, output=str(root / 'split'))
        run_baseline(split_config)
        split_log = root / 'split' / 'F01' / 'run_01' / 'train.jsonl'
        # An interrupted following epoch may leave committed-looking and partial lines.
        with split_log.open('a', encoding='utf-8') as stream:
            stream.write('{"update":3}\n{"update":')
        split_config['epochs'] = 2
        run_baseline(split_config, resume=True)
        same_model(whole[1], load_checkpoint(root / 'split' / 'F01' / 'run_01' / 'last.pt'))
        assert read_csv(root / 'full' / 'F01' / 'run_01' / 'fitness.csv') == read_csv(
            root / 'split' / 'F01' / 'run_01' / 'fitness.csv')
        assert len(read_csv(root / 'split' / 'F01' / 'run_01' / 'fitness.csv')) == 3
        records = [json.loads(line) for line in split_log.read_text().splitlines()]
        assert [row['update'] for row in records] == [1, 2, 3, 4]

        # Reuse only run 1's earlier curves; four NEW runs must match a fresh five-run experiment.
        reference_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (root / 'full').rglob('*') if path.is_file()}
        pooled_config = dict(config, epochs=1, runs=5, reference_first_run=str(root / 'full'),
                             output=str(root / 'pooled'))
        pooled = run_baseline(pooled_config)
        fresh = run_baseline(dict(config, epochs=1, runs=5, output=str(root / 'fresh_five')))
        assert pooled['results'] == fresh['results']
        assert all(result['run_seeds'] == [2026, 3026, 4026, 5026, 6026]
                   for result in pooled['results'].values())
        assert read_csv(root / 'pooled' / 'fitness_by_epoch.csv') == read_csv(
            root / 'fresh_five' / 'fitness_by_epoch.csv')
        for fid in config['functions']:
            assert not (root / 'pooled' / f'F{fid:02d}' / 'run_01').exists()
            individual = [[row for row in read_csv(root / 'full' / f'F{fid:02d}' /
                                                   'run_01' / 'fitness.csv') if int(row['epoch']) <= 1]]
            for index in range(2, 6):
                subpath = Path(f'F{fid:02d}') / f'run_{index:02d}'
                same_model(load_checkpoint(root / 'pooled' / subpath / 'last.pt'),
                           load_checkpoint(root / 'fresh_five' / subpath / 'last.pt'))
                individual.append(read_csv(root / 'pooled' / subpath / 'fitness.csv'))
            expected_final = np.mean([float(rows[-1]['fitness_best_of_batch']) for rows in individual])
            assert pooled['results'][f'F{fid}']['final_mean'] == expected_final
        run_baseline(pooled_config, resume=True)
        for path, digest in reference_hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        # Incompatible settings, missing prefix epochs and self-reference fail before output writes.
        invalid = [dict(pooled_config, lr=config['lr'] * 2, output=str(root / 'bad_lr')),
                   dict(pooled_config, epochs=3, output=str(root / 'too_long')),
                   dict(pooled_config, output=str(root / 'full'))]
        for candidate in invalid:
            try:
                run_baseline(candidate)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid reference was accepted')
        assert not (root / 'bad_lr').exists() and not (root / 'too_long').exists()
        for path, digest in reference_hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    print('PASS: all 24 BBOB formulas/bounds/backward, actual mutation/crossover updates, '
          'independent reproducible tasks, fixed evaluation preserves training RNG, '
          'exact CPU resume, output overwrite protection, baseline curves and summary, '
          'read-only earlier run reuse, four independent new runs and exact five-run mean.')


if __name__ == '__main__':
    main()
