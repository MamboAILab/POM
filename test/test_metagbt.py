"""Small CPU regression for shared-task MetaGBT; all outputs are temporary.

Run: python -m test.test_metagbt
"""
import json
from pathlib import Path
import tempfile

import torch

from train.train import (DEFAULTS, build_model, load_checkpoint, pom_loss, prepare,
                   train, validate_model)


def assert_exact(first, second):
    """Compare model, Adam and RNG state without numerical tolerances."""
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_exact(left, right)
    else:
        assert first == second


def check_equation16():
    previous = torch.tensor([[[2., -1., 1.], [4., 0., 2.],
                              [6., 1., 3.], [8., 2., 4.]]], dtype=torch.float64)
    current = previous.clone()
    current[..., 0] *= .6
    expected = pom_loss(previous, current)
    for factor in (1e-30, 1e30):
        before, after = previous.clone(), current.clone()
        before[..., 0] *= factor
        after[..., 0] *= factor
        actual = pom_loss(before, after)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(expected[1], torch.tensor([-.4], dtype=torch.float64))
    before = torch.zeros_like(previous)
    after = torch.zeros_like(previous, requires_grad=True)
    losses = pom_loss(before, after)
    assert all(torch.isfinite(value).all() for value in losses)
    losses[0].mean().backward()
    assert torch.isfinite(after.grad).all()


def main():
    config = dict(DEFAULTS, taskset='bbob', functions=[1, 9],
                  dtype='float64', variant='release', epochs=2, generations=3,
                  batch_size=2, popsize=12, dim=3, modelsize='vs', device='cpu',
                  threads=1, log_every=100, eval_runs=2, eval_generations=3)
    check_equation16()
    prepare(config)
    initial = build_model(config).state_dict()
    with tempfile.TemporaryDirectory(prefix='pom-metagbt-check-', dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        full_config = dict(config, output=str(root / 'full'))
        model = train(full_config)
        whole = load_checkpoint(root / 'full' / 'last.pt')
        assert whole['epoch'] == 2 and whole['updates'] == 6
        assert len(whole['validation_history']) == 3
        assert all(value.dtype == torch.float64 for value in whole['model'].values()
                   if value.is_floating_point())
        for prefix in ('agen.', 'crgen.'):
            assert any(not torch.equal(value, whole['model'][key])
                       for key, value in initial.items() if key.startswith(prefix)), prefix
        records = [json.loads(line) for line in (root / 'full' / 'train.jsonl').read_text().splitlines()]
        assert len(records) == 6 and [row['update'] for row in records] == list(range(1, 7))
        for row in records:
            assert set(row['task_mean_fitness']) == {'F1', 'F9'}
            assert set(row['task_losses']) == {'F1', 'F9'}
            assert abs(row['loss'] - sum(row['task_losses'].values()) / 2) < 1e-14
            assert torch.isfinite(torch.tensor([row['loss'], *row['task_mean_fitness'].values()])).all()
        assert any(row['mutation_grad_norm'] > 0 for row in records)
        assert any(row['crossover_grad_norm'] > 0 for row in records)

        model.train()
        rng = torch.get_rng_state().clone()
        grad_modes = []
        hook = model.register_forward_pre_hook(lambda *_: grad_modes.append(torch.is_grad_enabled()))
        report = validate_model(model, full_config)
        hook.remove()
        assert grad_modes and not any(grad_modes)
        assert model.training
        assert_exact(rng, torch.get_rng_state())
        assert set(report['best_trace']) == {'F1', 'F9'}
        for fid, trace in report['best_trace'].items():
            values = torch.tensor(trace, dtype=torch.float64)
            assert values.shape == (4, 2) and (values[1:] <= values[:-1]).all(), fid
            torch.testing.assert_close(values[-1], torch.tensor(report['results'][fid]['final_best_per_run'],
                                                               dtype=torch.float64))
        model.eval()
        assert_exact(report, validate_model(model, full_config))
        assert not model.training

        train(dict(config, eval_every=2, output=str(root / 'sparse')))
        sparse = load_checkpoint(root / 'sparse' / 'last.pt')
        for key in ('model', 'optimizer', 'rng_cpu', 'rng_cuda'):
            assert_exact(whole[key], sparse[key])
        assert [item['epoch'] for item in sparse['validation_history']] == [0, 2]

        split_config = dict(config, epochs=1, output=str(root / 'split'))
        train(split_config)
        checkpoint = load_checkpoint(root / 'split' / 'last.pt')
        with (root / 'split' / 'train.jsonl').open('a') as stream:
            stream.write('{"update":4}\n{"update":')
        train(dict(split_config, epochs=2), checkpoint=checkpoint)
        resumed = load_checkpoint(root / 'split' / 'last.pt')
        for key in ('model', 'optimizer', 'rng_cpu', 'rng_cuda', 'validation_history'):
            assert_exact(whole[key], resumed[key])
        assert (root / 'full' / 'train.jsonl').read_text() == (root / 'split' / 'train.jsonl').read_text()
    print('PASS: shared F1/F9 float64 MetaGBT, both networks update, balanced task logs, '
          'six shared Adam updates, validation RNG isolation and monotone no-grad traces, '
          'exact model/Adam/RNG resume and Eq.16 scale invariance')


if __name__ == '__main__':
    main()
