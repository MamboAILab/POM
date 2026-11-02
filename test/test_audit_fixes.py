"""Regressions for the independent 2026-10-30 audit findings."""
import json
from pathlib import Path
import tempfile

import torch

from train.train import DEFAULTS, ROOT, TRAINING_PROTOCOL, build_model, pom_loss, prepare, train
from train.train_bbob import DEFAULTS as BBOB_DEFAULTS, read_reference_first_run, run_baseline
from train.problems import BBOBTrainingProblem
from model.pom import finite_source_norm


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    x = torch.randn(3, 100, 30, requires_grad=True)
    torch.testing.assert_close(finite_source_norm(x), torch.sqrt(torch.sum(x**2, -1, keepdim=True)), rtol=0, atol=0)
    zero = torch.zeros(2, 12, 3, requires_grad=True)
    finite_source_norm(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()

    # This actual POM graph previously had a finite loss but 12 nonfinite
    # parameter gradients. Double reduction avoids the premature reciprocal.
    config = dict(DEFAULTS, device='cpu', dtype='float32', modelsize='vs', dim=3, popsize=12, batch_size=2)
    prepare(config)
    model = build_model(config)
    problem = BBOBTrainingProblem(1, 3, 2)
    previous = problem.calfitness(torch.randn(2, 12, 3) * 1e-20)[0].detach()
    current = model(previous, problem)[0]
    loss = pom_loss(previous, current)[0].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    try:
        before = torch.full((1, 12, 4), 1e-42)
        pom_loss(before, before.clone().requires_grad_())
    except FloatingPointError as error:
        assert 'float64' in str(error)
    else:
        raise AssertionError('Unrepresentable float32 gradients need an actionable precision error')

    with tempfile.TemporaryDirectory(prefix='pom-audit-fixes-', dir=ROOT) as directory:
        root = Path(directory)
        legacy = dict(BBOB_DEFAULTS, device='cpu', dtype='float64', functions=[1],
                      epochs=1, generations=1, batch_size=2, popsize=12, dim=3,
                      modelsize='vs', eval_batch_size=2, eval_generations=1,
                      output=str(root / 'double_training'))
        run_baseline(legacy)
        state = torch.load(root / 'double_training/F01/run_01/last.pt', weights_only=True)
        assert state['config']['dtype'] == 'float64'
        assert state['config']['protocol_version'] == TRAINING_PROTOCOL
        assert all(value.dtype == torch.float64 for value in state['model'].values())

        reference = root / 'reference/F01/run_01'
        reference.mkdir(parents=True)
        config = dict(BBOB_DEFAULTS, epochs=1, functions=[1], runs=2,
                      reference_first_run=str(root / 'reference'))
        saved = dict(config, fid=1, run_index=1, run_seed=config['seed'],
                     eval_seed=config['seed'] + 10001000)
        (reference / 'fitness.csv').write_text('epoch,update,fitness_best_of_batch,fitness_mean_best,fitness_std_best\n'
                                             '0,0,1,2,0.5\n1,100,0.5,1,0.2\n')
        for key, value in (('variant', 'paper'), ('dtype', 'float64'), ('taskset', 'tf'),
                           ('protocol_version', 'older-method')):
            (reference / 'config.json').write_text(json.dumps(dict(saved, **{key: value})))
            try:
                read_reference_first_run(config, 1)
            except ValueError as error:
                assert key in str(error)
            else:
                raise AssertionError(f'Incompatible reference accepted: {key}')
        saved.pop('protocol_version')
        (reference / 'config.json').write_text(json.dumps(saved))
        try:
            read_reference_first_run(config, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown legacy loss protocol accepted')
        try:
            train(dict(DEFAULTS, output=str(root / 'bad_resume')), checkpoint={'config': {}})
        except ValueError as error:
            assert 'protocol' in str(error)
        else:
            raise AssertionError('An unknown training protocol was silently resumed')
        assert not (root / 'bad_resume').exists()
    print('PASS: exact source norm/zero derivative, tiny-fitness backward, double training, incompatible-reference and resume guards')


if __name__ == '__main__':
    main()
