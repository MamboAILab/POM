"""One CPU integration check: objectives, gradients, resume and inference.

Run: python -m test.test_training  (no pytest required).
"""
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from train.train import (DEFAULTS, TrainingProblem, build_model, evaluate,
                   load_checkpoint, pom_loss, prepare, train)


def main():
    config = dict(DEFAULTS, epochs=2, generations=4, batch_size=2,
                  popsize=12, dim=3, modelsize='vs', device='cpu', log_every=100,
                  dtype='float32', eval_runs=2, eval_generations=3)
    prepare(config)
    problem = TrainingProblem([1, 2, 3, 4, 5], 3, 1, 'cpu')
    problem.offset.zero_()
    x = torch.tensor([1., 2., 3.]).view(1, 1, 3).repeat(5, 1, 1)
    pop, fitness = problem.calfitness(x)
    torch.testing.assert_close(fitness.flatten(), torch.tensor([6., 14., 14., 3., 201.]))
    # Bounds repair must return coordinates consistent with the fitness column.
    pop, fitness = problem.calfitness(x * 1000)
    assert pop[0, 0, 1:].max() <= 10
    torch.testing.assert_close(problem.calfitness(pop[..., 1:])[1], fitness)
    # Eq. 16 includes only decision coordinates in std, and averages per task.
    old = torch.tensor([[[4., 0., 0.], [8., 2., 4.]]])
    new = torch.tensor([[[2., 1., 0.], [4., 3., 4.]]])
    losses, relative, diversity = pom_loss(old, new)
    torch.testing.assert_close(relative, torch.tensor([-0.5], dtype=relative.dtype))
    torch.testing.assert_close(losses, relative - .005 * new[..., 1:].double().std(1).mean(-1))
    # Tied fitness and identical chromosomes used to cause division by zero.
    model = build_model(config)
    constant = problem.calfitness(torch.zeros(5, 12, 3))[0]
    out, _, _ = model(constant, problem)
    loss = pom_loss(constant, out)[0].mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Sorting must move whole [fitness, coordinates] rows and preserve gradients.
    from model.pom import sortIndivBND
    rows = torch.tensor([[[8., 80.], [2., 20.], [5., 50.]]], requires_grad=True)
    sorted_rows = sortIndivBND(rows)
    assert sorted_rows.tolist() == [[[2., 20.], [5., 50.], [8., 80.]]]
    sorted_rows.sum().backward()
    assert rows.grad.eq(1).all()

    # Directly compare the POM adapter against all 24 supplied NumPy functions,
    # including bound repair and alignment of multiple populations.
    from test.evaluate_platemo import PlatEMOBBOB, test as evaluate_bbob
    for fid in range(1, 25):
        local = PlatEMOBBOB(fid, 5)
        x = torch.rand(2, 7, 5) * 14 - 7
        pop, values = local.calfitness(x)
        clipped = np.clip(x.numpy().astype(np.float64), local.problem.lower, local.problem.upper)
        expected = local.problem.cal_obj(clipped.reshape(-1, 5)).reshape(2, 7, 1)
        torch.testing.assert_close(values, torch.as_tensor(expected, dtype=x.dtype), rtol=0, atol=0)
        assert pop.shape == (2, 7, 6) and torch.isfinite(pop).all()
        assert pop[..., 1:].abs().max() <= 5
    for fid in (1, 5, 20, 24):
        local = PlatEMOBBOB(fid, 5)
        optimum = torch.tensor(local.problem.x_opt, dtype=torch.float64).view(1, 1, 5)
        assert abs(local.calfitness(optimum)[1].item()) < 1e-8
    try:
        local.calfitness(torch.zeros(1, 2, 5, requires_grad=True))
    except RuntimeError:
        pass
    else:
        raise AssertionError('NumPy evaluation must not silently sever training gradients')

    with tempfile.TemporaryDirectory(prefix='pom-check-', dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        prepare(config)
        initial = {k: v.clone() for k, v in build_model(config).state_dict().items()}
        whole_config = dict(config, output=str(root / 'whole'))
        trained = train(whole_config)
        state = load_checkpoint(root / 'whole' / 'last.pt')
        rows = [json.loads(line) for line in (root / 'whole' / 'train.jsonl').read_text().splitlines()]
        assert len(rows) == 8 and state['updates'] == 8 and state['epoch'] == 2
        for prefix, metric in [('agen.', 'mutation_grad_norm'), ('crgen.', 'crossover_grad_norm')]:
            assert any(row[metric] > 0 for row in rows), f'{prefix} received no gradient'
            assert all(torch.isfinite(torch.tensor(row[metric])) for row in rows)
            assert any(not torch.equal(initial[k], v) for k, v in trained.state_dict().items()
                       if k.startswith(prefix)), f'{prefix} parameters did not change'
        assert all(torch.isfinite(v).all() for v in trained.state_dict().values())
        split_config = dict(config, output=str(root / 'split'), epochs=1)
        train(split_config)
        split_state = load_checkpoint(root / 'split' / 'last.pt')
        # Simulate a process dying partway through writing the following step.
        with (root / 'split' / 'train.jsonl').open('a') as log:
            log.write('{"update":')
        split_config['epochs'] = 2
        resumed = train(split_config, split_state)
        for k, v in trained.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[k], v, rtol=0, atol=0)
        raw = load_checkpoint(root / 'whole' / 'weights.pth')
        reloaded = build_model(config)
        reloaded.load_state_dict(raw, strict=True)
        report = evaluate(dict(config, output=str(root / 'evaluation')), raw, runs=2, generations=5)
        for values in report['best_trace'].values():
            trace = torch.tensor(values)
            assert (trace[1:] <= trace[:-1]).all(), 'Elitist selection lost a better individual'
        assert report['evaluations_per_run'] == 72
        assert not any(p.grad is not None for p in reloaded.parameters())
        bbob = evaluate_bbob(root / 'whole' / 'last.pt', dim=5, popsize=12, step=3,
                             runs=2, functions=[1, 24], device='cpu',
                             output=root / 'bbob')
        subset = evaluate_bbob(root / 'whole' / 'last.pt', dim=5, popsize=12, step=3,
                               runs=2, functions=[24], device='cpu',
                               output=root / 'bbob_subset')
        assert bbob['results']['F24'] == subset['results']['F24']
        assert bbob['evaluations_total'] == 192
        for result in bbob['results'].values():
            trace = torch.tensor(result['best_trace'])
            assert (trace[:, 1:] <= trace[:, :-1]).all()
    print('PASS: TF1--TF5, loss, finite tied-population backward, whole-row sort, '
          'both networks update, exact CPU resume, raw weight reload, no-grad inference, '
          'local BBOB F1--F24 parity/bounds, reproducible BBOB subsets.')


if __name__ == '__main__':
    main()
