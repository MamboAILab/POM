"""Evaluate POM using the user's platemo_bbob_problems/BBOB_F1--F24.py."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from train.train import (DEFAULTS, MODEL_SIZES, ROOT, build_model, config_for_weights,
                   load_checkpoint, prepare)


class PlatEMOBBOB:
    """Adapt the actual NumPy problem files to POM's [B,N,D+1] interface.

    This is a black-box evaluation interface. Direct BBOB training uses the
    differentiable adapter in train/problems.py through train/train_bbob.py.
    """
    def __init__(self, fid, dim, device='cpu', dtype=torch.float32):
        if fid not in range(1, 25) or dim < 2:
            raise ValueError('Require function 1..24 and dimension >= 2')
        self.fid = fid
        self.dim = dim
        module = importlib.import_module(f'platemo_bbob_problems.BBOB_F{fid}')
        self.problem = getattr(module, f'BBOB_F{fid}')(D=dim)
        self.lower = torch.as_tensor(self.problem.lower, dtype=dtype, device=device)
        self.upper = torch.as_tensor(self.problem.upper, dtype=dtype, device=device)

    def random_population(self, batch_size, popsize):
        x = torch.rand(batch_size, popsize, self.dim, device=self.lower.device, dtype=self.lower.dtype)
        return self.calfitness(self.lower + x * (self.upper - self.lower))[0]

    def calfitness(self, x):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError('Expected [batch, population_size, dimension]')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('NumPy BBOB is for no-grad evaluation; use train/train_bbob.py for differentiable BBOB training')
        x = x.clamp(min=self.lower, max=self.upper)
        # Evaluate the provided code itself in NumPy float64; do not replace its
        # rotations, shifts, or formulas with another BBOB implementation.
        flat = x.detach().cpu().numpy().reshape(-1, self.dim).astype(np.float64)
        values = self.problem.cal_obj(flat)
        fitness = torch.as_tensor(values, dtype=x.dtype, device=x.device).reshape(*x.shape[:-1], 1)
        if not torch.isfinite(fitness).all():
            raise FloatingPointError(f'Nonfinite fitness in local BBOB F{self.fid}')
        return torch.cat((fitness, x), dim=-1), fitness


@torch.no_grad()
def test(ckpt, dim=30, popsize=100, step=100, runs=3, functions=None,
         modelsize='m', device='auto', seed=2026, threads=1,
         output='outputs/platemo_bbob_eval'):
    state = load_checkpoint(ckpt)
    if 'model' in state:
        modelsize = state['config']['modelsize']
        state = state['model']
    config = prepare(config_for_weights(dict(DEFAULTS, modelsize=modelsize, popsize=popsize,
                          device=device, seed=seed, threads=threads), state))
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    # Model initialization/loading must not influence the evaluation seed.
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    functions = list(functions) if functions is not None else list(range(1, 25))
    results = {}
    for fid in functions:
        problem = PlatEMOBBOB(fid, dim, config['device'], getattr(torch, config['dtype']))
        traces = []
        run_seeds = []
        for run in range(runs):
            # Each function/run is reproducible even when evaluating a subset.
            run_seed = seed + fid * 1000 + run
            run_seeds.append(run_seed)
            torch.manual_seed(run_seed)
            pop = problem.random_population(1, popsize)
            trace = [pop[..., 0].min().item()]
            for _ in range(step):
                pop, _, _ = model(pop, problem)
                if not torch.isfinite(pop).all():
                    raise FloatingPointError(f'Nonfinite population in F{fid}')
                trace.append(pop[..., 0].min().item())
            traces.append(trace)
        final = np.asarray(traces)[:, -1]
        results[f'F{fid}'] = dict(initial_best_mean=float(np.mean(np.asarray(traces)[:, 0])),
                                  final_best_mean=float(final.mean()), final_best_std=float(final.std()),
                                  final_best_per_run=final.tolist(), run_seeds=run_seeds,
                                  lower=problem.lower.cpu().tolist(), upper=problem.upper.cpu().tolist(),
                                  best_trace=traces)
        print(f'F{fid}: {final.mean():.6E} ({final.std():.6E})', flush=True)
    report = dict(benchmark='platemo_bbob_problems', source_directory=str(ROOT / 'platemo_bbob_problems'),
                  dim=dim, popsize=popsize, generations=step, runs=runs, seed=seed,
                  modelsize=config['modelsize'], variant=config['variant'], dtype=config['dtype'],
                  device=config['device'], checkpoint=str(ckpt),
                  evaluations_per_run=popsize * (step + 1),
                  evaluations_total=len(functions) * runs * popsize * (step + 1),
                  results=results)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {output / "results.json"}', flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dim', type=int, default=30)
    parser.add_argument('-n', '--popsize', type=int, default=100)
    parser.add_argument('-t', '--step', type=int, default=100)
    parser.add_argument('-p', '--ckpt', default=str(ROOT / 'ckpt/pom_m_release.pth'))
    parser.add_argument('-s', '--modelsize', choices=MODEL_SIZES, default='m')
    parser.add_argument('--functions', nargs='+', type=int, choices=range(1, 25))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='outputs/platemo_bbob_eval')
    args = parser.parse_args()
    if args.dim < 2 or args.popsize < 2 or args.step < 0 or args.runs < 1 or args.threads < 1:
        parser.error('dim/popsize >= 2, step >= 0, runs/threads >= 1 required')
    if args.functions and len(set(args.functions)) != len(args.functions):
        parser.error('--functions must not contain duplicates')
    return args


if __name__ == '__main__':
    args = parse_args()
    test(ckpt=args.ckpt, dim=args.dim, popsize=args.popsize, step=args.step,
         runs=args.runs, functions=args.functions, modelsize=args.modelsize,
         device=args.device, seed=args.seed, threads=args.threads,
         output=args.output)
