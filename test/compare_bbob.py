"""Compare released and trained POM weights on identical BBOB trials.

Default: author's frozen BBOB code, supplied D30 offsets, bounds [-10,10],
one population of 100, 100 generations, three paired trials, float32.
Use --benchmark local explicitly for the existing PlatEMO problem files.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from test.evaluate_platemo import PlatEMOBBOB
from train.train import DEFAULTS, ROOT, build_model, config_for_weights, load_checkpoint

AUTHOR_REVISION = '41ee8804efd5efa237bb89d95e19fa4faa0cde0b'
AUTHOR_ROOT = ROOT / 'test/reference'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def storage_on_cpu(data):
    try:
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except TypeError:  # torch 1.12
        return torch.load(io.BytesIO(data), map_location='cpu')


class OffsetUnpickler(pickle.Unpickler):
    """The author's ordinary pickle embeds torch storages saved on CUDA."""
    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return storage_on_cpu
        return super().find_class(module, name)


def load_offsets(path):
    with Path(path).open('rb') as stream:
        return OffsetUnpickler(stream).load()


def move_offsets(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    if isinstance(value, dict):
        return {key: move_offsets(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_offsets(item, device) for item in value)
    return value


class AuthorBBOB:
    """Call the published objective code itself without rewriting its formulas."""
    def __init__(self, fid, dim, offsets, device):
        package_path = str(AUTHOR_ROOT / 'BBOB_pkg')
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        import BBOB.bbobfunctions as functions
        import BBOB.utils as utilities
        expected = AUTHOR_ROOT / 'BBOB_pkg' / 'BBOB' / 'bbobfunctions.py'
        if Path(functions.__file__).resolve() != expected.resolve():
            raise RuntimeError('Another BBOB package is already imported; run compare_bbob.py in a fresh process')
        for module in (functions, utilities):
            module.DEVICE = device
            for name, value in list(vars(module).items()):
                if torch.is_tensor(value):
                    setattr(module, name, value.to(device=device, dtype=torch.float32))
        self.fid, self.dim, self.device = fid, dim, device
        self.lower, self.upper = -10.0, 10.0
        self.fun = dict(functions.FUNCTIONS[fid], xlb=self.lower, xub=self.upper)
        utilities.setOffset(self.fun, move_offsets(offsets[fid], device))
        self.fopt = float(self.fun['fopt'])
        self.objective = utilities.getFitness

    def random_population(self, batch_size, popsize):
        if batch_size != 1:
            raise ValueError('Author protocol evaluates one population per trial')
        x = torch.rand(1, popsize, self.dim, device=self.device, dtype=torch.float32) * 20 - 10
        return self.calfitness(x)[0]

    def calfitness(self, x):
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != self.dim:
            raise ValueError('Expected [1, population_size, dimension]')
        x = x.clamp(self.lower, self.upper)
        # F20 in the author source is stochastic and population-coupled. Keep
        # that behavior; merging repeats into a batch would change the task.
        fitness = self.objective(x.reshape(-1, self.dim), self.fun).reshape(1, x.shape[1], 1)
        if not torch.isfinite(fitness).all():
            raise FloatingPointError(f'Nonfinite author BBOB F{self.fid}')
        return torch.cat((fitness, x), -1), fitness


def load_model(path, model_id, device, popsize):
    raw = load_checkpoint(path)
    weights = raw['model'] if 'model' in raw else raw
    config = config_for_weights(dict(DEFAULTS, device=str(device), popsize=popsize, init='default'), weights)
    original_dtype = config['dtype']
    # Both models must use the same evaluation precision, including when a
    # training checkpoint was saved in double. Author matrices are float32.
    config['dtype'] = 'float32'
    model = build_model(config)
    model.load_state_dict(weights, strict=True)
    model.eval()
    info = dict(id=model_id, checkpoint=str(Path(path).resolve()), sha256=file_hash(path),
                modelsize=config['modelsize'], saved_dtype=original_dtype, evaluation_dtype='float32',
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                epoch=raw.get('epoch') if 'model' in raw else None)
    return model, info


def trial_seed(seed, dim, fid, run):
    return int.from_bytes(hashlib.sha256(f'{seed}:{dim}:{fid}:{run}'.encode()).digest()[:4], 'little')


def reset_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(device):
    return (torch.get_rng_state(), np.random.get_state(),
            torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])
    if state[2] is not None:
        torch.cuda.set_rng_state(state[2], device)


def stats(values, fopt):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                median=float(np.median(values)), gap_mean=float((values - fopt).mean()),
                best=float(values.min()), worst=float(values.max()), worst_run=int(values.argmax()) + 1)


def csv_writer(stack, path, fields):
    stream = stack.enter_context(path.open('w', encoding='utf-8', newline=''))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


@torch.no_grad()
def run_comparison(reference, candidates=(), dims=(30,), functions=None, runs=3,
                   popsize=100, generations=100, device='auto', seed=2026, threads=1,
                   benchmark='author', output='outputs/bbob_compare'):
    functions = list(range(1, 25)) if functions is None else list(functions)
    dims, candidates = list(dims), list(candidates)
    if (benchmark not in ('author', 'local') or runs < 1 or popsize < 2 or generations < 0
            or threads < 1 or not dims or any(d < 2 for d in dims)
            or len(set(dims)) != len(dims) or not functions
            or len(set(functions)) != len(functions) or any(f not in range(1, 25) for f in functions)):
        raise ValueError('Invalid benchmark, dimensions, functions, runs, budget or threads')
    if benchmark == 'author' and any(d not in (30, 100) for d in dims):
        raise ValueError('Author offsets are published only for D=30 and D=100; use --benchmark local for other dimensions')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unavailable evaluation device: {device}')
    if device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        # Avoid reduced-mantissa matrix products in the checkpoint comparison.
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(threads)
    paths = [Path(reference), *map(Path, candidates)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}; pass weights.pth, last.pt or best.pt')
    offset_paths = {d: ROOT / 'test/data' / f'bbobOffsets_dim{d}.pkl' for d in dims} if benchmark == 'author' else {}
    offsets = {d: load_offsets(path) for d, path in offset_paths.items()}
    if offsets and not (AUTHOR_ROOT / 'manifest.json').is_file():
        raise FileNotFoundError(f'Missing frozen author source: {AUTHOR_ROOT}; include test/reference when copying the project')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; use a new --output to preserve existing results')
    models, model_info = [], []
    for index, path in enumerate(paths):
        name = 'reference' if index == 0 else f'candidate_{index:02d}'
        model, info = load_model(path, name, device, popsize)
        models.append((name, model))
        model_info.append(info)
        print(f'{name}: {path} | POM-{info["modelsize"]} | saved {info["saved_dtype"]} -> evaluation float32', flush=True)
    notes = ['std uses ddof=0; fitness_minus_fopt is logged separately and does not change the network input.',
             'Each trial is one population. All checkpoints share its initial population and CPU/CUDA/NumPy RNG state.',
             'Model is the locally repaired author network; audit_author_bbob.py compares it with the frozen original model and task.',
             'Float32 source fitness is not a high-precision certificate; saved best coordinates permit independent reevaluation.']
    if benchmark == 'author':
        notes.append('Author F20 draws fresh signs in each fitness call and contains a population-wide sum; its source behavior is preserved.')
    source_paths = ([AUTHOR_ROOT / 'BBOB_pkg' / 'BBOB' / name for name in ('bbobfunctions.py', 'utils.py')]
                    if benchmark == 'author' else [ROOT / 'platemo_bbob_problems' / f'BBOB_F{f}.py' for f in functions])
    source_paths += [ROOT / name for name in ('model/pom.py', 'train/train.py',
                                               'test/compare_bbob.py', 'test/evaluate_platemo.py')]
    if benchmark == 'local':
        source_paths.append(ROOT / 'platemo_bbob_problems/compat.py')
    report = dict(benchmark=benchmark, author_revision=AUTHOR_REVISION if benchmark == 'author' else None,
                  dims=dims, functions=functions, runs=runs, popsize=popsize, generations=generations,
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  dtype='float32', seed=seed, threads=threads, torch_version=str(torch.__version__),
                  numpy_version=np.__version__, evaluations_per_trial=popsize * (generations + 1),
                  evaluations_total=len(models) * len(dims) * len(functions) * runs * popsize * (generations + 1),
                  models=model_info, notes=notes, source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                  offset_sha256={str(d): file_hash(p) for d, p in offset_paths.items()}, results=[])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    solutions, curves = {}, {}
    with ExitStack() as stack:
        per_run = csv_writer(stack, output / 'per_run.csv',
                             ['dim', 'function', 'model', 'run', 'seed', 'fopt', 'initial_best',
                              'final_best', 'fitness_minus_fopt', 'evaluations', 'initial_population_sha256'])
        traces = csv_writer(stack, output / 'traces.csv',
                            ['dim', 'function', 'model', 'run', 'seed', 'generation', 'evaluations',
                             'best_fitness', 'fitness_minus_fopt'])
        summary_fields = ['dim', 'function', 'candidate', 'fopt', 'reference_mean', 'reference_std',
                          'reference_median', 'reference_gap_mean', 'reference_best', 'reference_worst', 'reference_worst_run',
                          'candidate_mean', 'candidate_std', 'candidate_median', 'candidate_gap_mean',
                          'candidate_best', 'candidate_worst', 'candidate_worst_run', 'mean_difference',
                          'candidate_over_reference_gap', 'paired_wins', 'paired_ties', 'paired_losses']
        summary = csv_writer(stack, output / 'comparison.csv', summary_fields)
        for dim in dims:
            for fid in functions:
                problem = (AuthorBBOB(fid, dim, offsets[dim], device) if benchmark == 'author'
                           else PlatEMOBBOB(fid, dim, device, torch.float32))
                fopt = problem.fopt if benchmark == 'author' else 0.0
                finals = {name: [] for name, _ in models}
                trajectories = {name: [] for name, _ in models}
                elites = {name: [] for name, _ in models}
                for run in range(runs):
                    run_seed = trial_seed(seed, dim, fid, run)
                    reset_seed(run_seed)
                    initial = problem.random_population(1, popsize)
                    shared_rng = capture_rng(device)
                    initial_hash = hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()
                    for name, model in models:
                        restore_rng(shared_rng, device)
                        pop = initial.clone()
                        trace = [pop[..., 0].min().item()]
                        for _ in range(generations):
                            pop, _, _ = model(pop, problem)
                            if not torch.isfinite(pop).all():
                                raise FloatingPointError(f'Nonfinite population: {name} D{dim} F{fid} run{run + 1}')
                            best = pop[..., 0].min().item()
                            if best > trace[-1]:
                                raise AssertionError('Elitist selection lost the previous best')
                            trace.append(best)
                        finals[name].append(trace[-1])
                        trajectories[name].append(trace)
                        best_index = pop[0, :, 0].argmin()
                        elites[name].append(pop[0, best_index, 1:].cpu().numpy().copy())
                        per_run.writerow(dict(dim=dim, function=f'F{fid}', model=name, run=run + 1,
                                              seed=run_seed, fopt=fopt, initial_best=trace[0], final_best=trace[-1],
                                              fitness_minus_fopt=trace[-1] - fopt,
                                              evaluations=report['evaluations_per_trial'], initial_population_sha256=initial_hash))
                        for generation, value in enumerate(trace):
                            traces.writerow(dict(dim=dim, function=f'F{fid}', model=name, run=run + 1,
                                                 seed=run_seed, generation=generation, evaluations=popsize * (generation + 1),
                                                 best_fitness=value, fitness_minus_fopt=value - fopt))
                reference_stats = stats(finals['reference'], fopt)
                entries = []
                for name in [item[0] for item in models[1:]] or [None]:
                    row = dict(dim=dim, function=f'F{fid}', candidate=name or '', fopt=fopt,
                               **{f'reference_{key}': value for key, value in reference_stats.items()})
                    if name is not None:
                        candidate_stats = stats(finals[name], fopt)
                        difference = np.asarray(finals[name]) - np.asarray(finals['reference'])
                        ratio = (candidate_stats['gap_mean'] / reference_stats['gap_mean']
                                 if reference_stats['gap_mean'] > 0 and candidate_stats['gap_mean'] >= 0 else None)
                        row.update({f'candidate_{key}': value for key, value in candidate_stats.items()})
                        row.update(mean_difference=float(difference.mean()), candidate_over_reference_gap=ratio,
                                   paired_wins=int((difference < 0).sum()), paired_ties=int((difference == 0).sum()),
                                   paired_losses=int((difference > 0).sum()))
                    summary.writerow(row)
                    entries.append(row)
                report['results'].extend(entries)
                for name, _ in models:
                    key = f'D{dim}_F{fid:02d}_{name}'
                    solutions[key] = np.asarray(elites[name])
                    curves[key] = np.asarray(trajectories[name])
                message = ' | '.join(f'{name}={stats(values, fopt)["mean"]:.6e}' for name, values in finals.items())
                print(f'D={dim} F{fid:02d}: {message}', flush=True)
    np.savez_compressed(output / 'best_solutions.npz', **solutions)
    np.savez_compressed(output / 'curves.npz', **curves)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved comparison: {output / "comparison.csv"}', flush=True)
    return report



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=str(ROOT / 'ckpt' / 'pom_m_release.pth'))
    parser.add_argument('--candidate', nargs='+', default=[], help='Trained weights.pth, last.pt or best.pt; supports multiple independent training runs')
    parser.add_argument('--dims', '--dim', nargs='+', type=int, default=[30])
    parser.add_argument('--functions', nargs='+', type=int, choices=range(1, 25))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--popsize', type=int, default=100)
    parser.add_argument('--generations', '--step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--benchmark', choices=['author', 'local'], default='author')
    parser.add_argument('--output', default='outputs/bbob_compare')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        run_comparison(args.reference, args.candidate, args.dims, args.functions, args.runs,
                       args.popsize, args.generations, args.device, args.seed, args.threads,
                       args.benchmark, args.output)
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        raise SystemExit(str(error)) from error
