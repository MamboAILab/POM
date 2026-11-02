"""MetaGBT (Algorithm 1 / Eq. 16), shared across all sampled training tasks.

Each generation updates theta once across TF1--TF5; population history is
detached between updates. The published author architecture is unchanged.
The author did not publish a trainer; this loop implements paper Algorithm 1.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
from model.pom import GB_GLHF


MODEL_SIZES = {'vs': (200, 4), 's': (500, 4), 'm': (1000, 4),
               'l': (2000, 20), 'vl': (5000, 50), 'xl': (10000, 100)}
TRAINING_PROTOCOL = 'author-safe-sqrt-eq16-f64-reduction-v1'
DEFAULTS = dict(epochs=100, generations=100, batch_size=8, popsize=100, dim=10,
                functions=[1, 2, 3, 4, 5], modelsize='m', lr=1e-4,
                diversity_weight=0.005, grad_clip=1.0, seed=2026, init='normal',
                device='auto', threads=1, log_every=10, output='outputs/pom_train',
                variant='release', dtype='float32', taskset='tf',
                eval_every=1, eval_runs=32, eval_generations=100,
                protocol_version=TRAINING_PROTOCOL)


class TrainingProblem:
    """Independent shifted tasks from Appendix Table 3, one per population.

    The batch contains batch_size instances of each function, in function order.
    Bounds/offsets are per instance, and stay fixed throughout one trajectory.
    """
    def __init__(self, functions, dim, batch_size, device, dtype=torch.float32):
        self.functions = tuple(functions)
        self.batch_size = batch_size
        self.dim = dim
        bounds = [10.0 if fid <= 2 else 100.0
                  for fid in functions for _ in range(batch_size)]
        shift_bounds = [10.0 if fid <= 2 else 50.0
                        for fid in functions for _ in range(batch_size)]
        self.upper = torch.tensor(bounds, device=device, dtype=dtype).view(-1, 1, 1)
        self.lower = -self.upper
        shift_bounds = torch.tensor(shift_bounds, device=device, dtype=dtype).view(-1, 1, 1)
        self.offset = (torch.rand(len(bounds), 1, dim, device=device, dtype=dtype) * 2 - 1) * shift_bounds

    def random_population(self, popsize):
        x = torch.rand(len(self.offset), popsize, self.dim, device=self.offset.device,
                       dtype=self.offset.dtype)
        return self.calfitness(self.lower + x * (self.upper - self.lower))[0]

    def calfitness(self, x):
        if x.ndim != 3 or x.shape[0] != len(self.offset) or x.shape[-1] != self.dim:
            raise ValueError('Expected [number_of_tasks, population_size, dimension]')
        # Engineering choice: use the released demo's hard boundary repair.
        x = x.clamp(min=self.lower, max=self.upper)
        fitness = []
        for i, fid in enumerate(self.functions):
            sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = x[sl] - self.offset[sl]
            if fid == 1:
                f = z.abs().sum(-1)
            elif fid == 2:
                # Table 3 does not specify the end index; use adjacent pairs,
                # i=1,...,d-1, without wrapping the last coordinate to the first.
                f = (z[..., :-1] + z[..., 1:]).abs().sum(-1) + z.abs().sum(-1)
            elif fid == 3:
                f = z.square().sum(-1)
            elif fid == 4:
                f = z.abs().amax(-1)
            elif fid == 5:
                f = (100 * (z[..., :-1].square() - z[..., 1:]).square()
                     + (z[..., :-1] - 1).square()).sum(-1)
            else:
                raise ValueError('Training functions must be TF1--TF5')
            fitness.append(f)
        f = torch.cat(fitness, 0).unsqueeze(-1)
        return torch.cat((f, x), -1), f


class BBOBTasks:
    """Existing local objective adapter; independent of the POM implementation."""
    def __init__(self, functions, dim, batch_size, device, dtype):
        from train.problems import BBOBTrainingProblem
        self.functions, self.dim, self.batch_size = tuple(functions), dim, batch_size
        self.problems = [BBOBTrainingProblem(fid, dim, batch_size, device, dtype=dtype)
                         for fid in functions]

    def random_population(self, popsize):
        return torch.cat([problem.random_population(popsize) for problem in self.problems])

    def calfitness(self, x):
        if (x.ndim != 3 or x.shape[0] != len(self.problems) * self.batch_size
                or x.shape[-1] != self.dim):
            raise ValueError('Expected [number_of_tasks, population_size, dimension]')
        populations = []
        for i, problem in enumerate(self.problems):
            sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
            populations.append(problem.calfitness(x[sl])[0])
        pop = torch.cat(populations)
        return pop, pop[..., :1]


def sample_tasks(config, batch_size=None, validation=False):
    args = (config['functions'], config['dim'], batch_size or config['batch_size'],
            config['device'], getattr(torch, config.get('dtype', 'float32')))
    if config.get('taskset', 'tf') == 'bbob':
        return BBOBTasks(*args)
    return TrainingProblem(*args)


def pom_loss(previous, current, diversity_weight=0.005):
    """Eq. 16, computed per population before taking the task average.

    previous is a constant at this optimizer update (Algorithm 1, line 12).
    Fitness is column 0; diversity is measured only on decision coordinates.
    """
    # Reduce in double before dividing: a float32 intermediate reciprocal can
    # overflow even while the final coordinate/parameter gradient is finite.
    before = previous[..., 0].detach().mean(-1, dtype=torch.float64)
    after = current[..., 0].mean(-1, dtype=torch.float64)
    # A fixed 1e-8 floor changes Eq.16 and kills the signal near an optimum.
    # Only a truly zero denominator needs a convention (zero improvement -> 0).
    denominator = before.abs()
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    if current.dtype == torch.float32:
        # Training averages the returned population losses over all tasks.
        fitness_gradient = 1.0 / (denominator * current.shape[1] * before.numel())
        if (fitness_gradient > torch.finfo(torch.float32).max).any():
            raise FloatingPointError('Eq.16 gradients exceed float32 range near zero; restart training with --dtype float64')
    relative = (after - before) / denominator
    diversity = current[..., 1:].to(torch.float64).std(dim=1).mean(-1)
    return relative - diversity_weight * diversity, relative, diversity


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # torch 1.12 does not expose weights_only.
        return torch.load(path, map_location='cpu')


def build_model(config):
    if config.get('variant', 'release') != 'release':
        raise ValueError('Only the published author architecture is supported; use its weights')
    mutdim, crdim = MODEL_SIZES[config['modelsize']]
    model = GB_GLHF(popsize=config['popsize'], muthdim=mutdim, crhdim=crdim)
    if config['init'] == 'normal':
        # Section 3.4: standard normal initialization of theta, including biases.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_()
    return model.to(device=config['device'], dtype=getattr(torch, config.get('dtype', 'float32')))


def config_for_weights(config, weights):
    """Identify old raw release weights by their bottleneck; never reinterpret them."""
    config = dict(config)
    mutdim, crdim = weights['agen.fq1.0.weight'].shape[0], weights['crgen.net.0.weight'].shape[0]
    sizes = [name for name, widths in MODEL_SIZES.items() if widths == (mutdim, crdim)]
    if not sizes:
        raise ValueError('Unrecognized POM weight dimensions')
    config['modelsize'] = sizes[0]
    if weights['agen.w.0.weight'].shape[0] != 100:
        raise ValueError('These weights use an experimental architecture, not the published author network')
    config['variant'] = 'release'
    config['dtype'] = str(weights['agen.w.0.weight'].dtype).split('.')[-1]
    return config


def atomic_torch_save(data, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def grad_norm(module):
    norms = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(norms).norm().item() if norms else 0.0


def prepare(config):
    config = dict(config)
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    return config


@torch.no_grad()
def validate_model(model, config, runs=None, generations=None):
    """Fixed independent populations; raw per-run traces, no historical envelope.

    Validation never consumes training RNG or changes the model's train/eval state.
    Evaluate functions separately to bound memory and preserve subset reproducibility.
    """
    runs = runs or config.get('eval_runs', 32)
    generations = config.get('eval_generations', 100) if generations is None else generations
    devices = list(range(torch.cuda.device_count())) if config['device'].startswith('cuda') else []
    was_training = model.training
    results, traces = {}, {}
    prefix = 'F' if config.get('taskset', 'tf') == 'bbob' else 'TF'
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            for fid in config['functions']:
                run_seed = config['seed'] + 100000 + fid * 1000
                torch.manual_seed(run_seed)
                problem = sample_tasks(dict(config, functions=[fid]), runs, validation=True)
                pop = problem.random_population(config['popsize'])
                initial = pop[..., 0].amin(-1)
                trace = [initial.cpu().tolist()]
                for _ in range(generations):
                    pop, _, _ = model(pop, problem)
                    if not torch.isfinite(pop).all():
                        raise FloatingPointError(f'Nonfinite validation population {prefix}{fid}')
                    trace.append(pop[..., 0].amin(-1).cpu().tolist())
                final = pop[..., 0].amin(-1)
                # Quantile 0.5 interpolates the middle pair for an even run count.
                results[f'{prefix}{fid}'] = dict(
                    initial_best_mean=initial.mean().item(), final_best_mean=final.mean().item(),
                    final_best_median=final.quantile(.5).item(),
                    final_best_std=final.std(unbiased=False).item(),
                    final_best_of_batch=final.min().item(), final_best_per_run=final.cpu().tolist(),
                    improved_fraction=(final < initial).to(final.dtype).mean().item(), seed=run_seed)
                traces[f'{prefix}{fid}'] = trace
    finally:
        model.train(was_training)
    return dict(results=results, best_trace=traces, runs=runs, generations=generations,
                evaluations_per_run=config['popsize'] * (generations + 1),
                metric='mean of per-population final best fitness; lower is better')


def write_validation(output, history, report):
    import csv
    (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'validation_history.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['epoch', 'update', *history[0]['results']]
    with (output / 'fitness_by_epoch.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in history:
            writer.writerow(dict(epoch=item['epoch'], update=item['update'],
                                 **{name: value['final_best_mean'] for name, value in item['results'].items()}))


def train(config, checkpoint=None, initial_weights=None):
    if config.get('protocol_version', TRAINING_PROTOCOL) != TRAINING_PROTOCOL:
        raise ValueError('Training protocol differs; start a new run with --init-checkpoint')
    if checkpoint is not None and checkpoint['config'].get('protocol_version') != TRAINING_PROTOCOL:
        raise ValueError('Checkpoint predates this training protocol; use --init-checkpoint to start a separate run')
    config = dict(config, protocol_version=TRAINING_PROTOCOL)
    config = prepare(config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint is None and any((output / name).exists() for name in
                                  ('config.json', 'last.pt', 'train.jsonl')):
        raise ValueError(f'{output} already contains a run; use --resume or another --output')
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    completed, updates, history = 0, 0, []
    best_score = float('inf')
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        completed, updates = checkpoint['epoch'], checkpoint['updates']
        history = checkpoint.get('validation_history', [])
        best_score = checkpoint.get('best_validation_score', float('inf'))
        torch.set_rng_state(checkpoint['rng_cpu'])
        if config['device'].startswith('cuda') and checkpoint['rng_cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    elif initial_weights is not None:
        model.load_state_dict(initial_weights)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log_path = output / 'train.jsonl'
    if checkpoint is not None and log_path.exists():
        # A stopped run may have logged steps after its last completed epoch.
        records = []
        for line in log_path.read_text(encoding='utf-8').splitlines(keepends=True):
            try:
                if line.endswith('\n') and json.loads(line)['update'] <= updates:
                    records.append(line)
            except (ValueError, KeyError):
                break
        log_path.write_text(''.join(records), encoding='utf-8')
    print(f"POM-{config['modelsize']} | {config['device']} | {sum(p.numel() for p in model.parameters())} parameters"
          f" | {len(config['functions']) * config['batch_size']} tasks/generation", flush=True)
    start = time.perf_counter()
    model.train()
    def record_validation(epoch):
        nonlocal best_score
        report = validate_model(model, config)
        record = dict(epoch=epoch, update=updates, results=report['results'])
        history.append(record)
        # Equal weighting across function families, independent of objective scale.
        score = sum(value['final_best_mean'] / max(value['initial_best_mean'], 1e-300)
                    for value in report['results'].values()) / len(report['results'])
        if score < best_score:
            best_score = score
            atomic_torch_save(dict(model=model.state_dict(), config=config, epoch=epoch,
                                   validation=report, validation_score=score), output / 'best.pt')
        write_validation(output, history, dict(report, epoch=epoch, update=updates))
        print(f'validation epoch={epoch} mean normalized final/initial={score:.6g}', flush=True)
    if not history:
        record_validation(completed)
    with log_path.open('a', encoding='utf-8') as log:
        for epoch in range(completed, config['epochs']):
            problem = sample_tasks(config)
            pop = problem.random_population(config['popsize'])
            for generation in range(config['generations']):
                # Algorithm 1 updates theta every generation. Holding the previous
                # population constant avoids reusing a graph after Adam mutates theta.
                previous = pop.detach()
                optimizer.zero_grad(set_to_none=True)
                pop, mutation, crossover = model(previous, problem)
                losses, relative, diversity = pom_loss(previous, pop, config['diversity_weight'])
                loss = losses.mean()
                if not torch.isfinite(pop).all() or not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite population/loss at epoch {epoch + 1}, generation {generation + 1}')
                loss.backward()
                mutation_grad, crossover_grad = grad_norm(model.agen), grad_norm(model.crgen)
                total_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'],
                                                           error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                row = dict(epoch=epoch + 1, generation=generation + 1, update=updates,
                           loss=loss.item(), relative_change=relative.mean().item(),
                           diversity=diversity.mean().item(), grad_norm=total_grad.item(),
                           mutation_grad_norm=mutation_grad, crossover_grad_norm=crossover_grad,
                           mutation_saturation=(mutation.detach().abs() > .99).to(pop.dtype).mean().item(),
                           crossover_mean=crossover.detach().mean().item(),
                           accepted_fraction=(pop[..., 0] < previous[..., 0].sort(-1).values).to(pop.dtype).mean().item(),
                           task_losses={f"{'F' if config.get('taskset') == 'bbob' else 'TF'}{fid}":
                               losses.detach()[i * config['batch_size']:(i + 1) * config['batch_size']].mean().item()
                               for i, fid in enumerate(config['functions'])},
                           task_mean_fitness={f"{'F' if config.get('taskset') == 'bbob' else 'TF'}{fid}": pop.detach()[i * config['batch_size']:
                               (i + 1) * config['batch_size'], :, 0].mean().item()
                               for i, fid in enumerate(config['functions'])})
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if updates == 1 or updates % config['log_every'] == 0:
                    print(f"epoch={epoch + 1} gen={generation + 1} update={updates} loss={loss.item():.6f}"
                          f" grad(LMM/LCM)={mutation_grad:.3g}/{crossover_grad:.3g}", flush=True)
                pop = pop.detach()
            log.flush()
            if (epoch + 1) % config.get('eval_every', 1) == 0 or epoch + 1 == config['epochs']:
                record_validation(epoch + 1)
            state = dict(format_version=3, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         config=config, epoch=epoch + 1, updates=updates,
                         validation_history=history, best_validation_score=best_score,
                         rng_cpu=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state_all() if config['device'].startswith('cuda') else [],
                         torch_version=str(torch.__version__))
            # Resume is exact at completed-epoch boundaries on the same device.
            atomic_torch_save(state, output / 'last.pt')
            atomic_torch_save(model.state_dict(), output / 'weights.pth')
            print(f"saved epoch={epoch + 1} updates={updates} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    return model


@torch.no_grad()
def evaluate(config, weights, runs=5, generations=100):
    config = prepare(config)
    model = build_model(config)
    model.load_state_dict(weights)
    model.eval()
    report = dict(validate_model(model, config, runs, generations), config=config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['results'], indent=2), flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('epochs', 'generations', 'batch-size', 'popsize', 'dim', 'seed', 'threads', 'log-every', 'eval-every'):
        p.add_argument('--' + name, type=int)
    for name in ('lr', 'diversity-weight', 'grad-clip'):
        p.add_argument('--' + name, type=float)
    p.add_argument('--functions', nargs='+', type=int, choices=range(1, 25))
    p.add_argument('--taskset', choices=['tf', 'bbob'])
    p.add_argument('--variant', choices=['release'], help=argparse.SUPPRESS)
    p.add_argument('--dtype', choices=['float32', 'float64'])
    p.add_argument('--modelsize', choices=MODEL_SIZES)
    p.add_argument('--init', choices=['normal', 'default'])
    p.add_argument('--device', help='auto, cpu, cuda, or cuda:N')
    p.add_argument('--output')
    paths = p.add_mutually_exclusive_group()
    paths.add_argument('--resume', help='Full last.pt checkpoint; restores Adam and RNG too')
    paths.add_argument('--init-checkpoint', help='Raw released/trained weights for fine-tuning or evaluation')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--eval-runs', type=int)
    p.add_argument('--eval-generations', type=int)
    args = p.parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    initial_state = load_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    if checkpoint is not None and ('config' not in checkpoint or
                                   (not args.evaluate and 'optimizer' not in checkpoint)):
        p.error('--resume requires last.pt; use --init-checkpoint for raw weights')
    if checkpoint is not None and not args.evaluate and checkpoint['config'].get('protocol_version') != TRAINING_PROTOCOL:
        p.error('Checkpoint uses an older/unknown training protocol; use --init-checkpoint to start a new run')
    if args.evaluate and not (args.resume or args.init_checkpoint):
        p.error('--evaluate requires --resume or --init-checkpoint')
    config = dict(DEFAULTS)
    if args.evaluate and initial_state is not None and 'config' in initial_state:
        config.update(initial_state['config'])
    if checkpoint is not None:
        config.update(checkpoint['config'])
        config.update({key: value for key, value in config_for_weights(config, checkpoint['model']).items()
                       if key in ('variant', 'dtype', 'modelsize')})
    elif (args.taskset == 'bbob' and args.functions is None
          and not (args.evaluate and initial_state is not None and 'config' in initial_state)):
        config['functions'] = list(range(1, 25))
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            if checkpoint is not None and not args.evaluate and key not in (
                    'epochs', 'device', 'threads', 'log_every', 'output') and value != config[key]:
                p.error(f'Cannot change {key} during resume; use --init-checkpoint to start a new run')
            if checkpoint is not None and args.evaluate and key == 'modelsize' and value != config[key]:
                p.error('Evaluation modelsize must match the checkpoint')
            config[key] = value
    weights = initial_state
    if weights is not None and 'model' in weights:
        weights = weights['model']
    if weights is not None:
        inferred = config_for_weights(config, weights)
        for key in ('modelsize', 'variant'):
            if getattr(args, key) is not None and config[key] != inferred[key]:
                p.error(f'{key} does not match the supplied weights')
            config[key] = inferred[key]
        if args.dtype is None:
            config['dtype'] = inferred['dtype']
    for key in ('epochs', 'generations', 'batch_size', 'threads', 'log_every', 'eval_every', 'eval_runs'):
        if config[key] < 1:
            p.error(f'{key} must be positive')
    if config['popsize'] < 2 or config['dim'] < 2:
        p.error('popsize and dim must be at least 2')
    if len(set(config['functions'])) != len(config['functions']):
        p.error('--functions must not contain duplicates')
    if config['taskset'] == 'tf' and any(fid > 5 for fid in config['functions']):
        p.error('TF tasks are 1..5; use --taskset bbob for BBOB 1..24')
    if not math.isfinite(config['lr']) or config['lr'] <= 0 or not math.isfinite(config['grad_clip']) or config['grad_clip'] <= 0:
        p.error('lr and grad_clip must be finite and positive')
    if not math.isfinite(config['diversity_weight']) or config['diversity_weight'] < 0:
        p.error('diversity_weight must be finite and nonnegative')
    if config['eval_generations'] < 0:
        p.error('eval-runs must be positive and eval-generations nonnegative')
    if checkpoint is not None and not args.evaluate and config['epochs'] <= checkpoint['epoch']:
        p.error('--epochs is the TOTAL epoch count; set it above the completed checkpoint epoch')
    return args, config, checkpoint, weights


if __name__ == '__main__':
    args, config, checkpoint, weights = parse_args()
    if args.evaluate:
        evaluate(config, checkpoint['model'] if checkpoint is not None else weights,
                 config['eval_runs'], config['eval_generations'])
    else:
        try:
            train(config, checkpoint, weights)
        except KeyboardInterrupt:
            print('\nStopped. Resume last.pt from the last completed epoch; unfinished epoch will rerun.')
            sys.exit(130)
