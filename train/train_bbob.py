"""Train a fresh POM on each local BBOB function, with fixed same-task evaluation.

No TF tasks, shared warm start, NCD, router or task selector are used here.
One epoch restarts the populations; POM still updates Adam once per generation.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from train.train import (DEFAULTS as POM_DEFAULTS, MODEL_SIZES, atomic_torch_save,
                   build_model, grad_norm, load_checkpoint, pom_loss, prepare,
                   TRAINING_PROTOCOL)
from train.problems import BBOBTrainingProblem


DEFAULTS = dict(POM_DEFAULTS, functions=list(range(1, 25)), epochs=500,
                runs=1, eval_generations=100, eval_batch_size=64, eval_every=1,
                log_every=100, output='outputs/bbob24_solo500_seed2026',
                reference_first_run=None)
DEFAULTS.update(taskset='bbob')
FITNESS_FIELDS = ['epoch', 'update', 'fitness_best_of_batch',
                  'fitness_mean_best', 'fitness_std_best']
RESUME_MUTABLE = {'epochs', 'device', 'threads', 'log_every', 'output'}


def existing_config(config):
    """Older runs predate explicit author-architecture and precision metadata."""
    return dict(dict(variant='release', dtype='float32', taskset='bbob', eval_runs=32), **config)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(config):
    if config.get('protocol_version') != TRAINING_PROTOCOL:
        raise ValueError('Unknown/older training protocol; use a new output directory')
    if config.get('dtype') not in ('float32', 'float64') or config.get('variant') != 'release' or config.get('taskset') != 'bbob':
        raise ValueError('Require the published model, BBOB tasks and an explicit float32/float64 dtype')
    for key in ('epochs', 'generations', 'batch_size', 'runs', 'eval_batch_size',
                'eval_every', 'threads', 'log_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['dim'] < 2 or config['popsize'] < 2 or config['eval_generations'] < 0:
        raise ValueError('dim/popsize >= 2 and eval_generations >= 0 required')
    if (not config['functions'] or len(set(config['functions'])) != len(config['functions'])
            or any(fid not in range(1, 25) for fid in config['functions'])):
        raise ValueError('functions must be distinct BBOB IDs in 1..24')
    for key in ('lr', 'grad_clip'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(config['diversity_weight']) or config['diversity_weight'] < 0:
        raise ValueError('diversity_weight must be finite and nonnegative')
    if config['modelsize'] not in MODEL_SIZES or config['init'] not in ('normal', 'default'):
        raise ValueError('Unknown model size or initialization')


@torch.no_grad()
def evaluate(model, problem, config):
    """Identical evaluation randomness each epoch, without advancing training RNG."""
    devices = list(range(torch.cuda.device_count())) if config['device'].startswith('cuda') else []
    was_training = model.training
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['eval_seed'])
            evaluation_problem = BBOBTrainingProblem(problem.fid, problem.dim,
                                                    config['eval_batch_size'], config['device'],
                                                    dtype=getattr(torch, config['dtype']))
            pop = evaluation_problem.random_population(config['popsize'])
            for _ in range(config['eval_generations']):
                pop, _, _ = model(pop, evaluation_problem)
                if not torch.isfinite(pop).all():
                    raise FloatingPointError(f'Nonfinite evaluation in F{problem.fid}')
            best = pop[..., 0].amin(-1)
            return dict(fitness_best_of_batch=best.min().item(),
                        fitness_mean_best=best.mean().item(),
                        fitness_std_best=best.std(unbiased=False).item())
    finally:
        model.train(was_training)


def train_one(config, resume=False):
    """One function, one fresh model/Adam; resume only from this exact run."""
    output = Path(config['output'])
    checkpoint_path = output / 'last.pt'
    checkpoint = load_checkpoint(checkpoint_path) if resume and checkpoint_path.exists() else None
    if output.exists() and any(output.iterdir()) and checkpoint is None:
        raise ValueError(f'{output} is not empty and has no resumable last.pt; use another --output')
    if checkpoint is not None:
        checkpoint['config'] = existing_config(checkpoint['config'])
        for key, value in config.items():
            if key not in RESUME_MUTABLE and value != checkpoint['config'].get(key):
                raise ValueError(f'Cannot change {key} when resuming {output}')
        if checkpoint['epoch'] > config['epochs']:
            raise ValueError('Requested epochs are below the completed checkpoint epoch')
    output.mkdir(parents=True, exist_ok=True)
    prepare(dict(config, seed=config['run_seed']))
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    problem = BBOBTrainingProblem(config['fid'], config['dim'],
                                  config['batch_size'], config['device'],
                                  dtype=getattr(torch, config['dtype']))
    completed, updates, history = 0, 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        completed, updates = checkpoint['epoch'], checkpoint['updates']
        history = checkpoint['fitness_history']
        torch.set_rng_state(checkpoint['rng_cpu'])
        if config['device'].startswith('cuda') and checkpoint['rng_cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    write_json(output / 'config.json', config)
    log_path = output / 'train.jsonl'
    if checkpoint is not None and log_path.exists():
        # Discard incomplete/uncommitted lines after a crash within an epoch.
        records = []
        for line in log_path.read_text(encoding='utf-8').splitlines(keepends=True):
            try:
                if line.endswith('\n') and json.loads(line)['update'] <= updates:
                    records.append(line)
            except (ValueError, KeyError):
                break
        log_path.write_text(''.join(records), encoding='utf-8')

    def save(epoch):
        state = dict(format_version=2, benchmark='local_bbob_single_task',
                     fid=config['fid'], run_index=config['run_index'],
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     config=config, epoch=epoch, updates=updates,
                     fitness_history=history, rng_cpu=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if config['device'].startswith('cuda') else [],
                     torch_version=str(torch.__version__))
        atomic_torch_save(state, checkpoint_path)
        atomic_torch_save(model.state_dict(), output / 'weights.pth')
        write_csv(output / 'fitness.csv', FITNESS_FIELDS, history)

    if not history:
        history.append(dict(epoch=0, update=0, **evaluate(model, problem, config)))
        save(0)
    start = time.perf_counter()
    model.train()
    print(f"F{config['fid']:02d} run={config['run_index']} seed={config['run_seed']}"
          f" | POM-{config['modelsize']} | {config['device']}"
          f" | epoch {completed}/{config['epochs']} | {config['generations']} Adam updates/epoch", flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        for epoch in range(completed + 1, config['epochs'] + 1):
            pop = problem.random_population(config['popsize'])
            for generation in range(1, config['generations'] + 1):
                previous = pop.detach()
                optimizer.zero_grad(set_to_none=True)
                pop, _, _ = model(previous, problem)
                losses, relative, diversity = pom_loss(previous, pop, config['diversity_weight'])
                loss = losses.mean()
                if not torch.isfinite(pop).all() or not torch.isfinite(loss):
                    raise FloatingPointError(f"F{config['fid']} epoch={epoch} generation={generation}: nonfinite population/loss")
                loss.backward()
                mutation_grad, crossover_grad = grad_norm(model.agen), grad_norm(model.crgen)
                total_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'],
                                                           error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                row = dict(fid=config['fid'], epoch=epoch, generation=generation, update=updates,
                           loss=loss.item(), relative_change=relative.mean().item(),
                           diversity=diversity.mean().item(), grad_norm=total_grad.item(),
                           mutation_grad_norm=mutation_grad, crossover_grad_norm=crossover_grad,
                           train_mean_fitness=pop.detach()[..., 0].mean().item())
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if updates == 1 or updates % config['log_every'] == 0:
                    print(f"F{config['fid']:02d} epoch={epoch} gen={generation} update={updates}"
                          f" loss={loss.item():.6f} grad(LMM/LCM)={mutation_grad:.3g}/{crossover_grad:.3g}", flush=True)
                pop = pop.detach()
            if epoch % config['eval_every'] == 0 or epoch == config['epochs']:
                history.append(dict(epoch=epoch, update=updates, **evaluate(model, problem, config)))
            log.flush()
            save(epoch)
            print(f"F{config['fid']:02d} epoch={epoch}/{config['epochs']} updates={updates}"
                  f" last_eval_epoch={history[-1]['epoch']} fitness={history[-1]['fitness_best_of_batch']:.6E}"
                  f" elapsed={time.perf_counter() - start:.1f}s", flush=True)
    # Re-export committed artifacts even if a previous process stopped after last.pt.
    save(config['epochs'])
    return history


def read_reference_first_run(config, fid):
    """Read an existing run's recorded prefix; never load/relabel its later weights."""
    folder = Path(config['reference_first_run']) / f'F{fid:02d}' / 'run_01'
    previous = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    expected = dict(config, fid=fid, run_index=1, run_seed=config['seed'],
                    eval_seed=config['seed'] + 10000000 + fid * 1000)
    # Epoch count/runs/output differ intentionally; every training/evaluation setting must match.
    for key in ('fid', 'run_index', 'run_seed', 'eval_seed', 'modelsize', 'init',
                'dim', 'popsize', 'batch_size', 'generations', 'lr', 'diversity_weight',
                'grad_clip', 'seed', 'eval_generations', 'eval_batch_size', 'eval_every',
                'variant', 'dtype', 'taskset', 'protocol_version'):
        if previous.get(key) != expected[key]:
            raise ValueError(f'Reference F{fid} differs in {key}; cannot pool these runs')
    if previous['epochs'] < config['epochs']:
        raise ValueError(f'Reference F{fid} has fewer than {config["epochs"]} epochs')
    with (folder / 'fitness.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FITNESS_FIELDS:
            raise ValueError(f'Reference F{fid} has an incompatible fitness.csv')
        history = [{key: int(row[key]) if key in ('epoch', 'update') else float(row[key])
                    for key in FITNESS_FIELDS}
                   for row in reader if int(row['epoch']) <= config['epochs']]
    epochs = [0] + [epoch for epoch in range(1, config['epochs'] + 1)
                    if epoch % config['eval_every'] == 0 or epoch == config['epochs']]
    if ([row['epoch'] for row in history] != epochs
            or any(row['update'] != row['epoch'] * config['generations'] for row in history)
            or any(not math.isfinite(value) for row in history for value in row.values())):
        raise ValueError(f'Reference F{fid} has incomplete or invalid recorded history')
    return history


def run_baseline(config, resume=False):
    config = dict(DEFAULTS, **config)
    validate(config)
    config = prepare(config)
    root = Path(config['output'])
    references = {}
    if config['reference_first_run']:
        reference = Path(config['reference_first_run']).resolve()
        if reference == root.resolve():
            raise ValueError('Reference and new output must use different directories')
        if config['runs'] < 2:
            raise ValueError('With --reference-first-run, --runs must include at least one new run')
        config['reference_first_run'] = str(reference)
        # Preflight every task before creating output or starting expensive GPU work.
        references = {fid: read_reference_first_run(config, fid) for fid in config['functions']}
    config_path = root / 'config.json'
    if root.exists() and any(root.iterdir()):
        if not resume or not config_path.exists():
            raise ValueError(f'{root} is not empty; use --resume or another --output')
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        previous = existing_config(previous)
        for key, value in config.items():
            if key not in RESUME_MUTABLE and previous.get(key) != value:
                raise ValueError(f'Cannot change {key} during resume')
    root.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    results, rows, curves = {}, [], {}
    report = dict(benchmark='local_bbob_single_task', tf_pretraining=False, config=config,
                  epoch_definition='fresh populations; one Adam update per generation',
                  final_metric='mean across training runs of best over eval_batch_size populations x popsize individuals',
                  eval_function_calls_per_epoch=config['eval_batch_size'] * config['popsize'] * (config['eval_generations'] + 1),
                  results=results)
    for fid in config['functions']:
        histories, seeds = [], []
        for run_index in range(1, config['runs'] + 1):
            # Same initial weights across functions within a repetition, but no
            # trained weights or Adam state are ever carried to another task.
            run_seed = config['seed'] + (run_index - 1) * 1000
            task_config = dict(config, functions=[fid], fid=fid, run_index=run_index,
                               run_seed=run_seed,
                               eval_seed=config['seed'] + 10000000 + fid * 1000 + run_index - 1,
                               output=str(root / f'F{fid:02d}' / f'run_{run_index:02d}'))
            if run_index == 1 and references:
                history = references[fid]
                print(f'REUSE F{fid:02d} run=1: recorded epochs 0--{config["epochs"]}; '
                      'reference files are read-only; no weights carried to new runs', flush=True)
            else:
                history = train_one(task_config, resume)
            histories.append(history)
            seeds.append(run_seed)
        final = [history[-1]['fitness_best_of_batch'] for history in histories]
        result = dict(final_mean=float(np.mean(final)), final_std=float(np.std(final)),
                      final_per_run=final, run_seeds=seeds, epochs=config['epochs'],
                      updates_per_run=config['epochs'] * config['generations'])
        results[f'F{fid}'] = result
        rows.append(dict(function=f'F{fid}', final_mean=result['final_mean'], final_std=result['final_std'],
                         runs=config['runs'], epochs=config['epochs'], updates_per_run=result['updates_per_run']))
        common_epochs = set.intersection(*(set(row['epoch'] for row in history) for history in histories))
        curves[f'F{fid}'] = {epoch: float(np.mean([next(row['fitness_best_of_batch'] for row in history
                                                      if row['epoch'] == epoch) for history in histories]))
                             for epoch in sorted(common_epochs)}
        write_json(root / 'summary.json', report)
        write_csv(root / 'summary.csv', ['function', 'final_mean', 'final_std', 'runs', 'epochs', 'updates_per_run'], rows)
        epochs = sorted(set().union(*(curve.keys() for curve in curves.values())))
        write_csv(root / 'fitness_by_epoch.csv', ['epoch', *curves],
                  [dict(epoch=epoch, **{key: curve.get(epoch, '') for key, curve in curves.items()}) for epoch in epochs])
        print(f"FINISHED F{fid:02d}: FINAL={result['final_mean']:.6E} std={result['final_std']:.6E}", flush=True)
    print(f'Saved baseline: {root}', flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('epochs', 'generations', 'batch_size', 'popsize', 'dim', 'seed', 'threads',
                'log_every', 'runs', 'eval_generations', 'eval_batch_size', 'eval_every'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('lr', 'diversity_weight', 'grad_clip'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    parser.add_argument('--functions', nargs='+', type=int, choices=range(1, 25))
    parser.add_argument('--modelsize', choices=MODEL_SIZES)
    parser.add_argument('--init', choices=['normal', 'default'])
    parser.add_argument('--dtype', choices=['float32', 'float64'])
    parser.add_argument('--device', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--output')
    parser.add_argument('--reference-first-run',
                        help='Read run_01 curves from this existing baseline; train only runs 2..N in a new output')
    parser.add_argument('--resume', action='store_true', help='Resume/skip each task from its own last.pt')
    args = parser.parse_args()
    config = dict(DEFAULTS)
    if args.resume:
        config_path = Path(args.output or config['output']) / 'config.json'
        if not config_path.exists():
            parser.error(f'Cannot resume: missing {config_path}')
        config.update(json.loads(config_path.read_text(encoding='utf-8')))
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    try:
        validate(config)
    except ValueError as error:
        parser.error(str(error))
    return args, config


if __name__ == '__main__':
    args, config = parse_args()
    try:
        run_baseline(config, args.resume)
    except KeyboardInterrupt:
        print('\nStopped. Repeat this command with --resume; only the unfinished epoch will rerun.', flush=True)
        sys.exit(130)
