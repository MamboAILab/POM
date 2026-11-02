"""Check the active network against the pinned author's code, without pytest.

Run: python -m test.test_author_code
"""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SHA = '41ee8804efd5efa237bb89d95e19fa4faa0cde0b'
UPSTREAM = ROOT / 'test' / 'reference'
from model.pom import GB_GLHF
from model.pom import sortIndivBND


class Sphere:
    def calfitness(self, x):
        fitness = x.square().sum(-1, keepdim=True)
        return torch.cat((fitness, x), -1), fitness


def load_source(name, path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing pinned author source: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_author():
    """Use the real pinned imports and sorter, without replacing active imports."""
    source = UPSTREAM / 'GLHF_pkg' / 'GLHF'
    imports = load_source('_pom_author_imports', source / 'imports.py')
    imports.DEVICE = torch.device('cpu')
    # Original GLHF.utils imports BBOB helpers even though Sphere does not use
    # them. Load the actual pinned package, not a substitute implementation.
    package = load_source('GLHF', source / '__init__.py')
    with patch.object(sys, 'path', [str(UPSTREAM / 'BBOB_pkg'), *sys.path]), \
            patch.dict(sys.modules, {'GLHF': package, 'GLHF.imports': imports}):
        for name in tuple(sys.modules):
            if name == 'BBOB' or name.startswith('BBOB.'):
                del sys.modules[name]
        utils = load_source('_pom_author_utils', source / 'utils.py')
        with patch.dict(sys.modules, {'GLHF.utils': utils}):
            author = load_source('_pom_author_model', source / 'GLHFMODEL.py')
    assert author.DEVICE == torch.device('cpu')
    assert author.sortIndivBND is utils.sortIndivBND
    assert Path(author.sortIndivBND.__code__.co_filename).resolve() == (source / 'utils.py').resolve()
    return author


def load_weights():
    path = ROOT / 'ckpt' / 'pom_m_release.pth'
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    author = load_author()
    n, dimension = 16, 5
    weights = load_weights()
    model = GB_GLHF(popsize=n).cpu()
    reference = author.GB_GLHF(popsize=n).cpu()
    for instance in (model, reference):
        assert sum(parameter.numel() for parameter in instance.parameters()) == 202329
        instance.load_state_dict(weights, strict=True)
        assert set(instance.state_dict()) == set(weights)
        for key, value in instance.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        assert instance.agen.w[0].out_features == 100
        assert instance.agen.fq1[0].in_features == 100
        assert instance.agen.fk1[0].in_features == 100
        assert isinstance(instance.crgen.net[1], nn.ReLU)
        instance.eval()
    torch.testing.assert_close(model.ranks, reference.ranks, rtol=0, atol=0)

    # The author's sorter already moves whole rows and supports gradients.
    # Compare the active gather implementation against that actual function.
    rows = torch.randn(3, n, dimension + 1)
    left = rows.clone().requires_grad_()
    right = rows.clone().requires_grad_()
    sorted_left = sortIndivBND(left)
    sorted_right = author.sortIndivBND(right)
    torch.testing.assert_close(sorted_left, sorted_right, rtol=0, atol=0)
    coefficients = torch.randn_like(rows)
    (sorted_left * coefficients).sum().backward()
    (sorted_right * coefficients).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    sphere = Sphere()
    # Repeat both stochastic forwards from the same RNG state. Compare the
    # mutation matrix, crossover rates and selected population each generation.
    with torch.no_grad():
        for horizon in (1, 20):
            torch.manual_seed(7000 + horizon)
            population = sphere.calfitness(torch.randn(3, n, dimension) * 3)[0]
            expected = population.clone()
            max_errors = dict(population=0.0, mutation=0.0, crossover=0.0)
            for generation in range(horizon):
                rng = torch.get_rng_state()
                previous_best = population[..., 0].amin(-1)
                actual = model(population, sphere)
                torch.set_rng_state(rng)
                original = reference(expected, sphere)
                for label, current, target in zip(('population', 'mutation', 'crossover'), actual, original):
                    assert torch.isfinite(current).all() and torch.isfinite(target).all()
                    max_errors[label] = max(max_errors[label], (current - target).abs().max().item())
                    torch.testing.assert_close(current, target, rtol=2e-5, atol=2e-6,
                                               msg=f'{label}, horizon={horizon}, generation={generation + 1}')
                population, expected = actual[0], original[0]
                assert (population[..., 0].amin(-1) <= previous_best).all()
            print(f'PARITY {horizon} generations, max absolute differences: {max_errors}; '
                  'rtol=2e-5, atol=2e-6')

    # Real Adam training through the published architecture must update both
    # learned modules, including the hard straight-through Gumbel crossover.
    model.train()
    torch.manual_seed(2026)
    population = sphere.calfitness(torch.randn(4, n, dimension) * 3)[0]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    updated, _, _ = model(population, sphere)
    loss = ((updated[..., 0].mean(-1) - population[..., 0].mean(-1))
            / population[..., 0].mean(-1) - .005 * updated[..., 1:].std(dim=1).mean(-1)).mean()
    loss.backward()
    for module in (model.agen, model.crgen):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    optimizer.step()
    for prefix in ('agen.', 'crgen.'):
        assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()
                   if key.startswith(prefix)), f'{prefix} did not update'

    # Only undefined cases receive numerical guards; tied zero fitness and
    # zero-vector cosine must produce finite values and finite gradients.
    model.zero_grad(set_to_none=True)
    tied = torch.zeros(2, n, requires_grad=True)
    token = model.genGBMutToken(tied, model.ranks)
    token.sum().backward()
    assert torch.isfinite(token).all() and torch.isfinite(tied.grad).all()
    zero_x = torch.zeros(2, n, dimension, requires_grad=True)
    zero_updated, strategy, crossover = model(sphere.calfitness(zero_x)[0], sphere)
    assert all(torch.isfinite(value).all() for value in (zero_updated, strategy, crossover))
    zero_updated.sum().backward()
    assert torch.isfinite(zero_x.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    print(f'PASS: author {AUTHOR_SHA}, strict M weights/202329 parameters, original sorter, '
          '1/20-generation stochastic parity, both modules train, finite tied-zero backward')


if __name__ == '__main__':
    main()
