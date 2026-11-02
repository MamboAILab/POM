"""Differentiable versions of this repository's zero-fopt BBOB F1--F24.

The formulas and fixed rotations match ``platemo_bbob_problems``. At a
non-differentiable zero, the fractional-power helpers choose a finite zero
subgradient without changing the objective value.
"""
import importlib
import math

import numpy as np
import torch


def _orth(seed, dim, device, dtype=torch.float32):
    matrix = np.random.RandomState(seed).rand(dim, dim)
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return torch.tensor(u, dtype=dtype, device=device)


def _positive_power(x, exponent):
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, safe**exponent, torch.zeros_like(x))


def _tosz(x):
    safe_abs = torch.where(x != 0, torch.abs(x), torch.ones_like(x))
    log_abs = torch.log(safe_abs)
    c1 = torch.where(x > 0, 10.0, 5.5)
    c2 = torch.where(x > 0, 7.9, 3.1)
    return torch.sign(x) * torch.exp(
        log_abs + 0.049 * (torch.sin(c1 * log_abs) + torch.sin(c2 * log_abs))
    )


def _tasy(x, beta):
    dim = x.shape[-1]
    if dim == 1:
        return x
    positions = torch.arange(dim, dtype=x.dtype, device=x.device) / (dim - 1)
    # torch.where alone does not shield backward from sqrt(0) or 0**exponent
    # in its unselected branch. Substitute a harmless positive base first.
    positive = torch.where(x > 0, x, torch.ones_like(x))
    exponent = 1.0 + beta * positions * torch.sqrt(positive)
    return torch.where(x > 0, positive**exponent, x)


def _penalty(x):
    return torch.sum(torch.clamp_min(torch.abs(x) - 5.0, 0.0) ** 2, dim=-1)


def _scale(alpha, dim, device, dtype=torch.float32):
    positions = torch.arange(dim, dtype=dtype, device=device) / (dim - 1)
    return alpha ** (0.5 * positions)


class TorchBBOB:
    """Local F1--F24 with differentiable fitness of shape ``[..., N]``."""

    def __init__(self, fid, dim, device='cpu', dtype=torch.float32):
        if fid not in range(1, 25):
            raise ValueError('fid must be in 1..24')
        if dim < 2:
            raise ValueError('BBOB dimension must be at least 2')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('BBOB dtype must be torch.float32 or torch.float64')
        self.fid = fid
        self.dim = dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.xopt = torch.zeros(dim, dtype=dtype, device=device)
        self.q = _orth(1, dim, device, dtype)
        self.r = _orth(2, dim, device, dtype)

        signs = np.sign(np.random.RandomState(1).rand(dim) - 0.5)
        if fid == 5:
            self.xopt = torch.tensor(5.0 * signs, dtype=dtype, device=device)
        elif fid in (20, 21, 22):
            self.xopt = torch.tensor(
                (4.2096874633 / 2.0) * signs, dtype=dtype, device=device
            )
        elif fid == 24:
            self.xopt = torch.tensor(1.25 * signs, dtype=dtype, device=device)
        if fid in (21, 22):
            self._init_gallagher(fid)

    def _init_gallagher(self, fid):
        count = 100 if fid == 21 else 20
        alpha = 1000.0 ** (2.0 * np.arange(count) / (count - 1))
        alpha = alpha[np.random.RandomState(2).permutation(count)]
        alpha = np.concatenate(([1000.0 if fid == 21 else 1.0e6], alpha))
        positions = np.arange(self.dim) / (self.dim - 1)
        matrices = np.stack(
            [np.diag(a ** (0.5 * positions)) / (a**0.25) for a in alpha]
        )
        first_bound = 4.0 if fid == 21 else 3.92
        other_bound = 5.0 if fid == 21 else 4.9
        first = np.random.RandomState(3).uniform(-first_bound, first_bound, size=1)
        others = np.random.RandomState(4).uniform(-other_bound, other_bound, size=count)
        centers = np.repeat(np.concatenate((first, others))[:, None], self.dim, axis=1)
        self.centers = torch.tensor(centers, dtype=self.dtype, device=self.device)
        self.peak_matrices = torch.tensor(matrices, dtype=self.dtype, device=self.device)
        self.peak_weights = torch.cat((
            torch.tensor([10.0], dtype=self.dtype, device=self.device),
            torch.linspace(1.1, 9.1, count, dtype=self.dtype, device=self.device),
        ))

    def repaire(self, x):
        return torch.clamp(x, -5.0, 5.0)

    def gen_random_pop(self, shape):
        return torch.rand(shape, dtype=self.dtype, device=self.device) * 10.0 - 5.0

    def calfitness(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError('Last dimension must match the BBOB dimension')
        values = self._objective(self.repaire(x).reshape(-1, self.dim))
        return values.reshape(x.shape[:-1])

    def optimal_point(self):
        if self.fid in (9, 19):
            scale = max(1.0, math.sqrt(self.dim) / 8.0)
            target = torch.full((self.dim,), 0.5 / scale, dtype=self.dtype, device=self.device)
            return target @ self.r.T
        if self.fid in (21, 22):
            return self.centers[0].clone()
        return self.xopt.clone()

    def loss_func(self, father, offspring):
        father_mean = self.calfitness(father).mean()
        offspring_mean = self.calfitness(offspring).mean()
        return (offspring_mean - father_mean) / (torch.abs(father_mean) + 1.0e-12)

    def _objective(self, x):
        return getattr(self, '_f%d' % self.fid)(x)

    def _f1(self, x):
        return torch.sum((x - self.xopt) ** 2, dim=1)

    def _f2(self, x):
        z = _tosz(x - self.xopt)
        weights = 10.0 ** (
            6.0 * torch.arange(self.dim, device=x.device, dtype=x.dtype) / (self.dim - 1)
        )
        return torch.sum(weights * z**2, dim=1)

    def _f3(self, x):
        z = _tasy(_tosz(x - self.xopt), 0.2)
        z = z * _scale(10.0, self.dim, x.device, x.dtype)
        return 10.0 * (self.dim - torch.sum(torch.cos(2.0 * torch.pi * z), dim=1)) + torch.sum(z**2, dim=1)

    def _f4(self, x):
        z = _tosz(x - self.xopt)
        scale = _scale(10.0, self.dim, x.device, x.dtype).expand_as(z).clone()
        odd = torch.arange(0, self.dim, 2, device=x.device)
        scale[:, odd] *= torch.where(z[:, odd] > 0, 10.0, 1.0)
        z = scale * z
        return 10.0 * (self.dim - torch.sum(torch.cos(2.0 * torch.pi * z), dim=1)) + torch.sum(z**2, dim=1) + 100.0 * _penalty(x)

    def _f5(self, x):
        z = torch.where(self.xopt * x >= 25.0, self.xopt, x)
        positions = torch.arange(self.dim, device=x.device, dtype=x.dtype) / (self.dim - 1)
        slope = torch.sign(self.xopt) * 10.0**positions
        return torch.sum(5.0 * torch.abs(slope) - slope * z, dim=1)

    def _f6(self, x):
        transform = self.q @ torch.diag(_scale(10.0, self.dim, x.device, x.dtype)) @ self.r
        z = (x - self.xopt) @ transform
        scale = torch.where(z * self.xopt > 0, 100.0, 1.0)
        return _positive_power(_tosz(torch.sum((scale * z) ** 2, dim=1)), 0.9)

    def _f7(self, x):
        z = (x - self.xopt) @ (torch.diag(_scale(10.0, self.dim, x.device, x.dtype)) @ self.r)
        zh = torch.where(torch.abs(z) > 0.5, torch.floor(0.5 + z), torch.floor(0.5 + 10.0 * z) / 10.0)
        rotated = z @ self.q
        weights = 10.0 ** (
            2.0 * torch.arange(self.dim, device=x.device, dtype=x.dtype) / (self.dim - 1)
        )
        term1 = torch.abs(zh[:, 0]) / 1.0e4
        term2 = torch.sum(weights * rotated**2, dim=1)
        return 0.1 * torch.maximum(term1, term2) + _penalty(x)

    def _rosenbrock(self, z):
        return torch.sum(100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2, dim=1)

    def _f8(self, x):
        scale = max(1.0, math.sqrt(self.dim) / 8.0)
        return self._rosenbrock(scale * (x - self.xopt) + 1.0)

    def _f9(self, x):
        scale = max(1.0, math.sqrt(self.dim) / 8.0)
        return self._rosenbrock(scale * ((x - self.xopt) @ self.r) + 0.5)

    def _f10(self, x):
        z = _tosz((x - self.xopt) @ self.r)
        weights = 10.0 ** (
            6.0 * torch.arange(self.dim, device=x.device, dtype=x.dtype) / (self.dim - 1)
        )
        return torch.sum(weights * z**2, dim=1)

    def _f11(self, x):
        z = _tosz((x - self.xopt) @ self.r)
        return 1.0e6 * z[:, 0] ** 2 + torch.sum(z[:, 1:] ** 2, dim=1)

    def _f12(self, x):
        z = _tasy((x - self.xopt) @ self.r, 0.5) @ self.r
        return z[:, 0] ** 2 + 1.0e6 * torch.sum(z[:, 1:] ** 2, dim=1)

    def _f13(self, x):
        transform = self.q @ torch.diag(_scale(10.0, self.dim, x.device, x.dtype)) @ self.r
        z = (x - self.xopt) @ transform
        return z[:, 0] ** 2 + 100.0 * _positive_power(torch.sum(z[:, 1:] ** 2, dim=1), 0.5)

    def _f14(self, x):
        z = (x - self.xopt) @ self.r
        powers = 2.0 + 4.0 * torch.arange(self.dim, device=x.device, dtype=x.dtype) / (self.dim - 1)
        return _positive_power(torch.sum(torch.abs(z) ** powers, dim=1), 0.5)

    def _f15(self, x):
        z = _tasy(_tosz((x - self.xopt) @ self.r), 0.2)
        transform = self.r @ torch.diag(_scale(10.0, self.dim, x.device, x.dtype)) @ self.q
        z = z @ transform
        return 10.0 * (self.dim - torch.sum(torch.cos(2.0 * torch.pi * z), dim=1)) + torch.sum(z**2, dim=1)

    def _f16(self, x):
        z = _tosz((x - self.xopt) @ self.r)
        transform = self.r @ torch.diag(_scale(0.01, self.dim, x.device, x.dtype)) @ self.q
        z = z @ transform
        k = torch.arange(12, device=x.device, dtype=x.dtype)
        weights, frequencies = 0.5**k, 3.0**k
        f0 = torch.sum(weights * torch.cos(torch.pi * frequencies))
        inner = torch.sum(weights * torch.cos(2.0 * torch.pi * frequencies * (z[:, :, None] + 0.5)), dim=2)
        return 10.0 * (inner.mean(dim=1) - f0) ** 3 + 10.0 / self.dim * _penalty(x)

    def _schaffers(self, x, alpha):
        z = _tasy((x - self.xopt) @ self.r, 0.5)
        z = z @ (torch.diag(_scale(alpha, self.dim, x.device, x.dtype)) @ self.q)
        s = _positive_power(z[:, :-1] ** 2 + z[:, 1:] ** 2, 0.5)
        item = _positive_power(s, 0.5) * (1.0 + torch.sin(50.0 * _positive_power(s, 0.2)) ** 2)
        return item.mean(dim=1) ** 2 + 10.0 * _penalty(x)

    def _f17(self, x):
        return self._schaffers(x, 10.0)

    def _f18(self, x):
        return self._schaffers(x, 1000.0)

    def _f19(self, x):
        scale = max(1.0, math.sqrt(self.dim) / 8.0)
        z = scale * ((x - self.xopt) @ self.r) + 0.5
        s = 100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1.0) ** 2
        return 10.0 * torch.mean(s / 4000.0 - torch.cos(s), dim=1) + 10.0

    def _f20(self, x):
        xtrans = 2.0 * torch.sign(self.xopt) * x
        z = torch.cat((xtrans[:, :1], xtrans[:, 1:] + 0.25 * (xtrans[:, :-1] - 2.0 * torch.abs(self.xopt[:-1]))), dim=1)
        z = 100.0 * ((z - 2.0 * torch.abs(self.xopt)) * _scale(10.0, self.dim, x.device, x.dtype) + 2.0 * torch.abs(self.xopt))
        return -0.01 * torch.mean(z * torch.sin(_positive_power(torch.abs(z), 0.5)), dim=1) + 4.189828872724339 + 100.0 * _penalty(z / 100.0)

    def _gallagher(self, x):
        diff = x[:, None, :] - self.centers[None, :, :]
        rotated = torch.einsum('npd,dk->npk', diff, self.r.T)
        scaled = torch.einsum('npd,pde->npe', rotated, self.peak_matrices)
        transformed = torch.einsum('npd,dk->npk', scaled, self.r)
        peaks = self.peak_weights * torch.exp(-0.5 * torch.mean(transformed * diff, dim=2))
        return _tosz(10.0 - torch.max(peaks, dim=1).values) ** 2 + _penalty(x)

    def _f21(self, x):
        return self._gallagher(x)

    def _f22(self, x):
        return self._gallagher(x)

    def _f23(self, x):
        transform = self.q @ torch.diag(_scale(100.0, self.dim, x.device, x.dtype)) @ self.r
        z = (x - self.xopt) @ transform
        powers = 2.0 ** torch.arange(1, 33, device=x.device, dtype=x.dtype)
        expanded = z[:, :, None] * powers
        fractions = torch.sum(torch.abs(expanded - torch.round(expanded)) / powers, dim=2)
        indexes = torch.arange(1, self.dim + 1, device=x.device, dtype=x.dtype)
        factors = (1.0 + indexes * fractions) ** (10.0 / (self.dim**1.2))
        return 10.0 / (self.dim**2) * torch.prod(factors, dim=1) - 10.0 / (self.dim**2) + _penalty(x)

    def _f24(self, x):
        mu0 = 2.5
        s = 1.0 - 1.0 / (2.0 * math.sqrt(self.dim + 20.0) - 8.2)
        mu1 = -math.sqrt((mu0**2 - 1.0) / s)
        xtrans = 2.0 * torch.sign(self.xopt) * x
        transform = self.q @ torch.diag(_scale(100.0, self.dim, x.device, x.dtype)) @ self.r
        z = (xtrans - mu0) @ transform
        basin1 = torch.sum((xtrans - mu0) ** 2, dim=1)
        basin2 = self.dim + s * torch.sum((xtrans - mu1) ** 2, dim=1)
        return torch.minimum(basin1, basin2) + 10.0 * (self.dim - torch.sum(torch.cos(2.0 * torch.pi * z), dim=1)) + 1.0e4 * _penalty(x)


class BBOBTrainingProblem:
    """One fixed BBOB function, with POM's packed population interface."""

    def __init__(self, fid, dim, batch_size, device='cpu', dtype=torch.float32):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.problem = TorchBBOB(fid, dim, device, dtype)
        self.fid, self.dim, self.batch_size = fid, dim, batch_size
        self.device = self.problem.device
        self.dtype = dtype

    def random_population(self, popsize):
        if popsize < 2:
            raise ValueError('popsize must be at least 2')
        return self.calfitness(self.problem.gen_random_pop((self.batch_size, popsize, self.dim)))[0]

    def calfitness(self, x):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError('Expected [batch, population_size, dimension]')
        x = self.problem.repaire(x)
        fitness = self.problem.calfitness(x).unsqueeze(-1)
        if not torch.isfinite(fitness).all():
            raise FloatingPointError('Nonfinite fitness in BBOB F%d' % self.fid)
        return torch.cat((fitness, x), dim=-1), fitness


def self_check():
    """Check objective parity and gradients, including non-smooth zero points."""
    torch.set_num_threads(1)
    for dim in (2, 5, 10, 30):
        for fid in range(1, 25):
            torch.manual_seed(fid)
            problem = TorchBBOB(fid, dim)
            module = importlib.import_module('platemo_bbob_problems.BBOB_F%d' % fid)
            numpy_problem = getattr(module, 'BBOB_F%d' % fid)(D=dim)
            points = torch.cat((
                torch.rand(20, dim) * 10.0 - 5.0,
                torch.zeros(1, dim), torch.full((1, dim), -5.0), torch.full((1, dim), 5.0),
                problem.optimal_point().view(1, dim),
            ))
            points.requires_grad_()
            actual = problem.calfitness(points)
            with np.errstate(invalid='ignore'):
                expected = numpy_problem.cal_obj(points.detach().numpy().astype(np.float64)).reshape(-1)
            # Float32 rotations and oscillatory F16/F23 limit numerical parity.
            np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=3e-3, atol=2e-3,
                                       err_msg='F%d D%d' % (fid, dim))
            actual.sum().backward()
            assert torch.isfinite(points.grad).all(), 'Nonfinite gradient F%d D%d' % (fid, dim)
            assert actual[-1].abs().item() < 2e-3, 'Nonzero optimum F%d D%d' % (fid, dim)
            adapter = BBOBTrainingProblem(fid, dim, 2)
            x = points.detach().unsqueeze(0).repeat(2, 1, 1).requires_grad_()
            packed, fitness = adapter.calfitness(x * 2)
            assert packed.shape == (2, points.shape[0], dim + 1)
            assert torch.equal(packed[..., 1:], (x * 2).clamp(-5.0, 5.0))
            assert torch.equal(packed[..., :1], fitness)
            packed.sum().backward()
            assert torch.isfinite(x.grad).all()
    # Double precision must start from the original NumPy constants, rather
    # than promoting rotations that were already rounded to float32.
    for fid in range(1, 25):
        adapter = BBOBTrainingProblem(fid, 5, 2, dtype=torch.float64)
        points = adapter.random_population(8)[..., 1:].detach().requires_grad_()
        assert points.dtype == torch.float64
        for value in vars(adapter.problem).values():
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.dtype == torch.float64
        adapter.calfitness(points)[1].sum().backward()
        assert torch.isfinite(points.grad).all(), 'Float64 backward F%d' % fid
    reference_rotation = np.linalg.svd(np.random.RandomState(2).rand(5, 5),
                                       full_matrices=False)[0]
    np.testing.assert_array_equal(adapter.problem.r.numpy(), reference_rotation)
    points = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float64,
                          requires_grad=True)
    torch.autograd.gradcheck(TorchBBOB(10, 5, dtype=torch.float64).calfitness,
                             (points,), atol=1e-3, rtol=1e-3)
    print('PASS: BBOB F1--F24 NumPy parity and finite gradients, D=2/5/10/30; '
          'float64 constants/populations/backward and F10 numerical gradients')


if __name__ == '__main__':
    self_check()
