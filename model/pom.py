"""Published POM network with finite zero-normalization gradients.

Parameter names and numerical operations match the audited release model."""
import torch
import torch.nn as nn

def sortIndivBND(batchPop):
    """Sort complete rows by fitness without detaching gradients."""
    indices = batchPop[..., 0].argsort(dim=1)
    return batchPop.gather(1, indices.unsqueeze(-1).expand_as(batchPop))

def finite_source_norm(x):
    """Author's exact sqrt(sum(x**2)) forward with a finite zero derivative."""
    squared = torch.sum(x ** 2, dim=-1, keepdim=True)
    positive = squared > 0
    safe = torch.where(positive, squared, torch.ones_like(squared))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(squared))

class SMBND(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batchpop1, batchpop2, minimize=True):
        """
        实现选择操作,默认是最小化函数，若minimize=False,则为最大化目标值问题
        batchpop1 是offpop
        """
        better = batchpop1[..., 0] < batchpop2[..., 0]
        if not minimize:
            better = batchpop1[..., 0] > batchpop2[..., 0]
        return torch.where(better.unsqueeze(-1), batchpop1, batchpop2)

class GBMutModel(nn.Module):

    def __init__(self, hdim=1000):
        super().__init__()
        fdim = 2
        hdim2 = 100
        qkdim = hdim
        self.fq1 = nn.Sequential(nn.Linear(hdim2, qkdim), nn.Tanh())
        self.fk1 = nn.Sequential(nn.Linear(hdim2, qkdim), nn.Tanh())
        self.w = nn.Sequential(nn.Linear(fdim, hdim2), nn.Tanh())

    def forward(self, x):
        x = self.w(x)
        q1 = self.fq1(x)
        k1 = self.fk1(x)
        A = torch.matmul(q1, torch.transpose(k1, -1, -2)) / torch.sqrt(torch.tensor(k1.shape[-1], device=k1.device))
        A = torch.tanh(A)
        mask = torch.rand_like(A)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        y = torch.eye(mask.shape[-2], mask.shape[-1], device=A.device, dtype=A.dtype)
        y = torch.unsqueeze(y, 0)
        mask = mask + y
        mask[mask == 2] = 1
        A = A * mask
        return A

class GBLearnCrRate(nn.Module):

    def __init__(self, hdim=100):
        super().__init__()
        inputdim = 3
        outputdim = 1
        self.net = nn.Sequential(nn.Linear(inputdim, hdim), nn.ReLU(), nn.LayerNorm(hdim), nn.Linear(hdim, outputdim), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x

class GB_GLHF(nn.Module):

    def __init__(self, popsize=100, selmod='1-to-1', cr_policy='learned', muthdim=1000, crhdim=4):
        super().__init__()
        if popsize < 2:
            raise ValueError('popsize must be at least 2')
        self.popsize = popsize
        self.cr_policy = cr_policy
        ranks = torch.arange(popsize, dtype=torch.float)
        ranks = (ranks - ranks.mean()) / ranks.std()
        self.register_buffer('ranks', ranks.view(1, -1, 1), persistent=False)
        ranks2 = torch.arange(popsize * 2 - 1, dtype=torch.float)
        ranks2 = (ranks2 - ranks2.mean()) / ranks2.std()
        self.register_buffer('ranks2', ranks2.view(1, -1, 1), persistent=False)
        self.sm = SMBND()
        self.agen = GBMutModel(hdim=muthdim)
        self.crgen = GBLearnCrRate(hdim=crhdim)
        self.selmod = selmod
        self.adapter = None

    def setAdapter(self, adapter):
        self.adapter = adapter

    def RoulleteSelectWithElite(self, pop, popsize):
        """
        保留精英的轮盘赌选择
        pop -> b,n,d+1
        """
        (b, n, d) = pop.shape
        fitness = pop[..., 0]
        p = 1 - torch.softmax(fitness, dim=-1)
        selected_index = torch.multinomial(p, popsize, replacement=False)
        offs = []
        for (idx, batchpop) in enumerate(pop):
            index = selected_index[idx]
            tmp = batchpop[index]
            tmp = torch.unsqueeze(tmp, 0)
            offs.append(tmp)
        offs = torch.cat(offs)
        return offs

    def genGBMutToken(self, x, ranks):
        (b, n) = x.shape
        miu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        x = (x - miu) / std
        x = x.view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks[..., :n, :]), dim=-1)
        return x

    def genCrRankToken(self, fitness):
        """
        input:b,n
        """
        (b, n) = fitness.shape
        (_, indexs) = torch.sort(fitness, dim=-1)
        ranks = self.ranks.repeat(b, 1, 1)
        miu = torch.mean(fitness, dim=-1, keepdim=True)
        std = torch.std(fitness, dim=-1, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        fitness = (fitness - miu) / std
        fitness = fitness.view(b, -1, 1)
        newRanks = []
        for bid in range(b):
            index = indexs[bid, ...]
            index = torch.unsqueeze(index, -1)
            rank = ranks[bid, ...]
            tmp = torch.cat((index, rank[:n, ...]), dim=-1)
            (_, tmp_index) = torch.sort(tmp[:, 0], dim=0)
            tmp = torch.index_select(tmp, 0, tmp_index)
            token = torch.cat((fitness[bid], torch.unsqueeze(tmp[:, 1], -1)), -1)
            newRanks.append(token)
        token = torch.stack(newRanks, 0)
        return token

    def genCrRankTokenWithoutFit(self, father, off):
        """
        input:b,n,改为不用适应度的版本
        """
        (b, n, d) = father.shape
        tmp = torch.cat((father, off), dim=-1)
        ave = torch.mean(tmp, dim=-1, keepdim=True)
        std = torch.std(tmp, dim=-1, keepdim=True)
        tmp = (tmp - ave) / (std + 1e-08)
        fpop = tmp[:, :, :d]
        opop = tmp[:, :, d:]
        fMod = finite_source_norm(fpop)
        oMod = finite_source_norm(opop)
        item = fMod * oMod
        item = torch.clamp(item, min=1e-08)
        sim = torch.sum(fpop * opop, dim=-1, keepdim=True) / item
        sim = sim.view(b, n)
        simstd = torch.std(sim, dim=-1, keepdim=True)
        simstd = torch.where(simstd > 0, simstd, torch.ones_like(simstd))
        sim = (sim - torch.mean(sim, dim=-1, keepdim=True)) / simstd
        sim = torch.unsqueeze(sim, -1)
        return sim

    def forward(self, batchPop=None, problem=None):
        """
        输入：
        已经有适应度的种群（batch,n,d）
        """
        paramcr = None
        (b, n, d) = batchPop.shape
        if n != self.popsize or d < 2:
            raise ValueError('Expected [batch, popsize, dimension + 1], fitness in column 0')
        batchPop = sortIndivBND(batchPop)
        batchfitness = batchPop[:, :, 0]
        fitnesstoken = self.genGBMutToken(batchfitness, self.ranks)
        Atoken = fitnesstoken
        params = self.agen(Atoken)
        batchChrom = batchPop[:, :, 1:]
        vchrom = torch.matmul(params, batchChrom)
        if self.cr_policy == 'learned':
            vpopToken = self.genCrRankTokenWithoutFit(batchChrom, vchrom)
            token = torch.cat((fitnesstoken, vpopToken), dim=-1)
            cr = self.crgen(token)
            cr = torch.unsqueeze(cr, -2)
            paramcr = cr.view(b, n)
            cr = cr.repeat(1, 1, batchChrom.shape[-1], 1)
            r = torch.rand_like(cr)
            select_mask = torch.cat((cr, r), dim=-1)
            select_mask = torch.nn.functional.gumbel_softmax(select_mask, tau=1, hard=True, eps=1e-10, dim=-1)
            offpopChrom = select_mask[..., 0] * batchChrom + select_mask[..., 1] * vchrom
        else:
            mask = torch.rand_like(batchChrom)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            offpopChrom = mask * batchChrom + (1 - mask) * vchrom
        if self.adapter:
            offpopChrom = self.adapter(offpopChrom)
        (offpop, _) = problem.calfitness(offpopChrom)
        mixpop = torch.cat((batchPop, offpop), dim=1)
        if self.selmod == 'learned':
            pass
        if self.selmod == '1-to-1':
            nextPop = self.sm(offpop, batchPop)
        if self.selmod == '轮盘赌':
            mixpop = sortIndivBND(mixpop)
            elitePop = mixpop[:, :1, :]
            mixpop = mixpop[:, 1:, :]
            selectednextPop = self.RoulleteSelectWithElite(mixpop, batchPop.shape[1] - 1)
            nextPop = torch.cat((elitePop, selectednextPop), dim=1)
        return (nextPop, params, paramcr)
