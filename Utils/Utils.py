import torch as t
import torch.nn.functional as F


def innerProduct(usrEmbeds, itmEmbeds):
    return t.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(params=None, model=None):
    ret = 0
    if params is not None:
        for W in params:
            ret += W.norm(2).square()
    if model is not None:
        for W in model.parameters():
            ret += W.norm(2).square()
    # ret += (model.usrStruct + model.itmStruct)
    return ret


def infoNCE(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return (-t.log(nume / deno)).mean()


def KLDiverge(tpreds, spreds, distillTemp):
    tpreds = (tpreds / distillTemp).sigmoid()
    spreds = (spreds / distillTemp).sigmoid()
    return -(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean()


def pointKLDiverge(tpreds, spreds):
    return -(tpreds * spreds.log()).mean()


def _L2_loss_mean(x):
    return t.mean(t.sum(t.pow(x, 2), dim=1, keepdim=False) / 2.)


def mainLosses(posPreds, negPreds):
    preds = 1.0 - posPreds + negPreds
    return t.where(preds > 0, preds, t.zeros_like(preds)).mean()


def KLDiverge2(tLS, sLS, temp):
    tLS = tLS / temp
    sLS = sLS / temp
    # return (sLS * (sLS / tLS).log()).mean()
    return (sLS * ((sLS + 1e-8).log() - (tLS + 1e-8).log())).mean()


def angleWise(rEmbeds, iEmbeds, eEmbeds, ancs, poss, negs):
    ancsEmbs = rEmbeds[ancs]
    pEmbeds = iEmbeds[poss]
    nEmbeds = eEmbeds[negs]
    embeds1 = F.normalize(ancsEmbs - pEmbeds, p=2)
    embeds2 = F.normalize(ancsEmbs - nEmbeds, p=2)
    return (embeds1 * embeds2).sum(-1)


def huberLoss(x, y):
    return t.where(t.abs(x - y) <= 1, 0.5 * (x - y) * (x - y), t.abs(x - y) - 0.5).mean()
# if t.abs(x-y) <= 1:
# 	return (0.5 * (x - y) * (x - y)).mean()
# else:
# 	return (t.abs(x-y) - 0.5).mean()

