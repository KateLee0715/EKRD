import torch as t
from torch import nn
import torch.nn.functional as F
from parse import args
from Utils.Utils import infoNCE, KLDiverge, pairPredict, calcRegLoss, mainLosses, _L2_loss_mean, KLDiverge2, angleWise, \
    huberLoss
import datetime
from data_loader import Loader
import scipy.sparse as sp
import numpy as np
# from temp_global import Global_T

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self, teacher, *input):
        super(Model, self).__init__()

        self.teacher = teacher
        self.student = MLPNet()
        if args.teacher_model == 'KACL':
            self.kg, self.g = input
            self.W = nn.Linear(2 * 144, 144)
            self.trans = nn.Linear(144 * 144, 144)
        elif args.teacher_model == 'KGAT':
            self.dataset = input[0]
            self.n_entities = self.dataset.n_entities
            self.W = nn.Linear(2 * (sum(eval(args.conv_dim_list)) + args.latdim), args.latdim)
        else:
            self.dataset = input[0]
            self.n_entities = self.dataset.n_entities
            self.W = nn.Linear(2 * args.latdim, args.latdim)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self):
        pass
    
    def getTeaAttn(self, entiEmbs, relEmbs, kg, item2relations, n_entities):
        if args.teacher_model == 'KACL':
            relEmbs = self.trans(relEmbs)
        entiEmbs = t.cat([entiEmbs, t.zeros((1, entiEmbs.shape[1])).cuda()], dim=0)
        relEmbs = t.cat([relEmbs, t.zeros((1, relEmbs.shape[1])).cuda()], dim=0)
        item_embs = entiEmbs[t.LongTensor(list(kg.keys()))]
        item_entities = t.stack(list(kg.values()))
        args.item_entities = item_entities
        item_relations = t.stack(list(item2relations.values()))
        args.item_relations = item_relations
        entity_embs = entiEmbs[item_entities]
        relation_embs = relEmbs[item_relations-1]
        padding_mask = t.where(item_entities != n_entities, t.ones_like(item_entities),
                                   t.zeros_like(item_entities)).float()
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        We = entity_embs
        a_input = t.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = t.multiply(self.W(a_input), relation_embs).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)
        # e_input = t.multiply(Wh * We, relation_embs).sum(-1)
        # e = self.leakyrelu(e_input)
        zero_vec = -9e15 * t.ones_like(e)
        attention = t.where(padding_mask.cuda() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        return attention

    def calcLoss(self, ancs, poss, negs, kg, item2relations, n_entities):
        uniqAncs = t.unique(ancs)
        uniqPoss = t.unique(poss)
        if args.teacher_model == 'KGIN':
            tiEmbeds, tuEmbeds, urelEmbeds = self.teacher.generate()
            urelEmbeds = urelEmbeds.detach()
        elif args.teacher_model == 'KGCL':
            tuEmbeds, tiEmbeds, entiEmbs, urelEmbeds = self.teacher()
            entiEmbs = entiEmbs.weight.detach()
            urelEmbeds = urelEmbeds.weight.detach()
        elif args.teacher_model == 'KACL':
            allEmbeds, entiEmbs, urelEmbeds = self.teacher('test', self.g, self.kg)
            tuEmbeds = allEmbeds[:args.n_users]
            tiEmbeds = allEmbeds[args.n_users:]
            urelEmbeds = urelEmbeds.detach()
            entiEmbs = entiEmbs.detach()
        elif args.teacher_model == 'KGAT':
            allEmbeds, urelEmbeds = self.teacher(mode='getEmbeds')
            tiEmbeds = allEmbeds[:args.n_entities]
            tuEmbeds = allEmbeds[args.n_entities:]
        elif args.teacher_model == 'CKE':
            tuEmbeds, tiEmbeds, urelEmbeds = self.teacher(is_train=False)
            urelEmbeds = urelEmbeds.detach()
        tiEmbeds = tiEmbeds.detach()
        tuEmbeds = tuEmbeds.detach()
        if args.teacher_model == 'KGCL':
            tAtten = self.getTeaAttn(entiEmbs, urelEmbeds, kg, item2relations, n_entities)
        elif args.teacher_model == 'KACL':
            tAtten = self.getTeaAttn(entiEmbs, urelEmbeds.view(args.n_relations+1,-1), kg, item2relations, n_entities)
        else:
            tAtten = self.getTeaAttn(tiEmbeds, urelEmbeds, kg, item2relations, n_entities)
        suEmbeds, siEmbeds = self.student()
        sAtten = self.student.getAttention()
        
        tAtten = tAtten.detach()
        

        rdmUsrs = t.randint(args.n_users, [args.topRange])  # ancs

        rdmItms1 = t.randint_like(rdmUsrs, args.n_items)
        rdmItms2 = t.randint_like(rdmUsrs, args.n_items)

        if args.teacher_model == 'KGIN':
            tEmbedsLst = self.teacher.generate(getMultOrder=True)
            highEmbeds = sum(tEmbedsLst[2:])
        elif args.teacher_model == 'KGCL':
            tEmbedsLst = self.teacher(getMultOrder=True)
            highEmbeds = sum(tEmbedsLst[2:])
        elif args.teacher_model == 'KACL':
            tEmbedsLst = self.teacher('test', self.g, self.kg, getMultOrder=True)        
            highEmbeds = tEmbedsLst
        elif args.teacher_model == 'KGAT':
            tEmbedsLst = self.teacher(mode='getEmbeds', getMultOrder=True)
            highEmbeds = tEmbedsLst
        elif args.teacher_model == 'CKE':
            tEmbedsLst = self.teacher(is_train=False, getMultOrder=True)
            highuEmbeds, highiEmbeds = tEmbedsLst
        
        if args.teacher_model == 'CKE':
            highiEmbeds = highiEmbeds.detach()
            highuEmbeds = highuEmbeds.detach()
        else:
            highuEmbeds = highEmbeds[:args.n_users].detach()
            highiEmbeds = highEmbeds[args.n_users:].detach()
        contrastDistill = 0
        if args.teacher_model == 'KGAT':
            len = sum(eval(args.conv_dim_list)[1:])
            contrastDistill += (infoNCE(highuEmbeds, suEmbeds[:,-len:], uniqAncs, args.tempcd) + infoNCE(highiEmbeds, siEmbeds[:,-len:],
                                                                                             uniqPoss,
                                                                                             args.tempcd)) * args.cdreg
        else:
            contrastDistill += (infoNCE(highuEmbeds, suEmbeds, uniqAncs, args.tempcd) + infoNCE(highiEmbeds, siEmbeds,
                                                                                             uniqPoss,
                                                                                             args.tempcd)) * args.cdreg

  
        eEmbeds, uEmbeds = self.student.getGrad()

        LSPLoss = _L2_loss_mean(tAtten[uniqPoss] - sAtten[uniqPoss]) * args.lsreg
        
        tAngles2 = angleWise(tuEmbeds, tiEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)
        sAngles2 = angleWise(suEmbeds, siEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
        angleLoss = huberLoss(tAngles2, sAngles2) * args.aglreg

        # soft-target-based distillation
        tpairPreds = self.teacher.pairPredictwEmbeds(tuEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)
        spairPreds = self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
        softTargetDistill = KLDiverge(tpairPreds, spairPreds, args.tempsoft) * args.softreg

        preds = self.student.pointPosPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss)
        negPreds = self.student.pointPosPredictwEmbeds(suEmbeds, siEmbeds, ancs, negs)
        mainLoss = (preds - negPreds).sigmoid().log().mean()
        mainLoss = -mainLoss


        # weight-decay reg
        regParams = [eEmbeds, uEmbeds]
        regLoss = calcRegLoss(params=regParams) * args.reg

        loss = mainLoss + contrastDistill + softTargetDistill + regLoss + LSPLoss + angleLoss
        losses = {'mainLoss': mainLoss, 'contrastDistill': contrastDistill, 'softTargetDistill': softTargetDistill,
                  'regLoss': regLoss}
        return loss, losses


class BLMLP(nn.Module):
    def __init__(self):
        super(BLMLP, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.5)
        self.dropout = nn.Dropout(0.5)
        self.entiEmbs = nn.Parameter(init(t.empty(args.entities+1, args.latdim)))
        self.att = nn.Parameter(init(t.empty(args.n_items, args.entity_num_per_item)))

    def forward(self, embeds):
        pass

    
    def featureExtract(self, embeds):
        return embeds

    def pairPred(self, embeds1, embeds2):
        return (self.featureExtract(embeds1) * self.featureExtract(embeds2)).sum(dim=-1)

    def crossPred(self, embeds1, embeds2):
        return self.featureExtract(embeds1) @ self.featureExtract(embeds2).T

    def getItemEmbeds(self):
        args.item_entities = args.item_entities.long()
        entity_embs = self.entiEmbs[args.item_entities]
        att = F.softmax(self.att, dim=1)
        entity_emb_weighted = t.bmm(att.unsqueeze(1), entity_embs).squeeze()
        return entity_emb_weighted+self.entiEmbs[:args.n_items]

    def getRelEmbeds(self):
        return self.entiEmbs

    def getAttention(self):
        att = F.softmax(self.att, dim=1)
        return att


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.uEmbeds = nn.Parameter(init(t.empty(args.n_users, args.latdim)))
        self.MLP = BLMLP()
        self.act = nn.LeakyReLU(negative_slope=0.5)

    def forward(self):
        uEmbeds = self.uEmbeds
        iEmbeds = self.getMid()
        return uEmbeds, iEmbeds

    def getMid(self):
        return self.MLP.getItemEmbeds()

    def pointPosPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        nume = self.MLP.pairPred(ancEmbeds, posEmbeds)
        return nume

    def pointNegPredictwEmbeds(self, embeds1, embeds2, nodes1, temp=1.0):
        pckEmbeds1 = embeds1[nodes1]
        preds = self.MLP.crossPred(pckEmbeds1, embeds2)
        return t.exp(preds / temp).sum(-1)

    def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        posPreds = self.MLP.pairPred(ancEmbeds, posEmbeds)
        negPreds = self.MLP.pairPred(ancEmbeds, negEmbeds)
        return posPreds - negPreds

    def predAll(self, pckUEmbeds, iEmbeds):
        return self.MLP.crossPred(pckUEmbeds, iEmbeds)

    def testPred(self, usr, trnMask):
        uEmbeds, iEmbeds = self.forward()
        allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
        return allPreds

    def getGrad(self):
        return self.MLP.getRelEmbeds(), self.uEmbeds

    def getAttention(self):
        return self.MLP.getAttention()
