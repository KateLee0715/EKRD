import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from parse import args
from EKRD import Model
from teachers.KGCL.kgcl import KGCL
from teachers.KGIN.KGIN import Recommender
from teachers.KGAT.KGAT import KGAT
from teachers.CKE.CKE import CKE
from teachers.KACL import GNN
from data_loader import Loader, TrnData, TstData
from torch.utils.data import DataLoader
import numpy as np
import pickle
from Utils.Utils import *
import os
from teachers.KGCL import world
from os.path import join
import time
import faulthandler
import dgl
# from torchsummary import summary


class Coach:
    def __init__(self, dataset: Loader):
        # self.handler = handler
        self.dataset = dataset
        trnDataset = TrnData(dataset.trainUser, dataset.trainItem, dataset.UserItemNet)
        tstDataset = TstData(dataset.testUser, dataset.testItem, dataset.UserItemNet)
        self.trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=True, num_workers=0)
        self.tstLoader = DataLoader(tstDataset, batch_size=args.tstBat, shuffle=False, num_workers=0)
        
        if args.teacher_model == 'KACL':
            adjM = dataset.lap_list
            g = dgl.DGLGraph(adjM)
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            self.g = g.to('cuda')

            kg_adjM = sum(dataset.kg_lap_list)
            kg2 = dgl.DGLGraph(kg_adjM)
            kg2 = dgl.remove_self_loop(kg2)
            kg2 = dgl.add_self_loop(kg2)
            self.kg2 = kg2.to('cuda')
        
        n_items = dataset.n_items
        self.kg, self.item2relations = dataset.get_kg_dict(n_items)
        self.n_entities = dataset.n_entities
        
        # print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        # print(summary(self.model))
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                starttime = time.time()
                reses = self.testEpoch()
                endtime = time.time()
                # print(f"It took {endtime-starttime:.2f} seconds to compute")
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def prepareModel(self):
        teacher = self.loadTeacher()
        if args.teacher_model == 'KACL':
            self.model = Model(teacher, self.kg2, self.g).cuda()
        else:
            self.model = Model(teacher, self.dataset).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.trnLoader
        trnLoader.dataset.negSampling(self.dataset.n_items)
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            loss, losses = self.model.calcLoss(ancs, poss, negs, self.kg, self.item2relations, self.n_entities)
            epLoss += loss.item()
            epPreLoss += losses['mainLoss'].item()
            regLoss = losses['regLoss'].item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False,
                oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.tstLoader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            starttime = time.time()
            allPreds = self.model.student.testPred(usr, trnMask)
            
            t.cuda.synchronize()
            endtime = time.time()
            # print(f"It took {endtime-starttime:.2f} seconds to compute")
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def loadTeacher(self):
        # ckp = t.load('../../Models/teacher_' + args.teacher_model + '.mod')
        # teacher = ckp['model']
        if args.teacher_model == 'KACL':
            import sys
            sys.path.append('./teachers/KACL/')
            teacher = t.load('./Models/' + args.teacher_model + '/' + args.teacher_model + '_' + args.dataset + '.pth', map_location='cuda:0')
            return teacher
        
        ckp = t.load('./Models/' + args.teacher_model + '/' + args.teacher_model + '_' + args.dataset + '.pth', map_location='cuda:0')
        if args.teacher_model == 'KGCL':
            teacher = KGCL(world.config, self.dataset)
            teacher.load_state_dict(ckp)
        elif args.teacher_model == 'KGIN':
            teacher = Recommender(self.dataset.n_params, args, self.dataset.graph, self.dataset.mean_mat_list[0], self.dataset)
            teacher.load_state_dict(ckp)
        elif args.teacher_model == 'KGAT':
            teacher = KGAT(args, self.dataset.n_users, self.dataset.n_entities, self.dataset.n_relations)
            teacher.load_state_dict(ckp['model_state_dict'])
        elif args.teacher_model == 'CKE':
            teacher = CKE(args, self.dataset.n_users, self.dataset.n_items, self.dataset.n_entities, self.dataset.n_relations)
            teacher.load_state_dict(ckp['model_state_dict'])
            
        return teacher

    def saveHistory(self):
        if args.epoch == 0:
            return
        directory = './History/' + args.teacher_model + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + args.teacher_model + '_' + args.dataset + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, directory + args.teacher_model + '_' + args.dataset + '.mod')
        log('Model Saved: %s' % directory + args.teacher_model + '_' + args.dataset)

    def loadModel(self):
        ckp = t.load('./Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    logger.saveDefault = True
    faulthandler.enable()

    log('Start')
    # handler = DataHandler()
    # handler.LoadData()
    dataset = Loader(args)
    log('Load Data')

    coach = Coach(dataset)
    coach.run()