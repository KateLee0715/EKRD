import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
from parse import args
import os
from os.path import join
import random
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from teachers.KGCL import world

from time import time
from collections import defaultdict
import warnings
import collections
warnings.filterwarnings('ignore')


class Loader:
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.n_params = {
            'n_users': int(self.n_users),
            'n_items': int(self.n_items),
            'n_entities': int(self.n_entities),
            'n_nodes': int(self.n_nodes),
            'n_relations': int(self.n_relations)
        }
        self.Graph = None
        args.n_users = int(self.n_users)
        args.n_items = int(self.n_items)
        args.n_rel = int(self.n_relations)
        args.n_entities = int(self.n_entities)

    def load_data(self):
        directory = self.args.data_path + self.args.dataset + '/'

        print('reading train and test user-item set ...')
        train_cf = self.read_cf(directory + 'train.txt')
        test_cf = self.read_cf(directory + 'test.txt', False)
        self.remap_item(train_cf, test_cf)

        print('combinating train_cf and kg data ...')
        triplets, self.kg_dict, heads = self.read_triplets(directory + 'kg_final.txt')

        print('building the graph ...')
        graph, relation_dict = self.build_graph(train_cf, triplets)

        print('building the adj mat ...')
        adj_mat_list, norm_mat_list, mean_mat_list = self.build_sparse_relational_graph(relation_dict)

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.n_items))

        self.train_cf, self.test_cf = train_cf, test_cf
        self.graph = graph
        self.path = 'data/' + args.dataset
        self.adj_mat_list, self.norm_mat_list, self.mean_mat_list = adj_mat_list, norm_mat_list, mean_mat_list
        if args.teacher_model == 'KACL':
            self.kg_dict, self.relation_dict = self._load_kg(directory + 'kg_final.txt')
            self.adj_list = self._get_cf_adj_list()
            self.kg_adj_list, self.adj_r_list = self._get_kg_adj_list()
            self.lap_list = self._get_lap_list()
            self.kg_lap_list = self._get_kg_lap_list()
    
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                a = sp.csr_matrix((self.n_users, self.n_users))
                b = sp.csr_matrix((self.n_items, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat = sp.vstack([sp.hstack([a, R]), sp.hstack([R.transpose(), b])])
                adj_mat = (adj_mat != 0) * 1.0
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)


            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")
        return self.Graph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def read_cf(self, file_name, train=True):
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        if train:
            self.trainUser = np.array(inter_mat)[:, 0]
            self.trainItem = np.array(inter_mat)[:, 1]
        else:
            self.testUser = np.array(inter_mat)[:, 0]
            self.testItem = np.array(inter_mat)[:, 1]

        return np.array(inter_mat)

    def remap_item(self, train_data, test_data):
        self.n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
        self.n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

        train_user_set = defaultdict(list)
        test_user_set = defaultdict(list)
        for u_id, i_id in train_data:
            train_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in test_data:
            test_user_set[int(u_id)].append(int(i_id))

        self.train_user_set = train_user_set
        self.test_user_set = test_user_set
        
    def _bi_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    
    def _get_kg_lap_list(self, is_subgraph=False, subgraph_adj=None):
        if is_subgraph is True:
            adj_list = subgraph_adj
        else:
            adj_list = self.kg_adj_list
        if args.adj_type == 'bi':
            lap_list = [self._bi_norm_lap(adj) for adj in adj_list]
        else:
            lap_list = [self._si_norm_lap(adj) for adj in adj_list]
        return lap_list

    def _get_lap_list(self, is_subgraph=False, subgraph_adj=None):
        if is_subgraph is True:
            adj = subgraph_adj
        else:
            adj = self.adj_list
        if args.adj_type == 'bi':
            lap_list = self._bi_norm_lap(adj)
        else:
            lap_list = self._si_norm_lap(adj)
        return lap_list
    
    def _get_cf_adj_list(self, is_subgraph=False, dropout_rate=None):
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_items
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size=int(dropout_rate * len(a_rows)), replace=False)
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]

            vals = [1.] * len(a_rows) * 2
            rows = np.concatenate((a_rows, a_cols))
            cols = np.concatenate((a_cols, a_rows))
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))
            return adj

        R = _np_mat2sp_adj(self.train_cf, row_pre=0, col_pre=self.n_users)
        return R
    
    def _get_kg_adj_list(self, is_subgraph=False, dropout_rate=None):
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat):
            n_all = self.n_entities
            # single-direction
            a_rows = np_mat[:, 0]
            a_cols = np_mat[:, 1]
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size=int(dropout_rate * len(a_rows)), replace=False)
                # print(subgraph_id[:10])
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        for r_id in self.relation_dict.keys():
            # print(r_id)
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]))
            adj_mat_list.append(K)
            adj_r_list.append(r_id)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + self.n_relations)
        self.n_relations = self.n_relations * 2
        args.n_relations = self.n_relations
        # print(adj_r_list)
        return adj_mat_list, adj_r_list
    
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                # amazon weak[2, 6, 7, 11, 12, 22]
                # amazon rich[1, 8, 9, 14, 17]
                # lastfm weak[]
                kg[head].append((relation, tail))
                kg[tail].append((relation+self.n_relations, head))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        args.n_entities = self.n_entities
        kg_dict, relation_dict = _construct_kg(kg_np)
        self.n_triples = 0
        for r in relation_dict.keys():
            self.n_triples += len(relation_dict[r])
        return kg_dict, relation_dict

    def get_kg_dict(self, item_num):
        entity_num = args.entity_num_per_item
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))
                if (len(tails) > entity_num):
                    i2es[item] = torch.LongTensor(tails).cuda()[:entity_num]
                    i2rs[item] = torch.LongTensor(relations).cuda()[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.n_entities] * (entity_num - len(tails)))
                    relations.extend([self.n_relations] * (entity_num - len(relations)))
                    i2es[item] = torch.LongTensor(tails).cuda()
                    i2rs[item] = torch.LongTensor(relations).cuda()
            else:
                i2es[item] = torch.LongTensor([self.n_entities] * entity_num).cuda()
                i2rs[item] = torch.LongTensor([self.n_relations] * entity_num).cuda()
        return i2es, i2rs

    def read_triplets(self, file_name):

        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        if self.args.inverse_r:
            # get triplets with inverse direction like <entity, is-aspect-of, item>
            inv_triplets_np = can_triplets_np.copy()
            inv_triplets_np[:, 0] = can_triplets_np[:, 2]
            inv_triplets_np[:, 2] = can_triplets_np[:, 0]
            inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
            # consider two additional relations --- 'interact' and 'be interacted'
            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
            inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
            # get full version of knowledge graph
            triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        else:
            # consider two additional relations --- 'interact'.
            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
            triplets = can_triplets_np.copy()

        if self.args.teacher_model == 'KGCL':
            self.n_entities = max(triplets[:, 2]) + 2
        else:
            self.n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
        args.entities = self.n_entities
        self.n_nodes = self.n_entities + self.n_users
        self.n_relations = max(triplets[:, 1]) + 1
        kg_dict = defaultdict(list)
        for idx in range(triplets.shape[0]):
            h, r, t = triplets[idx][0], triplets[idx][1], triplets[idx][2]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return triplets, kg_dict, heads
    
    

    def build_graph(self, train_data, triplets):
        ckg_graph = nx.MultiDiGraph()
        rd = defaultdict(list)

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            rd[0].append([u_id, i_id])

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            rd[r_id].append([h_id, t_id])

        return ckg_graph, rd

    def build_sparse_relational_graph(self, relation_dict):
        def _bi_norm_lap(adj):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            # D^{-1}A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        adj_mat_list = []
        print("Begin to build sparse relation matrix ...")
        for r_id in tqdm(relation_dict.keys()):
            np_mat = np.array(relation_dict[r_id])
            if r_id == 0:
                cf = np_mat.copy()
                cf[:, 1] = cf[:, 1] + self.n_users  # [0, n_items) -> [n_users, n_users+n_items)
                vals = [1.] * len(cf)
                adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(self.n_nodes, self.n_nodes))
            else:
                vals = [1.] * len(np_mat)
                adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(self.n_nodes, self.n_nodes))
            adj_mat_list.append(adj)

        norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
        mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
        # interaction: user->item, [n_users, n_entities]
        norm_mat_list[0] = norm_mat_list[0].tocsr()[:self.n_users, self.n_users:].tocoo()
        mean_mat_list[0] = mean_mat_list[0].tocsr()[:self.n_users, self.n_users:].tocoo()

        return adj_mat_list, norm_mat_list, mean_mat_list


class TrnData(Dataset):
    def __init__(self, trnUsrs, trnItms, mat):
        self.rows = trnUsrs
        self.cols = trnItms
        self.dokmat = mat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self, m_items):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(m_items)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(Dataset):
    def __init__(self, tstUsrs, tstItms, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        tstLocs = [None] * trnMat.shape[0]
        rows = set()

        for i in range(len(tstUsrs)):
            row = tstUsrs[i]
            col = tstItms[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()

            tstLocs[row].append(col)
            rows.add(row)
        rows = np.array(list(rows))
        self.tstUsrs = rows
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

