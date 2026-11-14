import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

class TSRepository(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        features = self.features.cpu().numpy()
        knn_model = NearestNeighbors(n_neighbors=features.shape[0],
                                 algorithm='brute',
                                 n_jobs=-1)
        knn_model.fit(features)

        distances, indices = knn_model.kneighbors(features, return_distance=True)
        k_furthest_neighbours = []
        k_nearest_neighbours = []
        for i in range(features.shape[0]):
            # sort the neighbours based on their distance to the point
            sorted_indices = np.argsort(distances[i])
            # get the k furthest neighbours for each point
            k_furthest_neighbours.append(indices[i][sorted_indices[-topk:]])
            k_nearest_neighbours.append(indices[i][sorted_indices[1:topk+1]])

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)

            return k_furthest_neighbours, k_nearest_neighbours, accuracy
        
        else:
            return k_furthest_neighbours, k_nearest_neighbours

    def furthest_nearest_neighbors(self, topk, chunk=50000):
        """Per-sample k nearest and k furthest neighbours in representation space,
        as described in the paper (Sec. 3.2).

        The released code ranked all samples by distance to ONE random query
        vector and paired sample i with rank i (upstream issue #8), so the
        "neighbours" were unrelated to the anchor and classification collapsed.

        Returns (k_furthest, k_nearest), each int64 [N, topk]:
          k_nearest[i]  : nearest first, excluding i itself
          k_furthest[i] : ordered so the LAST column is the furthest
        (NeighborsDataset uses N[:, :Q] and F[:, -Q:]).
        """
        x = np.ascontiguousarray(self.features.numpy().astype(np.float32))
        n, d = x.shape
        norms = np.linalg.norm(x, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), 'expects L2-normalised features'

        index_l2 = faiss.IndexFlatL2(d)
        index_l2.add(x)
        # unit vectors: ||a-b||^2 = 2 - 2<a,b>, so furthest = smallest <a,b> = largest <-a,b>
        index_ip = faiss.IndexFlatIP(d)
        index_ip.add(x)

        k_nearest = np.empty((n, topk), dtype=np.int64)
        k_furthest = np.empty((n, topk), dtype=np.int64)
        for s0 in range(0, n, chunk):
            q = x[s0:s0 + chunk]
            ids_self = np.arange(s0, s0 + q.shape[0])
            _, nn = index_l2.search(q, topk + 1)
            for r in range(q.shape[0]):            # drop self (may not be first when duplicates exist)
                row = nn[r][nn[r] != ids_self[r]]
                k_nearest[s0 + r] = row[:topk]
            _, fn = index_ip.search(-q, topk)       # most distant first
            k_furthest[s0:s0 + q.shape[0]] = fn[:, ::-1]  # last column = furthest
            print('Mine neighbours [%d/%d]' % (min(s0 + chunk, n), n))

        return k_furthest, k_nearest


    def reset(self):
        self.ptr = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if not torch.is_tensor(targets): targets = torch.from_numpy(targets)
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
