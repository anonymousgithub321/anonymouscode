import numpy as np
from collections import defaultdict
import numba as nb

def evaluate2(train, probe, r, k):
    nuser, nitem = np.shape(r)
    posprobe = defaultdict(list)
    for line in probe:
        posprobe[line[0]].append(line[1])
    userlist = list(posprobe.keys())
    testsize = len(userlist)

    postrain = defaultdict(list)
    for line in train:
        postrain[line[0]].append(line[1])
    for user in userlist:
        r[user, postrain[user]] = -9999

    pred = np.argsort(r, axis=1)[:, ::-1]

    recall = []
    precision = []
    map = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    ll += 1
            recall_tmp.append(np.float(np.sum(predict_tmp > 0)) / len(posprobe[user]))
            precision_tmp.append(np.float(np.sum(predict_tmp > 0)) / kk)
            map_tmp.append(np.sum(predict_tmp / (np.array(range(kk)) + 1)) / kk)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))

    return recall, precision, map


def evaluate3(postrain, posprobe, r, k):  # 隐式反馈最终版
    userlist = list(posprobe.keys())
    for user in userlist:
        r[user, postrain[user]] = -9999

    pred = np.argsort(r, axis=1)[:, ::-1]

    recall = []
    precision = []
    map = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    ll += 1
            recall_tmp.append(np.float(np.sum(predict_tmp > 0)) / len(posprobe[user]))
            precision_tmp.append(np.float(np.sum(predict_tmp > 0)) / kk)
            map_tmp.append(np.sum(predict_tmp / (np.array(range(kk)) + 1)) / kk)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))

    return recall, precision, map


def evaluate4(postrain, posprobe, r, k):  # 加入MRR，NDCG, HR
    userlist = list(posprobe.keys())
    for user in userlist:
        r[user, postrain[user]] = -9999

    pred = np.argsort(r, axis=1)[:, ::-1]

    recall = []
    precision = []
    map = []
    mrr = []
    ndcg = []
    hr = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        mrr_tmp = []
        ndcg_tmp = []
        hr_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            mrr_tmp.append(np.sum(boo_tmp/rank))
            hr_tmp.append(np.float(sum_tmp > 0))
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            # idcg[idcg == 0.] = 1.
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))
        mrr.append(np.mean(mrr_tmp))
        hr.append(np.mean(hr_tmp))
        ndcg.append(np.mean(ndcg_tmp))

    return recall, precision, map, mrr, ndcg, hr

@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def fastSort(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b

def evaluate11(posprobe, r, k):  # 采用numba加速，加入MRR，NDCG, HR,传入的评分矩阵已经去掉了训练集的正样本
    userlist = list(posprobe.keys())
    # for user in userlist:
    #     r[user, postrain[user]] = -9999

    r[userlist, :] = fastSort(r[userlist, :])
    pred = r[:, ::-1][:, 0:k[-1]]

    recall = []
    precision = []
    map = []
    # mrr = []
    ndcg = []
    # hr = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        # mrr_tmp = []
        ndcg_tmp = []
        # hr_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            # recall_tmp.append(sum_tmp / max_r)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            # mrr_tmp.append(np.sum(boo_tmp/rank))
            # hr_tmp.append(np.float(sum_tmp > 0))
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            # idcg[idcg == 0.] = 1.
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))
        # mrr.append(np.mean(mrr_tmp))
        # hr.append(np.mean(hr_tmp))
        ndcg.append(np.mean(ndcg_tmp))

    # return recall, precision, map, mrr, ndcg, hr
    return recall, precision, map, ndcg

def evaluate12(postrain, posprobe, r, k):  # 采用numba加速，加入MRR，NDCG, HR,为适配老版本代码相比evaluate11恢复了postrain
    userlist = list(posprobe.keys())
    for user in userlist:
        r[user, postrain[user]] = -9999

    r[userlist, :] = fastSort(r[userlist, :])
    pred = r[:, ::-1][:, 0:k[-1]]

    recall = []
    precision = []
    map = []
    # mrr = []
    ndcg = []
    # hr = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        # mrr_tmp = []
        ndcg_tmp = []
        # hr_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            # recall_tmp.append(sum_tmp / max_r)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            # mrr_tmp.append(np.sum(boo_tmp/rank))
            # hr_tmp.append(np.float(sum_tmp > 0))
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            # idcg[idcg == 0.] = 1.
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))
        # mrr.append(np.mean(mrr_tmp))
        # hr.append(np.mean(hr_tmp))
        ndcg.append(np.mean(ndcg_tmp))

    # return recall, precision, map, mrr, ndcg, hr
    return recall, precision, map, ndcg


