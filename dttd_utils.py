import numpy as np
from math import sqrt
from tensorly.base import unfold, fold
from config import *


def latent_mode_dot(R_s_u, U_s, mode):
    new_shape = list(R_s_u.shape)
    new_shape[mode] = U_s.shape[1]
    res = torch.matmul(U_s.permute(1, 0), unfold(R_s_u, mode))
    tensor = fold(res, mode, new_shape)
    return tensor


def getCoreTensor(R_s_u, factors):
    modes = list(range(len(R_s_u.shape)))
    tensor = R_s_u
    for mode in modes:
        tensor = latent_mode_dot(tensor, factors[mode], mode)
    return tensor


def getCore(Core_s, factors, R_s_u):
    divfactors = []
    for i in factors:
        i = torch.matmul(i.permute(1, 0), i)
        divfactors.append(i)
    divcore = getCoreTensor(Core_s, divfactors)
    Core_div = getCoreTensor(R_s_u, factors)
    e = torch.div(Core_div, divcore)
    Core_s = torch.mul(e, Core_s)
    return Core_s


def HUpdate(R_s, C_s, U_s, V_s, H, Critemp):
    for i in range(EPOCH_H):
        temp = torch.pow(
            torch.div(
                U_s.t().matmul(R_s.mul(C_s)).matmul(V_s).mul(Critemp)
                , U_s.t().matmul(C_s.mul(U_s.matmul(H).matmul(V_s.t()))).matmul(V_s).mul(
                    torch.matmul(Critemp, Critemp)) + eps
            )
            , 1 / 3
        )
    H = H - 0.01 * (H - H.mul(temp))
    return H.to(device)


def alignment(U_s, V_s, Criteria_s, U_t, V_t, Criteria_t):
    pairs = [[U_s, U_t], [V_s, V_t], [Criteria_s, Criteria_t]]
    aligned_s = []
    aligned_t = []
    loss = []
    for pair in pairs:
        X = pair[0]
        Y = pair[1]
        X = torch.t(X)
        Y = torch.t(Y)
        F1 = torch.mean(X, 1)
        F2 = torch.mean(Y, 1)
        M = F2.reshape(-1, 1) * F1
        U, Sigma, V = torch.svd(M)
        aligned_t.append(torch.mm(torch.mm(pair[1], U), torch.t(V)))
    return aligned_t[0], aligned_t[1], aligned_t[2]


def updateCore(Core_s, R_s, C_s, U_s, V_s, Criteria_s):
    numerator = getPreRating(U_s.t(), V_s.t(), Criteria_s.t(), R_s)
    temp = getPreRating(U_s, V_s, Criteria_s, Core_s)
    denominator = getPreRating(U_s.t(), V_s.t(), Criteria_s.t(), temp)
    Htemp = torch.pow(
        torch.div(numerator, denominator), 1 / 3)
    Core_s = Core_s - 0.01 * (Core_s - Core_s.mul(Htemp))
    return Core_s.to(device)


def updateCore_t(Core_t, R_s, C_s, U_s, V_s, Criteria_s, Core_s):
    numerator = getPreRating(U_s.t(), V_s.t(), Criteria_s.t(), R_s)
    temp = getPreRating(U_s, V_s, Criteria_s, Core_t)
    denominator = getPreRating(U_s.t(), V_s.t(), Criteria_s.t(), temp)
    Htemp = torch.pow(
        torch.div(numerator, denominator), 1 / 3)
    Core_t = Core_t - 0.11 * (Core_t - Core_s)
    return Core_t.to(device)



def step(data, grad, state, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0, eps=eps):
    # State initialization
    if len(state) == 0:
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros(grad.size()).to(device)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros(grad.size()).to(device)

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

    state['step'] += 1
    if weight_decay != 0:
        grad = grad.add(weight_decay, data)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(1 - beta1, grad)
    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

    denom = torch.sqrt(exp_avg_sq) + eps

    bias_correction1 = 1 - beta1 ** state['step']
    bias_correction2 = 1 - beta2 ** state['step']
    step_size = lr * sqrt(bias_correction2) / bias_correction1

    return data.addcdiv(-step_size, exp_avg, denom), state


def UVCupdate(U_s, V_s, Criteria_s, U_s_state, V_s_state, Criteria_s_state, u_rate, v_rate, c_rate, R_s_u, C_s_u,
              device, M_s_u, M_s_v, Core_s):
    ratingError = getRatingError(Core_s, U_s, V_s, Criteria_s, R_s_u, C_s_u)

    TFuLoss = torch.from_numpy(np.ones((U_s.shape))).to(device).float()
    TFvLoss = torch.from_numpy(np.ones((V_s.shape))).to(device).float()

    Core_mix = mode_dot(Core_s, Criteria_s, 2)

    for enum in range(ratingError.shape[2]):
        H = Core_mix[:, :, enum]
        vc = (torch.matmul(V_s, H))
        uc = (torch.matmul(U_s, H))

        TFuLoss += torch.matmul(ratingError[:, :, enum], vc)
        TFvLoss += torch.matmul(ratingError[:, :, enum].t(), uc)

    du = -2 * TFuLoss + lambda_u * (U_s - M_s_u) + lambda_reg * U_s
    dv = -2 * TFvLoss + lambda_v * (V_s - M_s_v) + lambda_reg * V_s

    TFcLoss = torch.from_numpy(np.ones((Criteria_s.shape))).to(device).float()

    Hu = mode_dot(Core_s, U_s, 0)
    for enum in range(ratingError.shape[0]):
        H = Hu[enum, :, :]
        uv = (torch.matmul(V_s, H))
        TFcLoss += torch.matmul(ratingError[enum, :, :].t(), uv)

    dc = -2 * TFcLoss + lambda_reg * Criteria_s

    U_s, U_state = step(U_s, du, U_s_state, lr=u_rate)
    U_s[U_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)
    V_s, V_s_state = step(V_s, dv, V_s_state, lr=v_rate)
    V_s[V_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)
    Criteria_s, Criteria_s_state = step(Criteria_s, dc, Criteria_s_state, lr=c_rate)
    Criteria_s[Criteria_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)

    return (U_s, V_s, Criteria_s, Core_s)


def UVCupdate_t(U_s, V_s, Criteria_s, U_s_state, V_s_state, Criteria_s_state, u_rate, v_rate, c_rate, R_s_u, C_s_u,
                device, M_s_u, M_s_v, Core_s):
    ratingError = getRatingError(Core_s, U_s, V_s, Criteria_s, R_s_u, C_s_u)

    TFuLoss = torch.from_numpy(np.ones((U_s.shape))).to(device).float()
    TFvLoss = torch.from_numpy(np.ones((V_s.shape))).to(device).float()

    Core_mix = mode_dot(Core_s, Criteria_s, 2)

    for enum in range(ratingError.shape[2]):
        H = Core_mix[:, :, enum]
        vc = (torch.matmul(V_s, H))
        uc = (torch.matmul(U_s, H))

        TFuLoss += torch.matmul(ratingError[:, :, enum], vc)
        TFvLoss += torch.matmul(ratingError[:, :, enum].t(), uc)

    du = -2 * TFuLoss + lambda_u * (U_s - M_s_u) + lambda_reg * U_s
    dv = -2 * TFvLoss + lambda_v * (V_s - M_s_v) + lambda_reg * V_s

    TFcLoss = torch.from_numpy(np.ones((Criteria_s.shape))).to(device).float()

    Hu = mode_dot(Core_s, U_s, 0)
    for enum in range(ratingError.shape[0]):
        H = Hu[enum, :, :]

        uv = (torch.matmul(V_s, H))
        TFcLoss += torch.matmul(ratingError[enum, :, :].t(), uv)

    dc = -2 * TFcLoss + lambda_reg * Criteria_s

    U_s, U_state = step(U_s, du, U_s_state, lr=u_rate)
    U_s[U_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)
    V_s, V_s_state = step(V_s, dv, V_s_state, lr=v_rate)
    V_s[V_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)
    Criteria_s, Criteria_s_state = step(Criteria_s, dc, Criteria_s_state, lr=c_rate)
    Criteria_s[Criteria_s < 0] = 0
    Core_s = updateCore(Core_s, R_s_u, C_s_u, U_s, V_s, Criteria_s)
    return (U_s, V_s, Criteria_s, Core_s)


def mode_dot(Core_s, U_s, mode):
    new_shape = list(Core_s.shape)
    new_shape[mode] = U_s.shape[0]
    res = torch.matmul(U_s, unfold(Core_s, mode))
    tensor = fold(res, mode, new_shape)
    return tensor


def factorTran(Criteria_s):
    temp = torch.matmul(Criteria_s,
                        torch.from_numpy(np.ones((Criteria_s.shape[1], Criteria_s.shape[0]))).to(device).float())
    return temp


def getPreRating(U_s, V_s, Criteria_s, Core_s):
    modes = list(range(len(Core_s.shape)))

    factors = [U_s, V_s, Criteria_s]
    tensor = Core_s

    for mode in modes:
        tensor = mode_dot(tensor, factors[mode], mode)
    preRating = tensor
    preRating[preRating > 5] = 5
    preRating[preRating < 0] = 0
    return (preRating)


def getRatingError(Core_s, U_s, V_s, Criteria_s, R_s_u, C_s_u):
    preRating = getPreRating(U_s, V_s, Criteria_s, Core_s)
    ratingError = C_s_u.mul(R_s_u - preRating)
    return ratingError


def getRatingError_stat(Core_s, U_s, V_s, Criteria_s, R_s_u, C_s_u):
    preRating = getPreRating(U_s, V_s, Criteria_s, Core_s)
    ratingError = C_s_u.mul(R_s_u - preRating)
    stat = []
    stat.append(len((abs(ratingError) <= 0).nonzero()))
    stat.append(len((abs(ratingError) <= 1).nonzero()))
    stat.append(len((abs(ratingError) <= 2).nonzero()))
    stat.append(len((abs(ratingError) <= 3).nonzero()))
    stat.append(len((abs(ratingError) <= 4).nonzero()))
    stat.append(len((abs(ratingError) <= 5).nonzero()))
    return ratingError, stat

def Noise(data, noise_percent):
    '''
    This function randomly set 30% of the entry to 0. But since data is a sparse matrix, a lot of this entry is already 0, so I suppose it's not useful.
    :param data:
    :param noise_percent:
    :return:
    '''
    noise = torch.FloatTensor((np.random.random(data.size()) > noise_percent) * 1).cuda()
    return torch.mul(data, noise)


def Noise2(data, noise_percent):
    '''
    This function randomly add some bias to the rating.
    :param data:
    :param noise_percent:
    :return:
    '''
    noise = (np.random.uniform(0, 1, data.size()) > noise_percent) * 1.0 * np.random.uniform(-0.1, 0.1, data.size())
    noise = np.maximum(noise, 0)
    noise = np.minimum(noise, 5)
    noise = torch.FloatTensor(noise).cuda()
    return noise + data


def ReadInit(Criteria_t_Num, input_size_x_s, r, data_dir, percent, word, device, c_r):
    path_t = data_dir + str(percent) + 'latent' + str(r) + 'U_%s' % word + '.pkl'
    print(path_t)
    if os.path.exists(path_t):
        U_s = torch.load(path_t, map_location='cuda')
    else:
        Ur = np.random.rand(int(input_size_x_s[0]), r)

        U_s = torch.from_numpy(Ur).to(device).float()
    path_t = data_dir + str(percent) + 'latent' + str(r) + 'V_%s' % word + '.pkl'
    if os.path.exists(path_t):
        V_s = torch.load(path_t, map_location='cuda')
    else:
        Ir = np.random.rand(int(input_size_x_s[1]), r)
        V_s = torch.from_numpy(Ir).to(device).float()
    path_t = data_dir + str(percent) + 'latent' + str(r) + 'Criteria_%s' % word + '.pkl'
    if os.path.exists(path_t):
        Criteria_s = torch.load(path_t, map_location='cuda')
    else:
        Ar = np.random.rand(Criteria_t_Num, c_r)
        Criteria_s = torch.from_numpy(Ar).to(device).float()

    path_t = data_dir + str(percent) + 'latent' + str(r) + 'core_%s' % word + '.pkl'
    if os.path.exists(path_t):
        Core_s = torch.load(path_t, map_location='cuda')
    else:
        Ar = np.random.rand(r, r, c_r)
        Core_s = torch.from_numpy(Ar).to(device).float() / 100
    return (U_s, V_s, Criteria_s, Core_s)


def ReadSize(rating):
    u_num = int(rating[:, 0].max() + 1.5)
    v_num = int(rating[:, 1].max() + 1.5)
    input_size_x = [u_num, v_num]
    input_size_x_u = input_size_x[1]
    input_size_x_v = input_size_x[0]

    input_size_p_u = -1.0
    input_size_p_v = -1.0

    hidden_size_u = 120
    hidden_size_v = 120

    return input_size_x, input_size_x_u, input_size_x_v, u_num, v_num, input_size_p_u, input_size_p_v, hidden_size_u, hidden_size_v


def changeSide(sideItem, v_num):
    input_size_p_v = int(sideItem[:, 1].max() + 1)
    itemSide = np.zeros((int(input_size_p_v), int(v_num)))
    for i in range(sideItem.shape[0]):
        itemSide[int(sideItem[i, 1]), int(sideItem[i, 0])] = 1
    return (itemSide, input_size_p_v)


def ReadSideInformation(trainAddress, u_num, v_num, device):
    sideItem = np.loadtxt('%s/sideMatItem.txt' % trainAddress, delimiter=',')
    (itemSide, input_size_p_v) = changeSide(sideItem, v_num)
    sideUser = np.loadtxt('%s/sideMatUser.txt' % trainAddress, delimiter=',')
    (userSide, input_size_p_u) = changeSide(sideUser, u_num)
    P_s_u = torch.from_numpy(userSide).to(device).float().t()
    P_s_v = torch.from_numpy(itemSide).to(device).float().t()
    return (P_s_u, P_s_v, input_size_p_u, input_size_p_v)


def avoidSparseError(ratingTensor, num, device):
    rt = ratingTensor.to(torch.device("cpu"))
    ratingNp = rt.numpy()
    ratingNp = np.minimum(ratingNp, num)
    ratingTensor = torch.tensor(ratingNp).to(device)
    return ratingTensor


def getRatingTensor(rating, device):
    userList = rating[:, 0]
    itemList = rating[:, 1]
    criteriaList = rating[:, 2]

    tensorLoc = rating[:, :3].T
    tensorLoc = tensorLoc.astype(int)
    value = rating[:, 3]
    i = torch.LongTensor(tensorLoc).to(device)
    v = torch.FloatTensor(value).to(device)

    ratingTensor = torch.sparse.FloatTensor(i, v, torch.Size(
        [int(userList.max() + 1), int(itemList.max() + 1), int(criteriaList.max() + 1)])).to_dense().to(device)
    # GPU->CPU
    ratingTensor = avoidSparseError(ratingTensor, int(v.max()), device)

    all_one = torch.FloatTensor(np.ones(rating[:, 3].size)).to(device)
    indicator = torch.sparse.FloatTensor(i, all_one, torch.Size(
        [int(userList.max() + 1), int(itemList.max() + 1), int(criteriaList.max() + 1)])).to_dense().to(device)
    indicator = avoidSparseError(indicator, 1, device)
    return (ratingTensor, indicator)


import math
import heapq
import json
import random


def evaluate_model(data, _K, preRating):
    hits, ndcgs = [], []

    # Single thread
    for u in data.keys():
        (hr, ndcg) = eval_one_rating(data, u, _K, preRating)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def makeTestItem(datadir, testData, thre):
    if os.path.exists('%s/testItem%s.json' % (datadir, thre)):
        with open('%s/testItem%s.json' % (datadir, thre), 'r') as fp:
            data = json.load(fp)
        return (data)
    else:
        dic = {}

        y = testData[:, 2]
        e = testData[y == 0]
        targetItem = e[e[:, 3] == 5]
        itemList = set(testData[:, 1])
        # l = np.zeros(len(targetItem))
        l = []
        for num in range(len(targetItem[:, 0])):
            u = targetItem[num, 0]
            i = targetItem[num, 1]
            target = e[e[:, 0] == u]
            ee = list(set(target[:, 1]))
            # start crete itemList

            ee.remove(int(i))
            ee.insert(0, int(i))
            targetList = ee
            newitemList = list(itemList)
            for i in targetList:
                newitemList.remove(i)

            random.shuffle(newitemList)

            targetList = (targetList + newitemList[:100 - len(targetList)])

            dic[int(u)] = targetList
            # l[num] = (len(set(target[:,1])))
            l.append((len(set(target[:, 1]))))
            # break

        if os.path.exists(datadir):
            1
        else:
            os.makedirs(datadir)

        with open('%s/testItem%s.json' % (datadir, thre), 'w') as fp:
            json.dump(dic, fp)
        with open('%s/testItem%s.json' % (datadir, thre), 'r') as fp:
            data = json.load(fp)
        return (data)


def getpre(u, items, preRating):
    itemInt = []
    for i in items:
        itemInt.append(int(i))
    try:
        result = preRating[int(u), itemInt, 0]
    except:
        result = preRating[int(u), itemInt]
    return result


def eval_one_rating(data, u, _K, preRating):
    items = data[u]
    gtItem = items[0]
    # Get prediction scores
    map_item_score = {}
    predictions = getpre(u, items, preRating)

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHR_NDCGS(datadir, testData, thre, _K, preRating):
    (data) = makeTestItem(datadir, testData, thre)

    (hits, ndcgs) = evaluate_model(data, _K, preRating)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return (hr, ndcg)


def getBigger(maxHr, maxNdcg, hr, ndcg):
    if maxHr < hr:
        maxHr = float(hr)
    if maxNdcg < ndcg:
        maxNdcg = float(ndcg)
    return (maxHr, maxNdcg)


def makeLogList(listlogAd, thelist, HrK, NdcgK):
    file_handle = open(listlogAd, mode='w')

    for l in thelist:
        temp = ''
        for j in l:
            temp = temp + '%.4f' % j + '\t,'
        file_handle.write(temp)
        file_handle.write('\n')
        print(temp)

    file_handle.write('HrK is \n')
    for l in HrK:
        temp = ''
        for j in l:
            temp = temp + '%.4f' % j + '\t,'
        file_handle.write(temp)
        file_handle.write('\n')
        print(temp)

    file_handle.write('NdcgK is \n')
    for l in NdcgK:
        temp = ''
        for j in l:
            temp = temp + '%.4f' % j + '\t,'
        file_handle.write(temp)
        file_handle.write('\n')
        print(temp)
    file_handle.close()
    return 0


from sklearn.model_selection import train_test_split


def ReadData(trainAddress, percent, randomNum, device):
    rating = np.genfromtxt('%s/NumChangeRatingsScale.csv' % trainAddress, delimiter=',')
    trainData, testData = train_test_split(rating, test_size=percent, random_state=randomNum)
    u_num = int(rating[:, 0].max() + 1.5) - 1
    v_num = int(rating[:, 1].max() + 1.5) - 1
    trainData = np.concatenate((trainData, np.array([[u_num, v_num, 0, 0]])))
    testData = np.concatenate((testData, np.array([[u_num, v_num, 0, 0]])))

    (ratingTensor, Indicator) = getRatingTensor(trainData, device)
    (R_s, C_s) = (ratingTensor, Indicator)
    R_s_u = R_s
    C_s_u = C_s
    (ratingTensor, Indicator) = getRatingTensor(testData, device)
    # test
    (R_s_2, C_s_2) = (ratingTensor, Indicator)
    R_s_u2 = R_s_2
    C_s_u2 = C_s_2
    input_size_x_s, input_size_x_s_u, input_size_x_s_v, s_u_num, s_v_num, none, none, hidden_size_s_u, hidden_size_s_v = ReadSize(
        rating)
    (P_s_u, P_s_v, input_size_p_s_u, input_size_p_s_v) = ReadSideInformation(trainAddress, s_u_num, s_v_num, device)
    features_c = R_s_u.shape[2]
    return (
        R_s_u, C_s_u, R_s_u2, C_s_u2, P_s_u, P_s_v, input_size_p_s_u, input_size_p_s_v, input_size_x_s,
        input_size_x_s_u,
        input_size_x_s_v, s_u_num, s_v_num, hidden_size_s_u, hidden_size_s_v, testData, features_c)
