"""
    Orthogonal Matching Pursuit Algorithms
    1. Orthogonal Matching Pursuit (OMP)
    2. Non-negative Orthogonal Matching Pursuit (NNOMP)
    3. Weighted Non-negative Orthogonal Matching Pursuit (WNNOMP)
    4. Second-order Orthogonal Matching Pursuit (SOMP)
    5. Second-order Weighted Orthogonal Matching Pursuit (SWOMP)
"""

import math
import numpy as np
from scipy.optimize import nnls

def omp(A, y, K):
    '''
    :param y: RSRP vector R_{M*1}
    :param A: dictionary matrix R_{M*N}
    :param K: number of iterations
    '''
    N = A.shape[1]
    residual = y                                   #初始化残差
    index = np.zeros(N,dtype=int)
    for i in range(N):                             #第i列被选中就是1，未选中就是-1
        index[i] = -1
    result = np.zeros((N,1))
    for j in range(K):                             #迭代次数
        product = np.fabs(np.dot(A.T,residual))
        pos = np.argmax(product)                   #最大投影系数对应的位置
        index[pos] = 1                             #对应的位置取1
        my = np.linalg.pinv(A[:,index>=0])         #最小二乘
        a = np.dot(my,y)
        residual = y-np.dot(A[:,index>=0],a)
    result[index>=0] = a
    return result

def nnomp(A, y, K):
    N = A.shape[1]
    S = []
    index = np.zeros(N, dtype=int)
    res = y
    j = 0
    result = np.zeros((N,1))
    while j < K:
        product = np.zeros((N, 1))
        product = np.dot(A.T, res)
        # for i in range(N):
        #     product[i] = np.dot(A[:, i].T, res) / (np.linalg.norm(A[:, i], 2))
        product[index > 0, 0] = float("-inf")
        pos = np.argmax(product)
        S.append(pos)
        index[pos] = 1
        nnls_out = nnls(A[:, index > 0], y[:, 0])
        a = np.zeros((len(nnls_out[0]),1))
        for t in range(0,len(nnls_out[0])):
            a[t, 0] = nnls_out[0][t]
        res = y - np.dot(A[:, index > 0], a)
        j += 1
        result[index>0] = a
    return result

def wnnomp(A, y, K, _lambda = 0.01):
    N = A.shape[1]
    residual = y
    residual_db = 10000
    y_star = np.zeros((y.shape[0], 1))
    pos_list = []
    index = np.zeros(N,dtype=int)
    for i in range(N):
        index[i] = -1
    result = np.zeros((N,1))
    A_norm = np.zeros((N,1))
    A_norm[:, 0] = np.linalg.norm(A, ord=None, axis=0, keepdims=False)
    prob = A_norm / np.linalg.norm(A_norm[:,0])
    A_normed = A / A_norm[:, 0]
    j = 0
    while j <= K or (j <= K * 2 and residual_db >= 0.2):
        product = np.dot(A_normed.T, residual)
        if np.max(product) <= 0:
            break
        product_prob = product / np.linalg.norm(product[:,0]) + _lambda * prob
        product_prob[index >= 0, 0] = 0
        pos = np.argmax(product_prob)
        pos_list.append(pos)
        pos_list = sorted(pos_list)
        index[pos] = 1
        nnls_out = nnls(A[:,index >= 0], y[:,0])
        a = np.zeros((len(nnls_out[0]),1))
        for t in range(0,len(nnls_out[0])):
            a[t, 0] = nnls_out[0][t]
        residual = y - np.dot(A[:, index >= 0], a)
        y_star = np.dot(A[:, index >= 0], a)
        residual_db = np.average(np.fabs(10 * np.log10(y_star) - 10 * np.log10(y)))
        result[index>=0] = a
        j += 1
    return  result

'''
    综合考虑一阶统计量和二阶统计量
'''
def somp(A, y, sigma_Y, K):
    _lambda = 9
    M = A.shape[0]
    N = A.shape[1]
    S = []
    index = np.zeros(N, dtype=int)
    z = sigma_Y.reshape(-1, 1)
    res1 = y
    res2 = z
    result = np.zeros((N,1))
    k = 0
    while k < K:
        product = np.zeros((N, 1))
        for i in range(N):
            product1 = (A[:, i].T @ res1) /(np.linalg.norm(A[:, i], 2) * np.linalg.norm(res1, 2))
            ai = np.zeros((M, 1))
            ai[:, 0] = A[:, i]
            Qii = (ai @ ai.T).reshape(-1, 1)[:,0]
            product2 = Qii.T @ res2 / (np.linalg.norm(Qii, 2) * np.linalg.norm(res2, 2))
            # for j in S:
            #     ai = np.zeros((M, 1))
            #     ai[:, 0] = A[:, i]
            #     aj = np.zeros((M, 1))
            #     aj[:, 0] = A[:, i]
            #     Qij = (ai @ aj.T).reshape(-1, 1)
            #     Qji = (aj @ ai.T).reshape(-1, 1)
            #     product2 += Qij.T @ res2 / (np.linalg.norm(Qij, 2) * np.linalg.norm(res2, 2))
            #     product2 += Qji.T @ res2 / (np.linalg.norm(Qji, 2) * np.linalg.norm(res2, 2))
            product[i, 0] = product1 + _lambda * product2
        product[index > 0, 0] = float("-inf")
        pos = np.argmax(product)
        S.append(pos)
        index[pos] = 1

        ''' res1 更新 '''
        nnls_out = nnls(A[:, index > 0], y[:, 0])
        a = np.zeros((len(nnls_out[0]),1))
        for t in range(0,len(nnls_out[0])):
            a[t, 0] = nnls_out[0][t]
        res1 = y - np.dot(A[:, index > 0], a)

        ''' res2 更新 '''
        k += 1
        index2 = np.kron(index, index)
        Q = np.zeros((M**2, len(S)* len(S)))
        for i in range(len(S)):
            for j in range(len(S)):
                ai = np.zeros((M, 1))
                ai[:, 0] = A[:, S[i]]
                aj = np.zeros((M, 1))
                aj[:, 0] = A[:, S[j]]
                Q[:, i * len(S) + j] = (ai @ aj.T).reshape(1,-1)
        t = np.linalg.lstsq(Q, z, rcond=None)[0]

        # k += 1
        # Q = np.zeros((M ** 2, len(S)))
        # for i in range(len(S)):
        #     ai = np.zeros((M, 1))
        #     ai[:, 0] = A[:, S[i]]
        #     Q[:, i] = (ai @ ai.T).reshape(1, -1)
        # t = np.linalg.lstsq(Q, z, rcond=None)[0]
        res2 = z - np.dot(Q, t)

        result[index>0] = a

    return result

def swomp(A, y, sigma_Y, K):
    _lambda = 9
    _gamma1 = 1e-5
    _gamma2 = 1e-10
    M = A.shape[0]
    N = A.shape[1]
    S = []
    index = np.zeros(N, dtype=int)
    z = sigma_Y.reshape(-1, 1)
    res1 = y
    res2 = z
    result = np.zeros((N, 1))
    k = 0
    while k < K:
        product = np.zeros((N, 1))
        for i in range(N):
            product1 = (A[:, i].T @ res1) / (np.linalg.norm(A[:, i], 2) * np.linalg.norm(res1, 2)) + _gamma1  * np.linalg.norm(A[:, i], 2)
            ai = np.zeros((M, 1))
            ai[:, 0] = A[:, i]
            Qii = (ai @ ai.T).reshape(-1, 1)[:, 0]
            product2 = Qii.T @ res2 / (np.linalg.norm(Qii, 2) * np.linalg.norm(res2, 2)) + _gamma2 * np.linalg.norm(Qii, 2)
            # print((A[:, i].T @ res1) / (np.linalg.norm(A[:, i], 2) * np.linalg.norm(res1, 2)))
            # print("product1", product1)
            # print(Qii.T @ res2 / (np.linalg.norm(Qii, 2) * np.linalg.norm(res2, 2)) )
            # print("product2", product2)
            # for j in S:
            #     ai = np.zeros((M, 1))
            #     ai[:, 0] = A[:, i]
            #     aj = np.zeros((M, 1))
            #     aj[:, 0] = A[:, i]
            #     Qij = (ai @ aj.T).reshape(-1, 1)
            #     Qji = (aj @ ai.T).reshape(-1, 1)
            #     product2 += Qij.T @ res2 / (np.linalg.norm(Qij, 2) * np.linalg.norm(res2, 2))
            #     product2 += Qji.T @ res2 / (np.linalg.norm(Qji, 2) * np.linalg.norm(res2, 2))
            product[i, 0] = product1 + _lambda * product2
        product[index > 0, 0] = float("-inf")
        pos = np.argmax(product)
        S.append(pos)
        index[pos] = 1

        ''' res1 更新 '''
        nnls_out = nnls(A[:, index > 0], y[:, 0])
        a = np.zeros((len(nnls_out[0]), 1))
        for t in range(0, len(nnls_out[0])):
            a[t, 0] = nnls_out[0][t]
        res1 = y - np.dot(A[:, index > 0], a)

        ''' res2 更新 '''
        k += 1
        index2 = np.kron(index, index)
        Q = np.zeros((M ** 2, len(S) * len(S)))
        for i in range(len(S)):
            for j in range(len(S)):
                ai = np.zeros((M, 1))
                ai[:, 0] = A[:, S[i]]
                aj = np.zeros((M, 1))
                aj[:, 0] = A[:, S[j]]
                Q[:, i * len(S) + j] = (ai @ aj.T).reshape(1, -1)
        t = np.linalg.lstsq(Q, z, rcond=None)[0]

        # k += 1
        # Q = np.zeros((M ** 2, len(S)))
        # for i in range(len(S)):
        #     ai = np.zeros((M, 1))
        #     ai[:, 0] = A[:, S[i]]
        #     Q[:, i] = (ai @ ai.T).reshape(1, -1)
        # t = np.linalg.lstsq(Q, z, rcond=None)[0]
        res2 = z - np.dot(Q, t)

        result[index>0] = a

    return result