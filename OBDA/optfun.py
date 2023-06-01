# -*- coding: utf-8 -*-

# import argparse
# from scipy.optimize import minimize
import copy
import numpy as np
from scipy import special
np.set_printoptions(precision=6, threshold=1e3)
import warnings
import cvxpy as cvx
import sys
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt

def DC_F(libopt, rho, h_d, T_K_A, X, W, index_t, indexnum, sub_index_P):
    X_pre = X
    obj0 = 0
    err_list = []
    obj_list = []
    obj_list2 = []
    for i in range(60):
        obj_pre = 0
        index = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_pre) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))
            obj_pre = obj_pre - 1 + 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / ( libopt.sigma)) * obj_k)))
            index = index + 1
        obj = 0
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X = cvx.Variable((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial = cvx.Parameter((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        index = 0
        for kk in index_t:
            obj_k = (cvx.trace(T_K_A[:, :, index_t.index(kk)] @ X) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))
            obj = obj - 1 + 2 * ((1 / 6) * cvx.exp(cvx.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * cvx.exp(cvx.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * cvx.exp(cvx.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
            index = index + 1
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(libopt.L + 1)]
        # constraints += [(1 / 6) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + ((1 / 12) * cvx.exp(cvx.real((-2 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + (1 / 4) * cvx.exp(cvx.real((-1 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))))) <= 0.5 for kk in range(indexnum)]

        cost = obj + rho * cvx.real(cvx.trace((np.eye(libopt.L + 1) - X_partial) @ X))
        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve(solver=cvx.SCS)
        if prob.status == 'infeasible' or prob.value is None:
            break
        err = np.abs(prob.value - obj0)
        X_pre = copy.deepcopy(X.value)
        X_DC =copy.deepcopy(X.value)
        obj_DC = 0
        obj_DC2 = 0
        index = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_DC) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))
            obj_DC = obj_DC + 1 - 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
            obj_DC2 = obj_DC2 + 1 - 2 * (0.5 - 0.5 * special.erf(np.real(((2 * W[kk] / (libopt.sigma)) * obj_k))** 0.5/ np.sqrt(2)))
            index = index + 1
        obj_list.extend(obj_DC.tolist())
        obj_list2.extend(obj_DC2.tolist())
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        err_list.append(err)
        if err < libopt.epislon:
            break
    return X_DC, err_list, obj_list, obj_list2

def DC_F2(libopt, rho, G, h_d, X, W, A):
    X_pre = X
    obj0 = 0
    err_list = []
    obj_list = []
    obj_list2 = []
    for i in range(60):
        obj_pre = 0
        index = 0
        for kk in range(libopt.M):
            obj_k = 0
            for kz in range(libopt.U):
                T_row = np.hstack(((np.conj((G[:, :, kz, kk])).T) @ G[:, :, kz, kk],
                                   (np.conj((G[:, :, kz, kk])).T) @ (
                                       (h_d[:, kz, kk]).reshape(libopt.N, 1))))
                # T_row = np.hstack((((G[:, :, kk]).T.conjugate()) @ G[:, :, kk], ((G[:, :, kk]).T.conjugate()) @ ((h_d[:, kk]).reshape(libopt.N, 1))))
                T_column = np.hstack(((np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T) @ G[:, :, kz, kk],
                     np.array([0]).reshape(1, 1)))
                # T_column = np.hstack(((((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ G[:, :, kk], np.array([0]).reshape(1, 1)))
                T_K = np.vstack((T_row, T_column))
                # print('T_K',(T_K[:, :, kk]).shape)
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T * (h_d[:, kk]).reshape(libopt.N, 1)))
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + (((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ (h_d[:, kk]).reshape(libopt.N, 1)))
                obj_k = obj_k + A[kz, kk] * (np.trace(T_K @ X_pre) + np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T @ (
                    h_d[:, kz, kk]).reshape(
                    libopt.N, 1))
            # obj_k = (np.trace(T_K_A[:, :, kk] @ X_pre) + np.conj(((h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))
            obj_pre = obj_pre - 1 + 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / ( libopt.sigma)) * obj_k)))
            index = index + 1
        obj = 0
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X = cvx.Variable((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial = cvx.Parameter((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        index = 0
        for kk in range(libopt.M):
            obj_k = 0
            for kz in range(libopt.U):
                T_row = np.hstack(((np.conj((G[:, :, kz, kk])).T) @ G[:, :, kz, kk],
                                   (np.conj((G[:, :, kz, kk])).T) @ (
                                       (h_d[:, kz, kk]).reshape(libopt.N, 1))))
                # T_row = np.hstack((((G[:, :, kk]).T.conjugate()) @ G[:, :, kk], ((G[:, :, kk]).T.conjugate()) @ ((h_d[:, kk]).reshape(libopt.N, 1))))
                T_column = np.hstack(((np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T) @ G[:, :, kz, kk],
                                      np.array([0]).reshape(1, 1)))
                # T_column = np.hstack(((((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ G[:, :, kk], np.array([0]).reshape(1, 1)))
                T_K = np.vstack((T_row, T_column))
                # print('T_K',(T_K[:, :, kk]).shape)
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T * (h_d[:, kk]).reshape(libopt.N, 1)))
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + (((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ (h_d[:, kk]).reshape(libopt.N, 1)))
                obj_k = obj_k + A[kz, kk] * (cvx.trace(T_K @ X) + np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T @ (
                    h_d[:, kz, kk]).reshape(
                    libopt.N, 1))
                #obj_k = (cvx.trace(T_K_A[:, :, index_t.index(kk)] @ X) + np.conj(((h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))
            obj = obj - 1 + 2 * ((1 / 6) * cvx.exp(cvx.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * cvx.exp(cvx.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * cvx.exp(cvx.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
            index = index + 1
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(libopt.L + 1)]
        # constraints += [(1 / 6) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + ((1 / 12) * cvx.exp(cvx.real((-2 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + (1 / 4) * cvx.exp(cvx.real((-1 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))))) <= 0.5 for kk in range(indexnum)]

        cost = obj + rho * cvx.real(cvx.trace((np.eye(libopt.L + 1) - X_partial) @ X))
        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve(solver=cvx.SCS)
        if prob.status == 'infeasible' or prob.value is None:
            break
        err = np.abs(prob.value - obj0)
        X_pre = copy.deepcopy(X.value)
        X_DC =copy.deepcopy(X.value)
        obj_DC = 0
        obj_DC2 = 0
        index = 0
        for kk in range(libopt.M):
            obj_k = 0
            for kz in range(libopt.U):
                T_row = np.hstack(((np.conj((G[:, :, kz, kk])).T) @ G[:, :, kz, kk],
                                   (np.conj((G[:, :, kz, kk])).T) @ (
                                       (h_d[:, kz, kk]).reshape(libopt.N, 1))))
                # T_row = np.hstack((((G[:, :, kk]).T.conjugate()) @ G[:, :, kk], ((G[:, :, kk]).T.conjugate()) @ ((h_d[:, kk]).reshape(libopt.N, 1))))
                T_column = np.hstack(((np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T) @ G[:, :, kz, kk],
                                      np.array([0]).reshape(1, 1)))
                # T_column = np.hstack(((((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ G[:, :, kk], np.array([0]).reshape(1, 1)))
                T_K = np.vstack((T_row, T_column))
                # print('T_K',(T_K[:, :, kk]).shape)
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T * (h_d[:, kk]).reshape(libopt.N, 1)))
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + (((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ (h_d[:, kk]).reshape(libopt.N, 1)))
                obj_k = obj_k + A[kz, kk] * (np.trace(T_K @ X_DC) + np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T @ (
                    h_d[:, kz, kk]).reshape(
                    libopt.N, 1))
            # obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_DC) + np.conj(((h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index], kk]).reshape(libopt.N, 1))
            obj_DC = obj_DC + 1 - 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
            obj_DC2 = obj_DC2 + 1 - 2 * (0.5 - 0.5 * special.erf(np.real(((2 * W[kk] / (libopt.sigma)) * obj_k))** 0.5/ np.sqrt(2)))
            index = index + 1
        obj_list.extend(obj_DC.tolist())
        obj_list2.extend(obj_DC2.tolist())
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        err_list.append(err)
        if err < libopt.epislon:
            break
    return X_DC, err_list, obj_list, obj_list2

def DC_F_W(libopt, rho, G, h_d, A, X, W):
    X_pre = X
    obj0 = 0
    err_list = []
    obj_list = []
    obj_list2 = []
    for i in range(120):
        # obj_pre = 0
        # for kk in index_t:
            # obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_pre) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))
            # obj_pre = obj_pre - 1 + 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / ( libopt.sigma)) * obj_k)))
        obj = 0
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X = cvx.Variable((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial = cvx.Parameter((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))

        for kk in range(libopt.M):
            obj_k = 0
            for kz in range(libopt.U):
                T_row = np.hstack(((np.conj((G[:, :, kz, kk])).T) @ G[:, :, kz, kk],
                                   (np.conj((G[:, :, kz, kk])).T) @ ((h_d[:, kz, kk]).reshape(libopt.N, 1))))
                # T_row = np.hstack((((G[:, :, kk]).T.conjugate()) @ G[:, :, kk], ((G[:, :, kk]).T.conjugate()) @ ((h_d[:, kk]).reshape(libopt.N, 1))))
                T_column = np.hstack(
                    ((np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T) @ G[:, :, kz, kk], np.array([0]).reshape(1, 1)))
                # T_column = np.hstack(((((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ G[:, :, kk], np.array([0]).reshape(1, 1)))
                T_K = np.vstack((T_row, T_column))
                obj_k = obj_k + A[kz, kk] *(cvx.trace(T_K @ X) + np.conj(((h_d[:, kz, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kz, kk]).reshape(libopt.N, 1))
            obj = obj - 1 + 2 * ((1 / 6) * cvx.exp(cvx.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * cvx.exp(cvx.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * cvx.exp(cvx.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(libopt.L + 1)]
        # constraints += [(1 / 6) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + ((1 / 12) * cvx.exp(cvx.real((-2 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1)))) + (1 / 4) * cvx.exp(cvx.real((-1 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(libopt.N, 1))))) <= 0.5 for kk in range(indexnum)]

        cost = obj + rho * cvx.real(cvx.trace((np.eye(libopt.L + 1) - X_partial) @ X))
        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve(solver=cvx.SCS)
        if prob.status == 'infeasible' or prob.value is None:
            break
        err = np.abs(prob.value - obj0)
        X_pre = copy.deepcopy(X.value)
        X_DC =copy.deepcopy(X.value)
        # obj_DC = 0
        # obj_DC2 = 0
        # for kk in index_t:
        #     obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_DC) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(libopt.N, 1))
        #     obj_DC = obj_DC + 1 - 2 * ((1 / 6) * np.exp(np.real((-4 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 12) * np.exp(np.real((-2 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)))
        #     obj_DC2 = obj_DC2 + 1 - 2 * (0.5 - 0.5 * special.erf(np.real(((2 * W[kk] / (libopt.sigma)) * obj_k))** 0.5/ np.sqrt(2)))
        # obj_list.extend(obj_DC.tolist())
        # obj_list2.extend(obj_DC2.tolist())
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        err_list.append(err)
        if err < libopt.epislon:
            break
    return X_DC, err_list, obj_list, obj_list2

def QC_W(args, G, h_d, opt_theta, A):
    w = np.ones([args.M, 1]) # 初始化权重为全1
    lamda = 0.001 #更新权重的参数
    w_max = 1 # 最大近似次数
    t_max = 10 # 最大泰勒展开次数
    q_max = 60 # 最大拟凸优化次数
    for iter_w in range(w_max): # 0范数近似
        p = (args.P0) * np.ones([args.M, 1]) # 初始化泰勒展开功率
        objlist = []
        errorlist = []
        objlist1 = []
        obj1 = 0
        for iter_t in range(t_max): #泰勒展开
            v = np.zeros([args.M, 1])
            init_s = 0
            # for Kc in range(args.M):
            #     v[Kc, 0] = 2 * w[Kc, 0] * p[Kc, 0]
            #     init_s = init_s + w[Kc, 0] * (p[Kc, 0])**2 #常数值
            for Kc in range(args.M):
                v[Kc, 0] = w[Kc, 0]
                init_s = init_s + w[Kc, 0] * (p[Kc, 0]) #常数值
            l = 0 #拟凸优化下界
            u = 10 #拟凸优化上界
            obj0 = 0
            obj_list = []
            error_list = []
            opt_pre = np.ones([args.M, 1])
            for iter_q in range(q_max): #拟凸优化
                t = (u+l)/2
                print('u:{},l:{},t:{}'.format(u, l, t))
                # 开始优化
                W_opt = cvx.Variable((args.M, 1)) # 定义变量
                chan_vector = []
                obj_W = 0
                for Kc in range(args.M):
                    chanstate = 0
                    for KK in range(args.U):
                        # H_r = G[:, :, Kc] @ opt_theta + h_d[:, Kc].reshape(libopt.N, 1)
                        chanstate = chanstate + A[KK, Kc] * (np.conj(
                            (G[:, :, KK, Kc] @ opt_theta + (
                                (h_d[:, KK, Kc]).reshape(args.N, 1)))).T) @ (
                                            G[:, :, KK, Kc] @ opt_theta + (
                                        (h_d[:, KK, Kc]).reshape(args.N, 1)))
                    obj_kw = (W_opt[Kc, 0] / args.sigma) * chanstate
                    obj_W = obj_W + 1 - 2 * ((1 / 6) * cvx.exp(cvx.real((-4) * obj_kw)) + (1 / 12) * cvx.exp(cvx.real((-2) * obj_kw)) + (1 / 4) * cvx.exp(
                        cvx.real((-1) * obj_kw))) # 分母的求和
                # print('obj_W ', obj_W.shape)
                constraints_w = [cvx.sum(W_opt[:, 0]) == args.M * args.P0]
                constraints_w += [W_opt[i, 0] >= 0 for i in range(args.M)]
                constraints_w += [np.sqrt(init_s) + (0.5/np.sqrt(init_s)) * (v.reshape(1, args.M)) @ (W_opt - p) + (cvx.norm(W_opt - p))**2 - t * obj_W <= 0]
                # constraints_w += [
                #         0.5 - (1 / 6) * cvx.exp(cvx.real((-4) * ((W_opt[i, 0] / ((args.sigma))) * chan_vector[i]))) - (
                #                     1 / 12) * cvx.exp(cvx.real((-2) * ((W_opt[i, 0] / ((args.sigma))) * chan_vector[i]))) - (
                #                     1 / 4) * cvx.exp(cvx.real(((-1 * W_opt[i, 0]) / (args.sigma)) * chan_vector[i])) >= 0
                #         for i in range(args.M)]
                prob_W = cvx.Problem(cvx.Minimize(0), constraints_w)
                prob_W.solve(solver=cvx.SCS)
                print('-----------------prob_W.status------------------', prob_W.status)
                if prob_W.status == 'optimal':
                    u = t
                    opt_pre = W_opt.value
                    opt_p = W_opt.value
                    for i in range(args.M):
                        if opt_p[i, 0] <= 1e-3:
                            opt_p[i, 0] = 0
                    obj_Kp = 0
                    for Kc in range(args.M):
                        chanstate1 = 0
                        for KK in range(args.U):
                            # H_r = G[:, :, Kc] @ opt_theta + h_d[:, Kc].reshape(libopt.N, 1)
                            chanstate1 = chanstate1 + A[KK, Kc] * (np.conj(
                                (G[:, :, KK, Kc] @ opt_theta + (
                                    (h_d[:, KK, Kc]).reshape(args.N, 1)))).T) @ (
                                                G[:, :, KK, Kc] @ opt_theta + (
                                            (h_d[:, KK, Kc]).reshape(args.N, 1)))
                        obj_p = (opt_p[Kc, 0] / args.sigma) * chanstate1
                        obj_Kp = obj_Kp + 1 - 2 * ((1 / 6) * np.exp(np.real((-4) * obj_p)) + (1 / 12) * np.exp(
                            np.real((-2) * obj_p)) + (1 / 4) * np.exp(
                            np.real((-1) * obj_p)))  # 分母的求和
                    obj_plot = (np.sqrt(init_s) + (0.5 / np.sqrt(init_s)) * (v.reshape(1, args.M)) @ (opt_p - p) + (np.linalg.norm(opt_p - p))**2)/obj_Kp
                    # obj_plot = np.sqrt(indexnum) / obj_Kp
                    obj_list.extend(obj_plot.tolist())
                    error_list.extend((np.abs(obj_plot - obj0)).tolist())
                    obj0 = obj_plot
                else:
                    l = t
                if np.abs(u - l) <=1e-4:
                    break
            print('-------------opt_power-------------', opt_pre)
            # print('-------------obj-------------', obj_list)
            # plt.plot(range(1, len(obj_list) + 1), obj_list)
            # plt.show()
            # print('-------------error-------------', error_list)
            # plt.plot(range(1, len(error_list) + 1), error_list)
            # plt.show()
            obj_Kp = 0
            obj_Kn = 0
            chanstatelist = []
            for Kc in range(args.M):
                chanstate1 = 0
                for KK in range(args.U):
                    # H_r = G[:, :, Kc] @ opt_theta + h_d[:, Kc].reshape(libopt.N, 1)
                    chanstate1 = chanstate1 + A[KK, Kc] * (np.conj(
                        (G[:, :, KK, Kc] @ opt_theta + (
                            (h_d[:, KK, Kc]).reshape(args.N, 1)))).T) @ (
                                         G[:, :, KK, Kc] @ opt_theta + (
                                     (h_d[:, KK, Kc]).reshape(args.N, 1)))
                chanstatelist.append(np.abs(chanstate1))
                obj_p = (opt_pre[Kc, 0] / args.sigma) * chanstate1
                obj_Kp = obj_Kp + 1 - 2 * ((1 / 6) * np.exp(np.real((-4) * obj_p)) + (1 / 12) * np.exp(
                    np.real((-2) * obj_p)) + (1 / 4) * np.exp(
                    np.real((-1) * obj_p)))  # 分母的求和
                obj_Kn = obj_Kn + w[Kc, 0] * (opt_pre[Kc, 0])#**2
            indexnum = np.count_nonzero(opt_pre)
            print('indexnum', indexnum)
            print('obj_Kp', obj_Kp)
            obj_plot = np.sqrt(obj_Kn) / obj_Kp
            obj_plot1 = np.sqrt(indexnum) / obj_Kp
            objlist.extend(obj_plot.tolist())
            objlist1.extend(obj_plot1.tolist())
            errorlist.extend((np.abs(obj_plot - obj1)).tolist())
            obj1 = obj_plot
            p = opt_pre
            # for i in range(args.M):
            #     if p[i, 0] <= 1e-5:
            #         p[i, 0] = 0
        print('-------------chanstatelist-------------', chanstatelist)
        print('-------------opt_powerpppp-------------', p)
        print('-------------obj after TL-------------', objlist)
        plt.plot(range(1, len(objlist) + 1), objlist)
        plt.show()
        print('-------------error after TL-------------', errorlist)
        plt.plot(range(1, len(errorlist) + 1), errorlist)
        plt.show()
        print('-------------obj1 after TL-------------', objlist1)
        plt.plot(range(1, len(objlist1) + 1), objlist1)
        plt.show()
        for kk in range(args.M):
            w[kk, 0] = 0.5/np.sqrt((p[kk, 0])**2 + lamda**2)
        print('--------------------w----------------', w)
    return p

def DC2_F(libopt, rho, h_d, T_K_A, X, W, index_t, indexnum):
    V = np.random.randn(libopt.L + 1, 1) + 1j * np.random.randn(libopt.L + 1, 1);
    V = V / np.abs(V)
    X_pre = copy.deepcopy(np.outer(V, V.conj()))
    obj0 = 0
    err_list = []
    obj_list = []
    for i in range(70):
        obj_pre = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_pre) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj_pre = obj_pre - 1 + 2 * ((1 / 12) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        obj = 0
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]

        X = cvx.Variable((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial = cvx.Parameter((libopt.L + 1, libopt.L + 1), hermitian=True)
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        for kk in index_t:
            obj_k = (cvx.trace(T_K_A[:, :, index_t.index(kk)] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj = obj - 1 + 2 * ((1 / 12) * cvx.exp(cvx.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * cvx.exp(cvx.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(libopt.L + 1)]
        constraints += [((1 / 12) * cvx.exp(cvx.real((-1 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, index_t[kk]]).reshape(libopt.N, 1)))) + (1 / 4) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (3 * libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, index_t[kk]]).reshape(libopt.N, 1))))) <= 0.5 for kk in range(indexnum)]

        cost = obj + rho * cvx.real(cvx.trace((np.eye(libopt.L + 1) - X_partial) @ X))
        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve(solver=cvx.SCS)
        if prob.status == 'infeasible' or prob.value is None:
            break
        err = np.abs(prob.value - obj0)
        X_pre = copy.deepcopy(X.value)
        X_DC =copy.deepcopy(X.value)
        obj_DC = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_DC) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj_DC = obj_DC + 1 - 2 * ((1 / 12) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        obj_list.extend(obj_DC.tolist())
        eigenValues, eigenVectors = np.linalg.eigh(X_pre)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        X_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        err_list.append(err)
        if err < libopt.epislon:
            break
    return X_DC, err_list, obj_list

def SROCR_F(libopt, sigma, h_d, T_K_A, X, W, index_t, indexnum):
    X_0 = X
    omiga = 0.0
    eigenValues, eigenVectors = np.linalg.eigh(X_0)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    u = eigenVectors[:, 0]
    obj0 = 0
    err_list = []
    obj_list = []
    dif_list = []
    for i in range(40):
        obj_pre = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_0) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj_pre = obj_pre + 1 - 2 * ((1 / 12) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        obj_list.extend(obj_pre.tolist())
        obj = 0
        X = cvx.Variable((libopt.L + 1, libopt.L + 1), hermitian=True)
        for kk in index_t:
            obj_k = (cvx.trace(T_K_A[:, :, index_t.index(kk)] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj = obj + 1 - 2 * ((1 / 12) * cvx.exp(cvx.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * cvx.exp(cvx.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        constraints = [X >> 0]
        constraints += [cvx.real((np.conj(u).T @ X) @ u) >= cvx.real((omiga * cvx.trace(X)))]
        constraints += [X[i, i] == 1 for i in range(libopt.L + 1)]
        constraints += [((1 / 12) * cvx.exp(cvx.real((-1 * W[index_t[kk]] / (libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, index_t[kk]]).reshape(libopt.N, 1)))) + (1 / 4) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (3 * libopt.sigma)) * (cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, index_t[kk]]).reshape(libopt.N, 1))).T @ (h_d[:, index_t[kk]]).reshape(libopt.N, 1))))) <= 0.5 for kk in range(indexnum)]
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        prob.solve(solver=cvx.SCS)
        if prob.status == 'infeasible' or prob.value is None:
            X.value = X_0
            sigma = sigma / 2
        eigenValues, eigenVectors = np.linalg.eigh(X.value)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        u = eigenVectors[:, 0]
        lam = eigenValues[0]
        omiga_t = (lam/np.trace(X.value)) + sigma
        omiga = min(1, omiga_t)
        err = np.abs(prob.value - obj0)
        obj0 = prob.value
        X_0 = copy.deepcopy(X.value)
        X_SROCR =copy.deepcopy(X.value)
        obj_SROCR = 0
        for kk in index_t:
            obj_k = (np.trace(T_K_A[:, :, index_t.index(kk)] @ X_SROCR) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T @ (h_d[:, kk]).reshape(libopt.N, 1))
            obj_SROCR = obj_SROCR + 1 - 2 * ((1 / 12) * np.exp(np.real((-1 * W[kk] / (libopt.sigma)) * obj_k)) + (1 / 4) * np.exp(np.real((-4 * W[kk] / (3 * libopt.sigma)) * obj_k)))
        err_list.append(err)
        diff = obj_SROCR - obj_pre
        dif_list.extend(diff.tolist())
        # if omiga == 1: # err < libopt.epislon:
        #     break
    return X_SROCR, err_list, obj_list, dif_list
