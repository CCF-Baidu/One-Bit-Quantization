#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarRes, CNNCifarRes18, CNNMnist2
from utils.averaging import quan_average_gradients_Imperfect_CSI, quan_average_gradients_AWGN, quan_average_gradients_Fading
import math
import cvxpy as cvx

from optfun import DC_F, DC2_F
from utils.OFDMA import quan_average_gradients_transmission
from models.test import test_img
import random
import time
import logging
from scipy.io import savemat
# 创建一个logger
logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件

#fh = logging.FileHandler('logger_{:.4f}.log'.format(time.time()))
fh = logging.FileHandler('logger_Fed_Performance_BDA_{:.4f}.log'.format(time.time()))

fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

#logger.info('fed_learn_mnist_cnn_100_iid_v2')
logger.info('fed_learn_cifar_cnn')


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    np.random.seed(args.seed + 2)  # 设置随机种子
    print('args.iid', args.iid)
    print('args.lr-scheduler', args.lr_scheduler)
    Noiseless = 0 # =1 run error-free benchmark;=0 run the proposed algorithm
    Proposed = 1
    # import pdb
    # pdb.set_trace()
    # print(torch.__version__)
    ####################################Define RIS ############################################
    args.alpha_direct = 3.76  # User-BS Path loss exponent,P L is the path loss exponent;
    # fc = 915 * 10 ** 6  # carrier frequency, wavelength lambda_c=3.0*10**8/fc，fc
    # fsc = np.ones([libopt.M], dtype=int)  # sub-channel carrier frequency
    # for i in range(libopt.M):
    #     fsc[i] = fc + i * libopt.subgap * 10 ** 6
    # fsc[i] = 915 * 10 ** 6
    # print(fsc)
    # BS_Gain = 10 ** (5.0 / 10)  # BS antenna gain,G_PS
    # RIS_Gain = 10 ** (5.0 / 10)  # RIS antenna gain,G_RIS
    # User_Gain = 10 ** (0.0 / 10)  # User antenna gain，G_D
    d_RIS = 1.0 / 10  # length of one RIS element/wavelength
    args.BS = np.array(
        [-50, 0, 10])  # location of the BS/PS，The PS is placed at (−50, 0, 10)，np.array创建一个[-50, 0, 10]数组。
    args.RIS = np.array([0, 0, 10])  # location of the RIS，RIS is placed at (0, 0, 10)

    x0 = np.ones([args.M], dtype=int)  # initial the device selection such that all devices are selected
    # sigma_n = np.power(10, -libopt.SNR / 10)  # noise power=P_0/SNR=0.1/SNR
    # sigma_n = 5.2 * 1e-3 #SNR = 5左右
    # sigma_n = 5.5 * 1e-8 # SNR = 5左右,mnist保底
    # sigma_n = 5 * 1e-10  # SNR = 5左右,mnist保底
    # np.power(10, -libopt.SNR / 10)  # noise power = P_0/SNR=0.1/SNR
    # to facilitate numerical optimization, we simultaneously scale up the channel coefficents and the noise
    # without loss of generality.
    # To this end, the noise variance is multipled by 1e10 and the channel coefficents are multipled by 1e5(their power scale 1e10)
    # By doing so, their values are guaranteed to be significant to allieviate error propogation in optimization.
    # ref = (1e-10) ** 0.5
    ref1 = (1e-10) ** 0.5
    sigma_n = np.power(10, -args.SNR / 10)
    print('sigma_n', sigma_n)
    print('SNR', 10 * np.log10(0.1/sigma_n))
    args.sigma = sigma_n / (ref1 ** 2)  # effective noise power after scaling
    print('sigma', args.sigma)
    # half devices have x\in[-20,0]
    args.dx1 = np.random.rand(int(
        np.round(args.M / 2))) * 20 - 20  # 返回一个或一组=int(np.round(libopt.M / 2))服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    if args.set == 1:
        # Setting 1:
        # the other half devices also have x\in[-20,0]
        args.dx2 = np.random.rand(int(args.M - np.round(args.M / 2))) * 20 - 20  # [100,100+range]
    else:
        # Setting 2:
        # the other half devices have x\in[100,120]
        args.dx2 = np.random.rand(int(args.M - np.round(args.M / 2))) * 20 + 100
    # concatenate all the x locations
    args.dx = np.concatenate((args.dx1, args.dx2))  # 数组，维度1*M
    # print('libopt.dx', libopt.dx)
    # y\in[-10,10]
    args.dy = np.random.rand(args.M) * 20 - 10
    # print('libopt.dy', libopt.dy)
    # distance between User to RIS，三维欧式距离
    args.d_UR = ((args.dx - args.RIS[0]) ** 2 + (args.dy - args.RIS[1]) ** 2 + args.RIS[2] ** 2
                   ) ** 0.5
    # print(libopt.d_UR)
    # distance between RIS to BS/PS
    args.d_RB = np.linalg.norm(args.BS - args.RIS)
    # print(libopt.d_RB)
    # distance of direct User-BS channel
    args.d_direct = ((args.dx - args.BS[0]) ** 2 + (args.dy - args.BS[1]) ** 2 + args.BS[2] ** 2
                       ) ** 0.5
    # print(libopt.d_direct)
    # print(libopt.d_direct)
    # Path loss of direct channel,every user 1 * M
    ###### XZ #####
    # K0 = 0.01
    # d0 = 1
    # alpha_UR = 3
    # alpha_RB = 3
    # alpha_SU = 3
    # G = np.zeros([libopt.N, libopt.L, libopt.M], dtype=complex)
    # h_d = np.zeros([libopt.N, libopt.M], dtype=complex)
    # H_RB = ((K0 * ((1 / libopt.d_RB) ** alpha_RB)) ** 0.5) * np.exp(1j * np.random.randn(libopt.N, libopt.L, libopt.M) * 2 * math.pi)  # The small-scale fading coeffificients follow the standard independent and identically distributed (i.i.d.) Gaussian distribution
    # h_UR = np.zeros([libopt.L, libopt.M], dtype=complex)
    # for j in range(libopt.M):
    #     h_d[:, j] = ((K0 * ((1/(libopt.d_direct[j])) ** alpha_SU) ) ** 0.5) * np.exp(1j * np.random.randn(libopt.N) * 2 * math.pi) #/ ref  # The small-scale fading coeffificients follow the standard independent and identically distributed (i.i.d.) Gaussian distribution
    #     h_UR[:, j] = ((K0 * ((1 / libopt.d_UR[j]) ** alpha_UR)) ** 0.5) * np.exp(1j * np.random.randn(libopt.L) * 2 * math.pi) #/ ref
    #     G[:, :, j] = H_RB[:, :, j] @ np.diag(h_UR[:, j])
    # print('hd', h_d)
    # print('hur', h_UR)
    # print('hrb', H_RB)
    # print('G', G[:, :, 1])
    # ###### WSR #####
    # The small-scale fading coeffificients
    K0 = 0.001
    d0 = 1

    alpha_UR = 2
    alpha_RB = 2
    alpha_direct = 5

    kappa_direct = 0  # direct channel Rician factor
    kappa_PS = 10 ** 0.3  # RIS-PS channel Rician factor
    kappa_User = 0  # RIS-device channel Rician factor

    H_RB_NLOS = (1 / (1 + kappa_PS)) ** 0.5 * (
            np.random.randn(args.N, args.L, args.U, args.M) + 1j * np.random.randn(args.N, args.L, args.U, args.M)) / 2 ** 0.5
    args.PL_RB = ((K0 * ((1 / args.d_RB) ** alpha_RB)) ** 0.5) # For every subcarrier, path loss is equal

    # RB H
    azi_out = np.pi / 2
    ste_out = np.exp(-1j * np.pi * np.arange(args.N) * np.sin(azi_out))
    # print('ste_out', ste_out.shape)
    ste_out = ste_out.reshape(args.N, 1)
    # azi_in = np.arctan((libopt.dx - libopt.RIS[0]) / (libopt.dy - libopt.RIS[1]))
    azi_in = np.arctan((args.BS[0] - args.RIS[0]) / (args.BS[1] - args.RIS[1]))
    # print('azi_in', azi_in.shape)
    ste_in = np.exp(-1j * np.pi * np.arange(args.L) * np.sin(azi_in))
    # print('ste_in', ste_in.shape)
    ste_in = ste_in.reshape(args.L, 1)
    H_RB_LOS = (kappa_PS / (1 + kappa_PS)) ** 0.5 * np.outer(ste_out, ste_in.conj())
    # print(H_RB_LOS.shape)
    H_RB = np.zeros([args.N, args.L, args.U, args.M], dtype=complex)
    for i in range(args.M):
        for j in range(args.U):
            H_RB[:, :, j, i] = args.PL_RB * (H_RB_NLOS[:, :, j, i] + H_RB_LOS) # For every subcarrier, large path loss is equal

    # UR Direct H
    G = np.zeros([args.N, args.L, args.U, args.M], dtype=complex)
    h_d = np.zeros([args.N, args.U, args.M], dtype=complex)
    h_UR = np.zeros([args.L, args.U, args.M], dtype=complex)
    for j in range(args.M):
        for i in range(args.U):
            h_d[:, i, j] = ((K0 * ((1 / (args.d_direct[j])) ** alpha_direct)) ** 0.5) * np.exp(1j * np.random.randn(
                args.N) * 2 * math.pi) / ref1  # The small-scale fading coeffificients follow the standard independent and identically distributed (i.i.d.) Gaussian distribution
            h_UR[:, i, j] = ((K0 * ((1 / args.d_UR[j]) ** alpha_UR)) ** 0.5) * np.exp(
                1j * np.random.randn(args.L) * 2 * math.pi) / ref1
            G[:, :, i, j] = H_RB[:, :, i, j] @ np.diag(h_UR[:, i, j])
    # 56print('h_d[:, i, j]', h_d[:, i, j])
    # print('h_d', h_d)
    #**************Communication  SNR**********
    # randn theta, random subcarrier allocation
    sub_index = random.sample(range(args.U), args.M)
    print('sub_index', sub_index)
    # # **************WTruncateRIS: Power and RIS matrix **********
    # opt_theta3 = np.exp(1j * np.random.randn(args.L, 1) * 2 * np.pi)
    # WTruncateRIS = args.M * args.P0 / np.repeat(args.M, args.M)
    # WTruncateRIS = WTruncateRIS.reshape(args.M, 1)
    # sum_h = 0
    # for i in range(args.M):
    #     sum_h = sum_h + ((np.linalg.norm(G[:, :, sub_index[i], i] @ opt_theta3 + h_d[:, sub_index[i], i])) ** 2)
    # gthRIS = sum_h / args.M
    # app_t0 = []
    # for i in range(args.M):
    #     if ((np.linalg.norm(G[:, :, sub_index[i], i] @ opt_theta3 + h_d[:, sub_index[i], i])) ** 2) <= gthRIS:
    #         WTruncateRIS[i, 0] = 0
    #         app_t0.append(i)
    # app_t1 = []
    # for i in range(args.M):
    #     if i not in app_t0:
    #         app_t1.append(i)
    # sum_1divh = 0
    # for i in app_t1:
    #     sum_1divh = sum_1divh + 1 / (
    #                 (np.linalg.norm(G[:, :, sub_index[i], i] @ opt_theta3 + h_d[:, sub_index[i], i])) ** 2)
    # for i in app_t1:
    #     WTruncateRIS[i, 0] = (args.M * args.P0 / sum_1divh) * 1 / (
    #             (np.linalg.norm(G[:, :, sub_index[i], i] @ opt_theta3 + h_d[:, sub_index[i], i])) ** 2)
    # # print('np.sum(WTruncateRIS)', np.sum(WTruncateRIS))
    # print('WTruncateRIS', WTruncateRIS)
    # SNR = 0
    # K_list = []
    # for i in range(WTruncateRIS.shape[0]):
    #     if WTruncateRIS[i, 0] != 0:
    #         K_list.append(i)
    # # hh = np.zeros([args.N, args.M], dtype=complex)
    # for i in range(args.M):
    #     hh = G[:, :, sub_index[i], i] @ opt_theta3 + ((h_d[:, sub_index[i], i]).reshape(args.N, 1)).reshape(
    #         args.N)
    #     SNR = SNR + 10 * np.log10((args.P0 * np.linalg.norm(hh) ** 2) / args.sigma)
    # print('Mean Receiver SNR', SNR / args.M)
    # file_name = 'channel.mat'
    # savemat(file_name, {'Hd': h_d, 'G': G, 'opt_theta': opt_theta3, 'noise': args.sigma, 'Pt': float(args.P0)})
    ####################################Communication para of every method : Power and RIS matrix ############################################
    args.K = np.ones(args.M, dtype=int) * int(30000.0 / args.M)

    if Proposed:
        # Opt W first
        iter_max = 1 # out
        iterw_max = 1  # in
        Whole_obj = 0
        objlist = []
        thoulist = []
        iter_LIST = []
        Pre_obj = 0
        increase = []
        # 初始化
        opt_theta = np.exp(1j * np.random.randn(args.L, 1) * 2 * np.pi)
        # W = args.M * 1 * args.P0 / np.repeat(args.M, args.M)
        # W = W.reshape(args.M, 1)
        A = np.vstack((np.zeros((args.U - args.M, args.M)), np.identity(args.M)))
        # print('A', A)

        for iter_opt in range(iter_max):
            # lamda = 0.00002 # * (iter_opt+1)
            # objslist = []
            # errslist = []
            # obj0 = 0
            # for iterw_opt in range(iter_max):
            #     indexnum = np.count_nonzero(W)
            #     print('indexnum', indexnum)
            #     A = cvx.Variable((args.U, args.M))
            #     obj_W = 0
            #     chan_vector = []
            #     obj_A = 0
            #     for Kc in range(args.M):
            #         for Kz in range(args.U):
            #             obj_A = obj_A + (A[Kz, Kc] - 2 * A[Kz, Kc] * A_P[Kz, Kc] + (A_P[Kz, Kc]) ** 2)
            #     for Kc in range(args.M):
            #         chanstate = 0
            #         for Kz in range(args.U):
            #             # H_r = G[:, :, Kc] @ opt_theta + h_d[:, Kc].reshape(libopt.N, 1)
            #             chanstate = chanstate + A[Kz, Kc] * (
            #                 np.conj((G[:, :, Kz, Kc] @ opt_theta + ((h_d[:, Kz, Kc]).reshape(args.N, 1)))).T) @ (
            #                                 G[:, :, Kz, Kc] @ opt_theta + ((h_d[:, Kz, Kc]).reshape(args.N, 1)))
            #             # chanstate1 = (H_r.T.conjugate()) @ H_r
            #             # chanstate2 = (np.linalg.norm(G[:, :, Kc] @ opt_theta + h_d[:, Kc].reshape(libopt.N, 1)))**2
            #             # print('chanstate', chanstate)
            #             # print('chanstate', 10 * np.log10(3*((chanstate)**2)/libopt.sigma))
            #         chan_vector.append(chanstate)
            #         obj_kw = (W[Kc, 0] / args.sigma) * chanstate  # Q函数里面一块
            #         obj_W = obj_W + 1 - 2 * (
            #                     (1 / 6) * cvx.exp(cvx.real((-4) * obj_kw)) + (1 / 12) * cvx.exp(cvx.real((-2) * obj_kw)) + (
            #                     1 / 4) * cvx.exp(cvx.real((-1) * obj_kw)))
            #     # print('obj_W ', obj_W.shape)
            #
            #     obj = obj_W - lamda * obj_A
            #     constraints_A = [cvx.sum(A[:, i]) <= 1 for i in range(args.M)]
            #     for i in range(args.M):
            #         constraints_A += [A[j, i] >= 0 for j in range(args.U)]
            #         constraints_A += [A[j, i] <= 1 for j in range(args.U)]
            #     constraints_A += [cvx.sum(A[i, :]) <= 1 for i in range(args.U)]
            #     # constraints_A += [
            #     #     0.5 - (1 / 6) * cvx.exp(cvx.real((-4) * ((W[index_t[i], 0] / ((args.sigma))) * chan_vector[i]))) - (
            #     #                 1 / 12) * cvx.exp(cvx.real((-2) * ((W[index_t[i], 0] / ((args.sigma))) * chan_vector[i]))) - (
            #     #                 1 / 4) * cvx.exp(cvx.real(((-1 * W[index_t[i], 0]) / (args.sigma)) * chan_vector[i])) >= 0
            #     #     for i in range(len(index_t))]
            #     prob_A = cvx.Problem(cvx.Maximize(obj), constraints_A)
            #     # try:
            #     prob_A.solve(solver=cvx.SCS)
            #     # except:
            #     #     prob_W.solve(solver=cvx.SCS)
            #     result_A = prob_A.value
            #     objslist.append(result_A )
            #     A = A.value
            #     print("A solution A is", A)
            #     A_P = A
            # print(objslist)
            # plt.plot(range(1, len(objslist) + 1), objslist)
            # plt.show()
            # for Kc in range(args.M):
            #     for Kz in range(args.U):
            #         if A[Kz, Kc] <= 0.5:
            #             A[Kz, Kc] = 0
            #         if A[Kz, Kc] >= 0.5:
            #             A[Kz, Kc] = 1
            # print("A solution A is", A)
            sub_index_P = []
            for i in range(args.M):
                for j in range(args.U):
                    if A[j, i] == 1:
                        A_index = j
                        sub_index_P.append(A_index)
            print('sub_index_P', sub_index_P)

            W_opt = cvx.Variable((args.M, 1))
            obj_W = 0
            chan_vector = []
            for Kc in range(args.M):
                chanstate = 0
                for Kz in range(args.U):
                    chanstate = chanstate + A[Kz, Kc] * (
                        np.conj((G[:, :, Kz, Kc] @ opt_theta + ((h_d[:, Kz, Kc]).reshape(args.N, 1)))).T) @ (
                                        G[:, :, Kz, Kc] @ opt_theta + ((h_d[:, Kz, Kc]).reshape(args.N, 1)))
                    chan_vector.append(chanstate)
                obj_kw = (W_opt[Kc, 0] / ((args.sigma))) * chanstate
                obj_W = obj_W + 1 - 2 * (
                        (1 / 6) * cvx.exp(cvx.real((-4) * obj_kw)) + (1 / 12) * cvx.exp(cvx.real((-2) * obj_kw)) + (
                        1 / 4) * cvx.exp(
                    cvx.real((-1) * obj_kw)))
            # print('obj_W ', obj_W.shape)
            constraints_w = [cvx.sum(W_opt[:, 0]) <= (args.P0 * args.M * 1)]
            constraints_w += [W_opt[i, 0] >= 0 for i in range(args.M)]
            # constraints_w += [
            #     0.5 - (1 / 6) * cvx.exp(cvx.real((-4) * ((W_opt[i, 0] / ((args.sigma))) * chan_vector[i]))) - (
            #                 1 / 12) * cvx.exp(cvx.real((-2) * ((W_opt[i, 0] / ((args.sigma))) * chan_vector[i]))) - (
            #                 1 / 4) * cvx.exp(cvx.real(((-1 * W_opt[i, 0]) / (args.sigma)) * chan_vector[i])) >= 0
            #     for i in range(args.M)]
            prob_W = cvx.Problem(cvx.Maximize(obj_W), constraints_w)
            prob_W.solve(solver=cvx.SCS)
            result_W = prob_W.value

            W = (W_opt.value).reshape(args.M, 1)
            W11 = copy.deepcopy(W)
            # print('chan_vector', chan_vector)
            # print("A solution W is", W)
            channelvvv = copy.deepcopy(chan_vector)
            print("A solution W is", W)

            for i in range(args.M):
                if W[i, 0] <= 1e-3:
                    W[i, 0] = 0
            print("A solution W is", W)
            # for Kc in range(args.M):
            #     for Kz in range(args.U):
            #         if A[Kz, Kc] <= 0.5:
            #             A[Kz, Kc] = 0
            #         if A[Kz, Kc] >= 0.5:
            #             A[Kz, Kc] = 1
            # print("A solution A is", A)
            # plt.plot(range(1, len(objslist) + 1), objslist)
            # plt.show()
            # plt.plot(range(1, len(errslist) + 1), errslist)
            # plt.show()
            # sub_index_P = np.zeros([args.M], dtype=int)
            # for i in range(args.M):
            #     for j in range(args.U):
            #         if A_P[j, i] == 1:
            #             A_index = j
            #             sub_index_P[i] = A_index
            # sub_index_P = list(sub_index_P)

            # for i in range(libopt.M):
            #     if W[i, 0] <= 1e-3:
            #         W[i, 0] = 0
            # print("A solution W is", W)
            # Using W optimize theta
            index_t = [i for i, e in enumerate(W) if e != 0]
            print('-------------index_t---------------', index_t)
            print('-------------sub_index_P---------------', sub_index_P)
            indexnum = np.count_nonzero(W)

            obj = 0
            X = cvx.Variable((args.L + 1, args.L + 1), hermitian=True)
            T_K_A = np.zeros([args.L + 1, args.L + 1, indexnum], dtype=complex)
            for kk in index_t:
                T_row = np.hstack(((np.conj((G[:, :, sub_index_P[kk], kk])).T) @ G[:, :, sub_index_P[kk], kk],
                                   (np.conj((G[:, :, sub_index_P[kk], kk])).T) @ ((h_d[:, sub_index_P[kk], kk]).reshape(args.N, 1))))
                # T_row = np.hstack((((G[:, :, kk]).T.conjugate()) @ G[:, :, kk], ((G[:, :, kk]).T.conjugate()) @ ((h_d[:, kk]).reshape(libopt.N, 1))))
                T_column = np.hstack(
                    ((np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(args.N, 1))).T) @ G[:, :, sub_index_P[kk], kk], np.array([0]).reshape(1, 1)))
                # T_column = np.hstack(((((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ G[:, :, kk], np.array([0]).reshape(1, 1)))
                T_K = np.vstack((T_row, T_column))
                T_K_A[:, :, index_t.index(kk)] = T_K
                # print('T_K',(T_K[:, :, kk]).shape)
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + np.conj(((h_d[:, kk]).reshape(libopt.N, 1))).T * (h_d[:, kk]).reshape(libopt.N, 1)))
                # obj_k = (W[kk] * 2 / ((libopt.sigma))) * cvx.real((cvx.trace(T_K[:, :, kk] @ X) + (((h_d[:, kk]).reshape(libopt.N, 1)).T.conjugate()) @ (h_d[:, kk]).reshape(libopt.N, 1)))
                obj_k = (cvx.trace(T_K @ X) + np.conj(((h_d[:, sub_index_P[kk], kk]).reshape(args.N, 1))).T @ (h_d[:, sub_index_P[kk], kk]).reshape(
                    args.N, 1))
                obj = obj + 1 - 2 * (
                            (1 / 6) * cvx.exp(cvx.real((-4 * W[kk] / (args.sigma)) * obj_k)) + (1 / 12) * cvx.exp(
                        cvx.real((-2 * W[kk] / (args.sigma)) * obj_k)) + (1 / 4) * cvx.exp(
                        cvx.real((-1 * W[kk] / (args.sigma)) * obj_k)))
            constraints = [X >> 0]
            constraints += [X[i, i] == 1 for i in range(args.L + 1)]
            # constraints += [((1 / 6) * cvx.exp(cvx.real((-4 * W[index_t[kk]] / (args.sigma)) * (
            #             cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1))).T @ (
            #     h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1)))) + (1 / 12) * cvx.exp(cvx.real(
            #     (-2 * W[index_t[kk]] / (args.sigma)) * (
            #                 cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1))).T @ (
            #         h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1)))) + (1 / 4) * cvx.exp(cvx.real(
            #     (-1 * W[index_t[kk]] / (args.sigma)) * (
            #                 cvx.trace(T_K_A[:, :, kk] @ X) + np.conj(((h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1))).T @ (
            #         h_d[:, sub_index_P[index_t[kk]], index_t[kk]]).reshape(args.N, 1))))) <= 0.5 for kk in range(indexnum)]
            prob = cvx.Problem(cvx.Maximize(obj), constraints)
            prob.solve(solver=cvx.SCS)
            result_X = prob.value
            iter_LIST.append(result_X)
            # Print result.
            # print("The optimal value is", prob.value)
            X_e = X.value

            # 打印特征值
            # print("A solution X is", X_e)
            # V, D = np.linalg.eig(X_e)
            # # print('打印特征值V：\n{}'.format(V.shape))
            # # print('打印特征值D：\n{}'.format(D.shape))
            # ind = np.argsort(V)
            # V = V[ind]
            # D = D[:, ind]
            # print('打印特征值a：\n{}'.format(V))

            # DC算法求最优rankone_X
            rho = 0.1 # * (iter_opt + 1)
            X_DC, err_list, obj_list_DC, obj_list2_DC = DC_F(args, rho, h_d, T_K_A, X_e, W, index_t, indexnum, sub_index_P)
            # print('--------------err_list-----------', err_list)
            # plt.plot(range(1, len(err_list) + 1), err_list)
            # plt.show()
            # print('--------------obj_list_DC-----------', obj_list_DC)
            # plt.plot(range(1, len(obj_list_DC) + 1), obj_list_DC)
            # plt.show()
            # print('--------------obj_list2_DC-----------', obj_list2_DC)
            # plt.plot(range(1, len(obj_list2_DC) + 1), obj_list2_DC)
            # plt.show()
            u, _, _ = np.linalg.svd(X_DC, compute_uv=True, hermitian=True)
            v_tilde = u[:, 0]
            opt_theta = v_tilde[0:args.L] / v_tilde[args.L]
            opt_theta = copy.deepcopy(opt_theta / np.abs(opt_theta))
            opt_theta1 = opt_theta.reshape(args.L, 1)
            # print('opt_theta_MD', opt_theta1)
            opt_theta = X_DC[0:args.L, args.L].reshape(args.L, 1)
            # print('abs(opt_theta)', np.abs(opt_theta))
            # print('opt_theta_1_N', opt_theta)
            # # # SROCR算法求最优rankone_X
            # sigma = 0.01
            # X_SROCR, err_list_SROCR, obj_list_SROCR, dif_list_SROCR  = SROCR_F(libopt, sigma, h_d, T_K_A, X_e, W, index_t, indexnum)
            # # plt.plot(range(1, len(err_list_SROCR) + 1), err_list_SROCR)
            # # plt.show()
            # # plt.plot(range(1, len(dif_list_SROCR) + 1), dif_list_SROCR)
            # # plt.show()
            # # plt.plot(range(1, len(obj_list_SROCR) + 1), obj_list_SROCR)
            # # plt.show()
            # opt_theta = X_SROCR[0:libopt.L, libopt.L].reshape(libopt.L, 1)
            # print('abs(opt_theta)', np.abs(opt_theta))

            # 计算目标函数，优化后
            index_w = [ii for ii, ee in enumerate(W) if ee != 0]
            indexnum_w = np.count_nonzero(W)
            # print(indexnum_w)
            Whole_sum = 0
            Whole_obj_klist = []
            # print(index_w)
            index = 0
            for i in index_w:
                chanstate = ((np.conj((G[:, :, sub_index_P[i], i] @ opt_theta + (h_d[:, sub_index_P[i], i]).reshape(args.N, 1))).T) @ (
                            G[:, :, sub_index_P[i], i] @ opt_theta + (h_d[:, sub_index_P[i], i]).reshape(args.N, 1)))
                # Whole_obj_k = (1 / 12) * np.exp((-1) * (W[i,0]/ ((libopt.sigma))) * np.real(((np.conj((G[:, :, Kc] @ opt_theta + (h_d[:, Kc]).reshape(libopt.N, 1))).T) @ (G[:, :, Kc] @ opt_theta + (h_d[:, Kc]).reshape(libopt.N, 1))))) + (1 / 4) * np.exp((-4 / 3)  * (W[i,0]/ ((libopt.sigma))) * np.real(((np.conj((G[:, :, Kc] @ opt_theta + (h_d[:, Kc]).reshape(libopt.N, 1))).T) @ (G[:, :, Kc] @ opt_theta + (h_d[:, Kc]).reshape(libopt.N, 1)))))
                Whole_obj_k = 1 - 2 * (0.5 - 0.5 * special.erf(((np.real(((2 * W[i, 0] / ((args.sigma))) * (
                            (np.conj((G[:, :, sub_index_P[i], i] @ opt_theta + (h_d[:, sub_index_P[i], i]).reshape(args.N, 1))).T) @ (
                                G[:, :, sub_index_P[i], i] @ opt_theta + (h_d[:, sub_index_P[i], i]).reshape(args.N, 1)))))) ** 0.5) / np.sqrt(
                    2)))  # Q(f) = 0.5 - 0.5 erf(f/sqrt(2))
                # Whole_obj_k = qfunc(((2 * W[i,0]/ ((libopt.sigma))) * np.real(((np.conj((G[:, :, i] @ opt_theta + (h_d[:, i]).reshape(libopt.N, 1))).T) @ (G[:, :, i] @ opt_theta + (h_d[:, i]).reshape(libopt.N, 1)))))**0.5)
                Whole_obj_klist.append(Whole_obj_k)
                Whole_sum = Whole_sum + Whole_obj_k
                index = index + 1
            Whole_obj = Whole_sum
            # if iter_opt != 0:
            #     if abs(Whole_obj - Whole_pre)/ abs(Whole_obj) <= libopt.threshold:
            #         break
            objlist.extend(Whole_obj.tolist())
        print('opt_W', W)
        print('opt_theta', opt_theta)
        print('sub_index_P', sub_index_P)

        theta1 = opt_theta
        # plt.plot(range(1, len(objlist) + 1), objlist)
        # plt.show()

    # **************With opt RIS: Power and RIS matrix **********
    ####################################fl para############################################
    # if args.set == 1:
    #     # Setting 1:
    #     # For M=40, K=750
    #     args.K = np.ones(args.M, dtype=int) * int(50000.0 / args.M)  # 一个device分配int(30000.0 / libopt.M)
    # else:
    #     # Setting 2:
    #     # 分配样本数，We randomly select half devices and draw the corresponding values of {Km} uniformly from [1000, 2000]. The values of {Km} for the other half devices areuniformly drawn from [100, 200].
    #     # Half (random selected) devices have Uniform[1000,2000] data, the other half have Uniform[100,200] data
    #     args.K = np.random.randint(1000, high=2001, size=(int(args.M)))
    #     lessuser_size = int(args.M / 2)
    #     args.K2 = np.random.randint(100, high=201, size=(lessuser_size))
    #     args.lessuser = np.random.choice(args.M, size=lessuser_size, replace=False)
    #     args.K[args.lessuser] = args.K2

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.M)
        else:
            dict_users = mnist_noniid(dataset_train, args.M)
    elif args.dataset == 'femnist':
        trans_femnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_train = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=True, transform=trans_femnist)
        # dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=False, transform=trans_femnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.M)
        else:
            dict_users = mnist_noniid(dataset_train, args.M)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.M)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device) # Without communication
        net_glob_L = CNNCifar(args=args).to(args.device) # With opt RIS

    elif args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = CNNMnist2(num_classes=10,num_channels=1,batch_norm=True).to(args.device)
        net_glob_L = CNNMnist2(num_classes=10,num_channels=1,batch_norm=True).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        net_glob_L = CNNMnist(args=args).to(args.device)

    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = CNNCifarRes18(args=args).to(args.device)
        net_glob_L = CNNMnist(args=args).to(args.device)

    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    net_glob_L.train()
    w_glob = net_glob.state_dict()
    w_glob_L = net_glob_L.state_dict()

    # training
    # cv_loss, cv_acc = [], []
    # val_loss_pre, counter = 0, 0
    # net_best = None
    # best_loss = None
    # val_acc_list, net_list = [], []

    # 记录日志
    logger.info(args)
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer_L = torch.optim.SGD(net_glob_L.parameters(), lr=args.lr, momentum=args.momentum)

    # optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[11, 70, 140, 141], gamma=0.35)
        scheduler_L = torch.optim.lr_scheduler.MultiStepLR(optimizer_L, milestones=[11, 70, 140, 141], gamma=0.35)
    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params:', net_total_params)

    if args.dataset == 'mnist' or args.dataset == 'femnist' :
        log_probs_dummy = net_glob(torch.ones(1, 1, 28, 28).to(args.device))
        log_probs_dummy_L = net_glob_L(torch.ones(1, 1, 28, 28).to(args.device))

    else:
        log_probs_dummy = net_glob(torch.ones(1, 3, 32, 32).to(args.device))
        log_probs_dummy_L = net_glob_L(torch.ones(1, 3, 32, 32).to(args.device))

    loss_dummy = F.cross_entropy(log_probs_dummy, torch.ones(1, ).long())
    loss_dummy_L = F.cross_entropy(log_probs_dummy_L, torch.ones(1, ).long())

    loss_dummy.backward()
    loss_dummy_L.backward()

    optimizer.zero_grad()
    optimizer_L.zero_grad()


    accuracy_test= []
    loss_train_1 = []
    loss_train_2 = []

    accuracy_test_L = []
    loss_train_L1 = []
    loss_train_L2 = []

    for iter in range(1, args.epochs+1):
        if Noiseless:
            print('Noiseless Case is running')
            # all devices are active
            w_locals, loss_locals = [], []
            buffer_locals = []
            idxs_users = range(args.M)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, buffer, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                buffer_locals.append(copy.deepcopy(buffer))
                # del w, buffer, loss
            # update global weights
            if args.mode == 'AWGN':
                w_glob = quan_average_gradients_AWGN(w_locals)
            # elif args.mode == 'P_CSI':
            #     w_glob = quan_average_gradients_Fading(w_locals, snr=args.snr, g_th=args.thd)
            # elif args.mode == 'NP_CSI':
            #     w_glob = quan_average_gradients_Imperfect_CSI(w_locals, snr=args.snr, delta=args.delta, g_th=args.thd)
            else:
                raise NotImplementedError

            for key, value in net_glob.named_parameters():
                value.grad.data = w_glob[key].data.detach()


            def average_buffer(w, layer):
                w_avg = copy.deepcopy(w[0][layer])
                for k in w_avg.keys():
                    for i in range(1, len(w)):
                        w_avg[k] += w[i][layer][k]
                    w_avg[k] = torch.true_divide(w_avg[k], len(w))
                return w_avg


            for (key, module) in net_glob.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    buffer_avg = average_buffer(buffer_locals, key)
                    module._buffers['running_mean'].data = buffer_avg['running_mean'].data
                    module._buffers['running_var'].data = buffer_avg['running_var'].data
                    module._buffers['num_batches_tracked'].data = buffer_avg['num_batches_tracked'].data

            optimizer.step()

            if args.lr_scheduler:
                scheduler.step()

            loss_avg = sum(loss_locals) / len(loss_locals)
            logger.info('Epoch: {}'.format(iter))
            logger.info('-----Train loss----- in Noiseless Case: {:.4f}'.format(loss_avg))

            del w_locals, loss_locals, buffer_locals
            # testing
            if iter % 1 == 0:
                acc_train, loss_train = test_img(net_glob, dataset_train, args)
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                logger.info('-----Train loss(testing) ----------- in Noiseless Case: {:.4f}'.format(loss_train))
                logger.info('------Test acc----- in Noiseless Case: {:.4f}'.format(acc_test))
                loss_train_1.append(loss_avg)
                loss_train_2.append(loss_train)
                accuracy_test.append(acc_test)

        if Proposed:
            print('Proposed is running')
            # partial devices are active
            idxs_users_L = np.asarray(range(args.M))
            K_list = []
            for i in range(W.shape[0]):
                if W[i, 0] != 0:
                    K_list.append(i)
            idxs_users_L = idxs_users_L[K_list]
            # the channel
            h1 = np.zeros([args.N, args.M], dtype=complex)
            SNR = 0
            index = 0
            for i in K_list:
                h1[:, i] = ((h_d[:, sub_index_P[i], i]).reshape(args.N, 1) + G[:, :, sub_index_P[i], i] @ theta1).reshape(args.N)
                SNR = SNR + 10 * np.log10((W[i, 0] * np.linalg.norm(h1[:, i]) ** 2) / args.sigma)
                index = index + 1
            print('Mean Receiver SNR', SNR / len(K_list))
            w_locals_L, loss_locals_L = [], []
            buffer_locals_L = []
            for idx in idxs_users_L:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, buffer, loss = local.train(net=copy.deepcopy(net_glob_L).to(args.device))
                w_locals_L.append(copy.deepcopy(w))
                loss_locals_L.append(copy.deepcopy(loss))
                buffer_locals_L.append(copy.deepcopy(buffer))
            w_glob_L, th_err_rate_set_L, prac_err_rate_set_L= quan_average_gradients_transmission(args, w_locals_L, idxs_users_L, W, h1)
            logger.info("error rate in theory with OPT RIS: {}".format(np.array(th_err_rate_set_L)))
            logger.info("error rate in practice with OPT RIS: {}".format(np.array(prac_err_rate_set_L)))
            for key, value in net_glob_L.named_parameters():
                value.grad.data = w_glob_L[key].data.detach()

            def average_buffer(w, layer):
                w_avg = copy.deepcopy(w[0][layer])
                for k in w_avg.keys():
                    for i in range(1, len(w)):
                        w_avg[k] += w[i][layer][k]
                    w_avg[k] = torch.true_divide(w_avg[k], len(w))
                return w_avg


            for (key, module) in net_glob_L.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    buffer_avg = average_buffer(buffer_locals_L, key)
                    module._buffers['running_mean'].data = buffer_avg['running_mean'].data
                    module._buffers['running_var'].data = buffer_avg['running_var'].data
                    module._buffers['num_batches_tracked'].data = buffer_avg['num_batches_tracked'].data

            optimizer_L.step()

            if args.lr_scheduler:
                scheduler_L.step()

            loss_avg_L = sum(loss_locals_L) / len(loss_locals_L)

            logger.info('Train loss running Proposed: {:.4f}'.format(loss_avg_L))

            del w_locals_L, loss_locals_L, buffer_locals_L
            # testing
            if iter % 1 == 0:
                acc_train_L, loss_train_L = test_img(net_glob_L, dataset_train, args)
                acc_test_L, loss_test_L = test_img(net_glob_L, dataset_test, args)
                logger.info(
                    '-----Train loss(testing) ----------- running Proposed: {:.4f}'.format(
                        loss_train_L))
                logger.info(
                    '------Test acc----- running Proposed: {:.4f}'.format(
                        acc_test_L))
                loss_train_L1.append(loss_avg_L)
                loss_train_L2.append(loss_train_L)
                accuracy_test_L.append(acc_test_L)

    if Noiseless:
        logger.info("average train loss over noiseless channel: {}".format(loss_train_1))
        logger.info("average train loss(Testing the training dataset) over noiseless channel: {}".format(loss_train_2))
        logger.info("average test acc over noiseless channel: {}%".format(accuracy_test))

    if Proposed:
        logger.info("average train loss with OPT RIS: {}".format(loss_train_L1))
        logger.info("average train loss(Testing the training dataset) with OPT  RIS: {}".format(loss_train_L2))
        logger.info("average test acc with OPT  RIS: {}%".format(accuracy_test_L))

