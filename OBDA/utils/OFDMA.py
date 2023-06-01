import numpy as np
import torch
from scipy import integrate
import scipy.stats as stats
import copy
import random
from scipy import special
def quan_average_gradients_transmission(args, w, idxs_users, W, h):
    w_avg = copy.deepcopy(w[0])
    index_to_size = []
    index_to_key = []
    cumsum = [0]
    for key in w_avg.keys():
        w_avg[key].zero_()
        index_to_size.append(w_avg[key].size())
        index_to_key.append(key)
        cumsum.append(cumsum[-1] + w_avg[key].numel())
    th_err_rate_set = []
    prac_err_rate_set = []
    for i in range(0, len(w)):
        err_rate = 0.5 - 0.5 * special.erf((((2 * W[idxs_users[i], 0] / ((args.sigma))) * np.real(
            ((np.conj((h[:, idxs_users[i]])).T) @ (h[:, idxs_users[i]])))) ** 0.5) / np.sqrt(2))
        th_err_rate_set.append(err_rate)
        temp_flat = []
        for k in w_avg.keys():
            temp_flat.append(w[i][k].view(-1))

        temp_flat = torch.cat(temp_flat, -1)
        temp_flat_quan, Quan, Dec_Quan, = OFDMA_bpsk(temp_flat, args, W[idxs_users[i]], h[:, idxs_users[i]])
        corr_num = np.sum(Quan == Dec_Quan)
        err_rate = (Quan.shape[0] - corr_num) / Quan.shape[0]
        prac_err_rate_set.append(err_rate)

        for j in range(len(cumsum)-1):
            begin = cumsum[j]
            end = cumsum[j+1]
            temp_flat_raw = temp_flat_quan[begin:end].view(*index_to_size[j])
            w[i][index_to_key[j]] = temp_flat_raw

    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]

        # w_avg[k] = torch.div(w_avg[k], len(w))
        mask_neg_all = w_avg[k] < 0.
        mask_pos_all = w_avg[k] > 0.
        quan_all = mask_neg_all.float() * (-1.0) + mask_pos_all.float() * 1.0
        w_avg[k].zero_().add_(quan_all)

    return w_avg, th_err_rate_set, prac_err_rate_set

def OFDMA_bpsk(g_temp, args, W, h):
    x_input = g_temp.cpu().numpy()
    Quan = (x_input > 0.).astype(float) + (x_input < 0.).astype(float)*(-1.)
    Vote_eq = [-1, 1]
    for j in range(Quan.shape[0]):
        if Quan[j] == 0:
            Quan[j] = np.array(random.sample(Vote_eq, 1))
    Dec_Quan = copy.deepcopy(Quan)
    for i in range(Quan.shape[0]):
        Dec_Quan[i] = transmission(args, Quan[i], W, h.reshape(args.N, 1))
    # Dec_Quan = transmission2(args, Quan, W, h.reshape(args.N, 1))
    y_temp = torch.from_numpy(Dec_Quan)
    y = y_temp.float()
    return y, Quan, Dec_Quan

def transmission2(libopt, signal,transmitpower, h):
    # given channel, and POWER, aggregate the d-dim signals
    N = libopt.N
    g = signal
    g = g.reshape(1, len(signal))
    # print('encode_signal', g)
    # noise
    # noise_power = libopt.sigma * transmitpower # noise_power = 1
    noise_power = libopt.sigma
    # print('libopt.sigma', libopt.sigma)
    # n = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / (2)**0.5 * noise_power ** 0.5
    n = (np.random.randn(N, len(signal)) + 1j * np.random.randn(N, len(signal))) / (2) ** 0.5 * noise_power ** 0.5  # N 行 d 列，本来是N*1，一个一个发，则是N*d
    # transmit signals
    x_signal = transmitpower ** 0.5 * g
    # received signals
    y = h * x_signal + n
    # linear estimator eq.(11)
    # y_decode = (np.real((np.conj(h).T @ y) / (np.conj(h).T @ h) + (np.conj(h).T @ n)/(np.conj(h).T @ h)) > 0.).astype(float) + (np.real((np.conj(h).T @ y) / (np.conj(h).T @ h)+ (np.conj(h).T @ n)/(np.conj(h).T @ h)) < 0.).astype(float) * (-1.)
    y_decode = (h.conj().T @ y) / (np.linalg.norm(h) ** 2) +(h.conj().T @ n ) / (np.linalg.norm(h) ** 2)
    y_decode = (y_decode > 0.).astype(float) + (y_decode < 0.).astype(float) * (-1.)
    y_decode = y_decode.reshape(len(signal))
    # x_sig = [-1.0, 1.0]
    # sig_wig = copy.deepcopy(x_sig)
    # for i in range(len(x_sig)):
    #     # sig_wig[i] = np.conj(y - h * x_sig[i]).T @ (y - h * x_sig[i])
    #     sig_wig[i] = np.linalg.norm(y - h * x_sig[i])
    # y_decode = x_sig[sig_wig.index(min(sig_wig))]
    # print('decode_signal', y_decode)
    return y_decode

def transmission(libopt, signal,transmitpower, h):
    # given channel, and POWER, aggregate the d-dim signals
    N = libopt.N
    g = signal
    # print('encode_signal', g)
    # noise
    # noise_power = libopt.sigma * transmitpower # noise_power = 1
    noise_power = libopt.sigma
    # print('libopt.sigma', libopt.sigma)
    # n = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / (2)**0.5 * noise_power ** 0.5
    n = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / (2) ** 0.5 * noise_power ** 0.5  # N 行 d 列，本来是N*1，一个一个发，则是N*d
    # transmit signals
    x_signal = transmitpower ** 0.5 * g
    # received signals
    y = h * x_signal + n
    # linear estimator eq.(11)
    # y_decode = (np.real((np.conj(h).T @ y) / (np.conj(h).T @ h)) > 0.).astype(float) + (np.real((np.conj(h).T @ y) / (np.conj(h).T @ h)) < 0.).astype(float) * (-1.)
    x_sig = [-1.0, 1.0]
    sig_wig = copy.deepcopy(x_sig)
    for i in range(len(x_sig)):
        # sig_wig[i] = np.conj(y - h * x_sig[i]).T @ (y - h * x_sig[i])
        sig_wig[i] = np.linalg.norm(y - h * x_sig[i])
    y_decode = x_sig[sig_wig.index(min(sig_wig))]
    # print('decode_signal', y_decode)
    return y_decode