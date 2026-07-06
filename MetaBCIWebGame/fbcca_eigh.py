import numpy as np
from scipy.linalg import eigh, qr, pinv
from scipy.signal import cheby1, cheb1ord, sosfilt
from config import STIM_FREQS, SAMPLE_RATE

def cca_corr_eigh(X, Y):
    """
    基于广义特征值分解的典型相关分析，返回第一典型相关系数
    X: (n_channels, n_samples)
    Y: (n_ref, n_samples)
    """
    # 去均值
    X = X - np.mean(X, axis=1, keepdims=True)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    # 正交投影 Y
    Q, R = qr(Y.T, mode='economic')
    P = Q @ Q.T
    # 构造矩阵 Z = X.T
    Z = X.T
    # 求解广义特征值问题: Z^T Z 和 Z^T P Z
    Cxx = Z.T @ Z
    Cxp = Z.T @ P @ Z
    # 添加小正则化避免奇异
    eps = 1e-8
    Cxx += eps * np.eye(Cxx.shape[0])
    Cxp += eps * np.eye(Cxp.shape[0])
    # 解广义特征值
    eigvals, eigvecs = eigh(Cxx, Cxp)
    # 取最大特征值对应特征向量
    idx = np.argmax(eigvals)
    u = eigvecs[:, idx]
    # 计算相关系数
    a = u.T @ Z.T
    b = pinv(R) @ Q.T @ X.T @ u.reshape(-1,1)  # 简化，实际可直接用公式
    a = a.flatten()
    b = (b.T @ Y).flatten()
    rho = np.corrcoef(a, b)[0,1]
    return rho

def generate_reference(freqs, n_samples, srate, n_harmonics=5):
    t = np.arange(n_samples) / srate
    Yf = []
    for freq in freqs:
        ref = []
        for h in range(1, n_harmonics+1):
            ref.append(np.sin(2*np.pi*h*freq*t))
            ref.append(np.cos(2*np.pi*h*freq*t))
        Yf.append(np.vstack(ref))
    return np.array(Yf)

def filterbank_cca(X, Yf_refs, filterbank, weights):
    # X: (n_trials, n_channels, n_samples)
    n_trials, n_classes = X.shape[0], Yf_refs.shape[0]
    rho_sub = np.zeros((len(filterbank), n_trials, n_classes))
    for sb, sos in enumerate(filterbank):
        Xf = sosfilt(sos, X, axis=-1)  # (n_trials, n_channels, n_samples)
        for t in range(n_trials):
            for c in range(n_classes):
                rho_sub[sb, t, c] = cca_corr_eigh(Xf[t], Yf_refs[c])
    # 加权平方和
    w = np.array(weights)[:, None, None]
    weighted = np.sum(w * (rho_sub**2), axis=0)
    pred = np.argmax(weighted, axis=1)
    return pred, weighted

def get_default_filterbank(srate):
    """
    返回针对 SSVEP 的默认滤波器组和权重（Chen 2015 论文推荐）
    """
    passbands = [[6, 90], [14, 90], [22, 90], [30, 90], [38, 90]]
    stopbands = [[4, 100], [10, 100], [16, 100], [24, 100], [32, 100]]
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
        sos = cheby1(N, 0.5, wn, btype='bandpass', output='sos', fs=srate)
        filterbank.append(sos)
    weights = [(i+1)**(-1.25) + 0.25 for i in range(len(passbands))]
    return filterbank, weights