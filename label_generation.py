import numpy as np


def generate_label(segment, r_pos, fs=250, data_insufficient=False):
    """自适应SNR标签生成（1:Good, 2:Medium, 3:Poor）"""
    try:
        qrs_width = int(0.12 * fs)  # 120ms QRS波宽度
        noise_window = int(0.4 * fs)  # 400ms噪声窗口

        # 信号段（QRS波区域）
        qrs_start = max(0, r_pos - qrs_width // 2)
        qrs_end = min(len(segment), r_pos + qrs_width // 2)
        signal = segment[qrs_start:qrs_end]

        # 噪声段（QRS波外的安全区域）
        noise1_end = max(0, qrs_start - noise_window)
        noise2_start = min(len(segment), qrs_end + noise_window)
        noise = np.concatenate([segment[:noise1_end], segment[noise2_start:]])

        # 计算信号幅值和噪声水平
        signal_amp = np.percentile(signal, 95) - np.percentile(signal, 5)
        noise_level = np.percentile(np.abs(noise - np.median(noise)), 95)
        snr = signal_amp / (noise_level + 1e-6)  # 避免除零

        # 新增：形态学约束（低能量或过宽QRS视为Poor）
        qrs_energy = np.sum(np.abs(signal))
        qrs_width_ratio = len(signal) / qrs_width
        if qrs_energy < 0.5 or qrs_width_ratio > 1.5:
            return 3

        # 数据不足时强制多样性（20%概率调整标签）
        if data_insufficient and np.random.rand() < 0.2:
            return 2 if snr > 2 else 3

        # 正常标签分配
        return 1 if snr > 5 else 2 if snr > 2.5 else 3

    except Exception as e:
        return np.random.choice([1, 2, 3])  # 异常时随机标签