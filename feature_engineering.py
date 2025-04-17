import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt
from signal_processing import pan_tompkins_detector  # 同目录导入


def extract_features(segment, fs, rr_interval):
    """提取12维特征（含基线、波形、频域、HRV特征）"""
    try:
        # 基线漂移校正
        low_pass_b, low_pass_a = butter(2, 0.5 / (0.5 * fs), btype='lowpass')
        baseline = filtfilt(low_pass_b, low_pass_a, segment)
        corrected = segment - baseline

        # 基线特征（F2, F3）
        F2 = np.max(baseline) - np.min(baseline)  # 基线波动范围
        F3 = np.var(baseline)  # 基线方差

        # QRS波个数（F4）
        qrs_positions = pan_tompkins_detector(corrected, fs)
        F4 = len(qrs_positions)

        # 波形特征（F8, F9）
        F8 = kurtosis(corrected)  # 峰度
        F9 = np.abs(skew(corrected))  # 偏度绝对值

        # 频域特征（F10, F11, F12）
        fft_vals = np.abs(np.fft.rfft(corrected))
        freq = np.fft.rfftfreq(len(corrected), 1 / fs)

        # 5-15Hz能量占比（F10）
        idx5_15 = np.where((freq >= 5) & (freq <= 15))[0]
        idx5_45 = np.where((freq >= 5) & (freq <= 45))[0]
        F10 = np.sum(fft_vals[idx5_15]) / np.sum(fft_vals[idx5_45]) if np.sum(fft_vals[idx5_45]) != 0 else 0

        # 1-45Hz能量占比（F11）
        idx1_45 = np.where((freq >= 1) & (freq <= 45))[0]
        idx0_45 = np.where((freq >= 0) & (freq <= 45))[0]
        F11 = np.sum(fft_vals[idx1_45]) / np.sum(fft_vals[idx0_45]) if np.sum(fft_vals[idx0_45]) != 0 else 0

        # 0.5-2Hz能量占比（F12，呼吸相关）
        idx0_5_2 = np.where((freq >= 0.5) & (freq <= 2))[0]
        F12 = np.sum(fft_vals[idx0_5_2]) / np.sum(fft_vals[idx0_45]) if np.sum(fft_vals[idx0_45]) != 0 else 0

        # 形态学特征
        qrs_width = int(0.12 * fs)
        r_peak_pos = np.argmax(segment)
        qrs_segment = segment[r_peak_pos - qrs_width // 2: r_peak_pos + qrs_width // 2]
        qrs_amp = np.max(qrs_segment) - np.min(qrs_segment)  # QRS波幅值
        qrs_width_ratio = len(qrs_segment) / qrs_width  # QRS波宽度比

        # 心率变异性（HRV）特征
        rr_intervals = np.diff(rr_interval)
        sdnn = np.std(rr_intervals) if len(rr_intervals) >= 2 else 0  # SDNN
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) >= 2 else 0  # RMSSD

        # 组合特征（按文档顺序排列）
        features = [F2, F3, F4, F8, F9, F10, F11, F12, qrs_amp, qrs_width_ratio, sdnn, rmssd]

        # 检查NaN值
        if any(np.isnan(features)):
            return None
        return features

    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None