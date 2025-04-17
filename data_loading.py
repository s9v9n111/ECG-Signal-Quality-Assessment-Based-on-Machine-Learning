import numpy as np
from pathlib import Path
import pandas as pd
from signal_processing import load_ecg_data, pan_tompkins_detector
from label_generation import generate_label
from feature_engineering import extract_features


def check_label_match(label_file, data_dir):
    """检查标签文件与数据文件匹配性，返回匹配的{文件名:标签}"""
    try:
        label_df = pd.read_excel(label_file, header=None)
        if len(label_df.columns) < 2:
            print(f"⚠️ 标签文件{label_file.name}列数不足")
            return {}

        name_to_label = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))
        data_files = {f.stem for f in data_dir.glob("*.*")
                      if f.suffix.lower() in ('.csv', '.txt')}

        # 过滤无效匹配
        valid_names = name_to_label.keys() & data_files
        return {k: v for k, v in name_to_label.items() if k in valid_names}

    except Exception as e:
        print(f"标签文件加载失败: {str(e)}")
        return {}


def process_file(file_path, fs, window_size, data_insufficient, forced_label=None):
    """单文件处理流程，返回特征和标签列表"""
    features, labels = [], []
    try:
        ecg = load_ecg_data(file_path)
        print(f"Signal Stats: μ={np.mean(ecg):.2f} σ={np.std(ecg):.2f} pp={np.ptp(ecg):.2f}mV")

        peaks = pan_tompkins_detector(ecg, fs)
        print(f"Detected R-peaks: {len(peaks)} (e.g. {peaks[:3]})")

        window_center = window_size // 2
        for i in range(1, min(50, len(peaks))):  # 最多处理50个R峰
            peak = peaks[i]
            start = max(0, peak - window_center)
            end = start + window_size
            if end > len(ecg):
                end = len(ecg)
                start = end - window_size

            segment = ecg[start:end]
            if len(segment) != window_size:
                continue  # 跳过长度不足的片段

            rr_interval = peaks  # 传递所有R峰位置计算HRV
            feat = extract_features(segment, fs, rr_interval)
            if feat is None:
                continue  # 跳过特征提取失败的样本

            # 标签生成
            if forced_label is not None:
                label = forced_label
            else:
                label = generate_label(segment, window_center, fs, data_insufficient)

            features.append(feat)
            labels.append(label)

            if i < 4:  # 打印前3个样本信息
                print(f"Sample {i}: mean={feat[0]:.2f} label={label}")

    except Exception as e:
        print(f"文件处理失败: {str(e)}")
        return [], []

    return features, labels