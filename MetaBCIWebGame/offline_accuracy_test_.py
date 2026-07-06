# offline_accuracy_test_.py
"""
离线准确率测试脚本（模拟在线固定窗口预测）
用于验证 SSVEP 四分类在 2 秒窗口下的准确率，满足赛方 ≥70% 要求。
与在线推理预处理完全一致（仅去均值，无 std 归一化）。
"""

import os
import numpy as np
import joblib
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from config import WINDOW_LEN_SAMPLES, SAMPLE_RATE
from online_decode import DynamicStoppingDecoder

# ========== 配置 ==========
DATA_ROOT = r"D:\pyproject\MetaBCI\data_self"  # 数据根目录
MODEL_PATH = "self_ssvep_model.pkl"  # 模型路径
OCCIPITAL_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]  # 枕区8通道
WINDOW_SAMPLES = WINDOW_LEN_SAMPLES  # 500


def load_data(root):
    """
    加载所有试次，返回 X (n_trials, 8, 500), y (n_trials)
    跳过已知异常试次（hw_trial_0000.npy）
    """
    X_list, y_list = [], []
    for label in range(4):
        folder = os.path.join(root, str(label + 1))
        if not os.path.isdir(folder):
            continue
        files = glob.glob(os.path.join(folder, "*.npy"))
        for f in files:
            # 跳过已知异常试次（与离线演示一致）
            if "hw_trial_0000.npy" in f:
                print(f"[跳过] 已知异常试次: {f}")
                continue
            data = np.load(f)  # (14, 500)
            data = data[OCCIPITAL_INDICES, :]  # (8, 500)
            X_list.append(data)
            y_list.append(label)
    if not X_list:
        raise RuntimeError("未找到数据！")
    X = np.array(X_list)  # (n, 8, 500)
    y = np.array(y_list)
    print(f"加载数据: {X.shape}, 标签分布: {np.bincount(y)}")
    return X, y


def main():
    print("===== 离线准确率测试（模拟在线固定窗口预测）=====")

    # 1. 加载模型
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ 模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 加载数据
    X, y_true = load_data(DATA_ROOT)

    # 3. 预处理：仅去均值（与在线推理一致，不做 std 归一化）
    X = X - np.mean(X, axis=2, keepdims=True)
    print("预处理完成（仅去均值）")

    # 4. 预测
    decoder = DynamicStoppingDecoder(model=model)
    y_pred = []
    for i in range(X.shape[0]):
        window = X[i]  # (8, 500)
        decision, conf, all_conf = decoder.predict_window(window)
        if decision is not None:
            y_pred.append(decision)
        else:
            # 若置信度低于阈值，取最大概率作为预测（保障统计完整性）
            decision = np.argmax(all_conf)
            y_pred.append(decision)
    y_pred = np.array(y_pred)

    # 5. 评估
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['up', 'down', 'left', 'right'])

    print(f"\n总体准确率: {acc * 100:.2f}%")
    print("\n各类别准确率:")
    for label in range(4):
        mask = (y_true == label)
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"  {['up', 'down', 'left', 'right'][label]}: {class_acc * 100:.2f}%")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(report)

    if acc >= 0.70:
        print("\n✅ 准确率达标（≥70%），满足赛方要求。")
    else:
        print(f"\⚠️ 准确率 {acc * 100:.2f}% 低于70%，需要进一步优化。")

    # 保存结果到文件（便于提交）
    with open("offline_accuracy_result.txt", "w") as f:
        f.write(f"总体准确率: {acc * 100:.2f}%\n")
        f.write("\n混淆矩阵:\n")
        f.write(np.array2string(cm))
        f.write("\n\n分类报告:\n")
        f.write(report)


if __name__ == "__main__":
    main()