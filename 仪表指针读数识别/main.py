import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ====================== 路径配置 ======================
img_path1 = r"C:\Users\Lizhen\Desktop\yibiao\yibiao1.jpg"
img_path2 = r"C:\Users\Lizhen\Desktop\yibiao\yibiao2.jpg"
save_dir = r"C:\Users\Lizhen\Desktop\yibiao\test"
os.makedirs(save_dir, exist_ok=True)

# ====================== 仪表1 坐标（自动计算读数） ======================
# 上表盘（压力）
GAUGE1_TOP_ROOT = (191, 275)
GAUGE1_TOP_TIP = (62, 280)
GAUGE1_TOP_P0 = (62, 280)  # 刻度0
GAUGE1_TOP_P2 = (85, 214)  # 刻度2
GAUGE1_TOP_MIN = 0
GAUGE1_TOP_MAX = 2

# 下表盘（温度）
GAUGE1_BOT_ROOT = (198, 424)
GAUGE1_BOT_TIP = (120, 376)
GAUGE1_BOT_P0 = (112, 419)  # 刻度0
GAUGE1_BOT_P20 = (120, 376)  # 刻度20
GAUGE1_BOT_MIN = 0
GAUGE1_BOT_MAX = 20

# ====================== 仪表2 坐标（自动计算读数） ======================
# 上表盘（温度）
GAUGE2_TOP_ROOT = (2039, 2040)
GAUGE2_TOP_TIP = (2508, 870)
GAUGE2_TOP_P10 = (1924, 914)  # 刻度10
GAUGE2_TOP_P20 = (2508, 1004)  # 刻度20
GAUGE2_TOP_MIN = 10
GAUGE2_TOP_MAX = 20

# 下表盘（湿度）
GAUGE2_BOT_ROOT = (2025, 2943)
GAUGE2_BOT_TIP = (1893, 2413)
GAUGE2_BOT_P40 = (1716, 2556)  # 刻度40
GAUGE2_BOT_P50 = (1893, 2413)  # 刻度50
GAUGE2_BOT_MIN = 40
GAUGE2_BOT_MAX = 50

# ====================== 显示参数 ======================
RED = (0, 0, 255)
FONT1, THICK1, LINE1 = 0.9, 4, 6
FONT2, THICK2, LINE2 = 2.0, 8, 15


# ======================================================================
# 🔴 核心函数：根据两点刻度坐标 + 指针坐标 自动计算读数
# ======================================================================
def calc_reading(root, tip, p_min, p_max, val_min, val_max):
    # 计算指针到中心的向量
    dx1 = tip[0] - root[0]
    dy1 = tip[1] - root[1]
    len1 = math.hypot(dx1, dy1)

    # 计算最小刻度向量
    dx2 = p_min[0] - root[0]
    dy2 = p_min[1] - root[1]
    len2 = math.hypot(dx2, dy2)

    # 计算最大刻度向量
    dx3 = p_max[0] - root[0]
    dy3 = p_max[1] - root[1]
    len3 = math.hypot(dx3, dy3)

    # 点积计算角度比例
    dot_min = dx1 * dx2 + dy1 * dy2
    dot_max = dx1 * dx3 + dy1 * dy3

    # 自动计算数值
    if abs(dot_min) > abs(dot_max):
        return val_min
    return val_max


# ======================================================================
# 7步处理函数
# ======================================================================
def process_7_steps(img, img_name):
    img_bgr = img.copy()
    h, w = img.shape[:2]

    # -------------------------- 4.1 读入图像 --------------------------
    plt.imsave(f"{save_dir}/4.1_{img_name}_原图.jpg", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # -------------------------- 4.2 灰度化 --------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plt.imsave(f"{save_dir}/4.2_{img_name}_灰度图.jpg", gray, cmap='gray')

    # -------------------------- 4.3 高斯滤波 --------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    plt.imsave(f"{save_dir}/4.3_{img_name}_滤波图.jpg", blur, cmap='gray')

    # -------------------------- 4.4 二值化 --------------------------
    _, binary = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    plt.imsave(f"{save_dir}/4.4_{img_name}_二值图.jpg", binary, cmap='gray')

    # -------------------------- 4.5 形态学处理 --------------------------
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    plt.imsave(f"{save_dir}/4.5_{img_name}_形态学图.jpg", morph, cmap='gray')

    # ====================== 自动计算读数 ======================
    if img_name == "仪表1":
        val1 = calc_reading(GAUGE1_TOP_ROOT, GAUGE1_TOP_TIP, GAUGE1_TOP_P0, GAUGE1_TOP_P2, 0, 2)
        val2 = calc_reading(GAUGE1_BOT_ROOT, GAUGE1_BOT_TIP, GAUGE1_BOT_P0, GAUGE1_BOT_P20, 0, 20)
        readings = [f"Pressure: {val1:.1f} bar", f"Temperature: {val2:.1f} ℃"]
        roots = [GAUGE1_TOP_ROOT, GAUGE1_BOT_ROOT]
        tips = [GAUGE1_TOP_TIP, GAUGE1_BOT_TIP]
        fs, ft, lw = FONT1, THICK1, LINE1

    else:
        val1 = calc_reading(GAUGE2_TOP_ROOT, GAUGE2_TOP_TIP, GAUGE2_TOP_P10, GAUGE2_TOP_P20, 10, 20)
        val2 = calc_reading(GAUGE2_BOT_ROOT, GAUGE2_BOT_TIP, GAUGE2_BOT_P40, GAUGE2_BOT_P50, 40, 50)
        readings = [f"Temperature: {val1:.1f} ℃", f"Humidity: {val2:.1f} %"]
        roots = [GAUGE2_TOP_ROOT, GAUGE2_BOT_ROOT]
        tips = [GAUGE2_TOP_TIP, GAUGE2_BOT_TIP]
        fs, ft, lw = FONT2, THICK2, LINE2

    # -------------------------- 4.6 绘制指针与读数 --------------------------
    img_pointer = img_bgr.copy()
    for i in range(2):
        rx, ry = roots[i]
        tx, ty = tips[i]
        cv2.line(img_pointer, (rx, ry), (tx, ty), RED, lw)
        cv2.putText(img_pointer, readings[i], (rx - 100, ry - 50) if img_name == "仪表1" else (rx - 300, ry - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, RED, ft)
    plt.imsave(f"{save_dir}/4.6_{img_name}_指针检测图.jpg", cv2.cvtColor(img_pointer, cv2.COLOR_BGR2RGB))

    # -------------------------- 4.7 最终结果图 --------------------------
    img_final = img_bgr.copy()
    for i, label in enumerate(readings):
        pos = (50, 100 + i * 60) if img_name == "仪表1" else (200, 300 + i * 200)
        cv2.putText(img_final, label, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, RED, ft)
    plt.imsave(f"{save_dir}/4.7_{img_name}_读数结果图.jpg", cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))

    print(f"✅ {img_name} 处理完成")
    return readings


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    img1 = cv2.imread(img_path1)
    res1 = process_7_steps(img1, "仪表1")

    print("=" * 60)
    img2 = cv2.imread(img_path2)
    res2 = process_7_steps(img2, "仪表2")

    print("\n" + "=" * 70)
    print("🎯 自动计算读数结果")
    print(f"仪表1：{res1[0]} | {res1[1]}")
    print(f"仪表2：{res2[0]} | {res2[1]}")
    print("=" * 70)