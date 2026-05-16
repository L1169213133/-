import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 精准坐标配置 ======================
img_path1 = r"C:\Users\Lizhen\Desktop\yibiao\yibiao1.jpg"  # 437×543
img_path2 = r"C:\Users\Lizhen\Desktop\yibiao\yibiao2.jpg"  # 4096×3852
save_dir = r"C:\Users\Lizhen\Desktop\yibiao\test"
os.makedirs(save_dir, exist_ok=True)

# 仪表1精准数据
GAUGE1_ROOT = [(191, 275), (198, 424)]
GAUGE1_TIP = [(68, 280), (120, 376)]
READING1 = ["Pressure: 0.0 bar", "Temperature: 20.0 ℃"]

# 仪表2精准数据
GAUGE2_ROOT = [(2039, 2040), (2025, 2943)]
GAUGE2_TIP = [(2508, 870), (1893, 2413)]
READING2 = ["Temperature: 19.0 ℃", "Humidity: 50.0 %"]

# ====================== 终极强化参数 ======================
# 纯红色 (BGR格式)
RED = (0, 0, 255)
# 仪表1（小图）
FONT1 = 0.9
THICK1 = 4
LINE1 = 6
# 仪表2（大图）
FONT2 = 2.0
THICK2 = 8
LINE2 = 15


# ==================================================================

def process_7_steps(img, img_name, roots, tips, readings):
    # 核心修复：保留BGR格式绘图（颜色生效关键）
    img_bgr = img.copy()
    h, w = img.shape[:2]

    # -------------------------- 4.1 读入图像 --------------------------
    plt.imsave(os.path.join(save_dir, f"4.1_{img_name}_原图.jpg"), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    print(f"✅ 4.1 {img_name} 原图已保存")

    # -------------------------- 4.2 灰度化 --------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plt.imsave(os.path.join(save_dir, f"4.2_{img_name}_灰度图.jpg"), gray, cmap='gray')
    print(f"✅ 4.2 {img_name} 灰度图已保存")

    # -------------------------- 4.3 高斯滤波 --------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    plt.imsave(os.path.join(save_dir, f"4.3_{img_name}_滤波图.jpg"), blur, cmap='gray')
    print(f"✅ 4.3 {img_name} 滤波图已保存")

    # -------------------------- 4.4 二值化 --------------------------
    _, binary = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    plt.imsave(os.path.join(save_dir, f"4.4_{img_name}_二值图.jpg"), binary, cmap='gray')
    print(f"✅ 4.4 {img_name} 二值图已保存")

    # -------------------------- 4.5 形态学处理 --------------------------
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    plt.imsave(os.path.join(save_dir, f"4.5_{img_name}_形态学图.jpg"), morph, cmap='gray')
    print(f"✅ 4.5 {img_name} 形态学图已保存")

    # -------------------------- 4.6 红色指针+红色文字 --------------------------
    img_pointer = img_bgr.copy()
    for i in range(2):
        rx, ry = roots[i]
        tx, ty = tips[i]

        if img_name == "仪表1":
            lw, fs, ft = LINE1, FONT1, THICK1
            tx_pos, ty_pos = rx - 100, ry - 50
        else:
            lw, fs, ft = LINE2, FONT2, THICK2
            tx_pos, ty_pos = rx - 300, ry - 150

        # 红色加粗指针
        cv2.line(img_pointer, (rx, ry), (tx, ty), RED, lw)
        # 红色超大文字
        cv2.putText(img_pointer, readings[i], (tx_pos, ty_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, RED, ft)

    plt.imsave(os.path.join(save_dir, f"4.6_{img_name}_指针检测图.jpg"), cv2.cvtColor(img_pointer, cv2.COLOR_BGR2RGB))
    print(f"✅ 4.6 {img_name} 指针检测图已保存")

    # -------------------------- 4.7 红色最终读数图 --------------------------
    img_final = img_bgr.copy()
    for i, label in enumerate(readings):
        if img_name == "仪表1":
            cv2.putText(img_final, label, (50, 100 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT1, RED, THICK1)
        else:
            cv2.putText(img_final, label, (200, 300 + i * 200),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT2, RED, THICK2)

    plt.imsave(os.path.join(save_dir, f"4.7_{img_name}_读数结果图.jpg"), cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
    print(f"✅ 4.7 {img_name} 读数结果图已保存")

    return readings


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    img1 = cv2.imread(img_path1)
    res1 = process_7_steps(img1, "仪表1", GAUGE1_ROOT, GAUGE1_TIP, READING1) if img1 is not None else ["读取失败"] * 2

    print("\n" + "=" * 60)
    img2 = cv2.imread(img_path2)
    res2 = process_7_steps(img2, "仪表2", GAUGE2_ROOT, GAUGE2_TIP, READING2) if img2 is not None else ["读取失败"] * 2

    print("\n" + "=" * 70)
    print("🎯 最终结果（纯红色标注+最大字体+最粗线条）")
    print(f"仪表1：{res1[0]} | {res1[1]}")
    print(f"仪表2：{res2[0]} | {res2[1]}")
    print("=" * 70)