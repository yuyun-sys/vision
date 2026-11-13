import cv2, numpy as np, json, glob

with open("calib_result.json") as f:
    r = json.load(f)
K = np.array(r["K"]); D = np.array(r["D"])

# 取前 5 张标定图进行对比
paths = sorted(glob.glob("calib_imgs/*.png"))[:5]
rows = []

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f"读取失败: {path}")
        continue

    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    und = cv2.undistort(img, K, D, None, newK)

    # 添加文字标注
    cv2.putText(img, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(und, "Undistorted", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 水平拼接：左原图，右去畸变
    row = np.hstack((img, und))
    rows.append(row)

if len(rows) == 0:
    print("没有可用图像")
else:
    # 垂直堆叠所有行
    compare = np.vstack(rows)
    cv2.imwrite("undistort_compare.png", compare)
    print(f"已保存 undistort_compare.png，共 {len(rows)} 组对比")
    
