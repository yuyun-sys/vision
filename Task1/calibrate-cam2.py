import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob
import json
import cv2.aruco as aruco

# ================= 参数配置区域 =================
# 🔴 必须和你 Task 1 的标定板参数完全一致！
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
SQUARES_X = 14
SQUARES_Y = 10
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.015

CAM2_IMG_DIR = "calib_imgs_cam2"
RESULT_FILE = "calib_result2.json"
TARGET_COUNT = 20  # 🔴 自动采集目标张数（达到后自动停止）
# ===============================================

def capture_images():
    """第一步：全自动采集 Camera 2 的图片"""
    os.makedirs(CAM2_IMG_DIR, exist_ok=True)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
    detector = aruco.CharucoDetector(board)
    
    count = 0
    cooldown = 0 # 冷却帧计数，防止连续采集一模一样的图片
    
    print(f"\n📷 开始自动采集 Camera 2 图像。目标: {TARGET_COUNT} 张。")
    print("👉 请拿着标定板在镜头前变换不同距离和角度！程序自动保存VALID图像。")
    print("👉 如果提前按【Q】或【ESC】键，将直接进入分析阶段。")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            
            display = color_image.copy()
            found = False
            if charuco_ids is not None and len(charuco_ids) > 5:
                found = True
                pts = charuco_corners.reshape(-1, 2)
                for pt in pts:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            
            # 🔴 自动保存逻辑：检测到有效 + 冷却帧数足够
            if found and cooldown >= 10:
                path = f"{CAM2_IMG_DIR}/{count:03d}.png"
                cv2.imwrite(path, color_image)
                print(f"  ✅ 自动保存: {path} ({count+1}/{TARGET_COUNT})")
                count += 1
                cooldown = 0 # 重置冷却，防止一秒拍十张同一角度
                
                # 达到目标，自动结束采集
                if count >= TARGET_COUNT:
                    print(f"\n🎉 已收集够 {TARGET_COUNT} 张图像，自动结束采集！")
                    break
            elif not found:
                cooldown = 0 # 如果没找到，冷却重置
            else:
                cooldown += 1
            
            # 在画面上显示当前状态
            status_text = f"Saved: {count}/{TARGET_COUNT}"
            if found:
                cv2.putText(display, f"VALID | {status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"INVALID | {status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Camera 2 Auto Capture", display)
            
            # 允许提前退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 是 ESC 键
                print("\n⚠️ 用户手动提前结束采集，进入分析阶段...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"📸 采集结束，共收集 {count} 张有效图像。")

def calculate_calibration():
    """第二步：自动计算 K 和 D，并保存为 JSON"""
    print("\n🧠 开始分析图像并计算 Camera 2 的内参...")
    
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
    detector = aruco.CharucoDetector(board)
    
    all_corners, all_ids = [], []
    image_size = None
    
    image_paths = sorted(glob.glob(f"{CAM2_IMG_DIR}/*.png"))
    if len(image_paths) == 0:
        print("❌ 没有找到图像，分析终止。")
        return False

    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]
            
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) > 5:
            all_corners.append(charuco_corners.reshape(-1, 1, 2))
            all_ids.append(charuco_ids.reshape(-1, 1))
            print(f"  检测成功: {os.path.basename(path)} (角点数: {len(charuco_ids)})")
        else:
            print(f"  检测失败: {os.path.basename(path)}")

    if len(all_corners) < 10:
        print(f"⚠️ 有效图像仅 {len(all_corners)} 张，不足 10 张，无法标定！")
        return False

    all_obj_points, all_img_points = [], []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) > 0:
            all_obj_points.append(obj_pts)
            all_img_points.append(img_pts)

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None)

    print(f"\n✅ Camera 2 标定完成！重投影误差: {ret:.4f} 像素")
    
    result = {
        "device": "Camera 2 (现场标定)",
        "resolution": f"{image_size[0]}x{image_size[1]}",
        "board": {
            "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
            "square_length": SQUARE_LENGTH, "marker_length": MARKER_LENGTH,
            "aruco_dict": "DICT_6X6_1000"
        },
        "K": K.tolist(), "D": D.flatten().tolist(),
        "reprojection_error": ret,
        "num_images": len(all_corners)
    }
    
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"📁 已将结果保存为: {RESULT_FILE}")
    return True

if __name__ == "__main__":
    print("🎬 启动 Camera 2 全自动标定流水线...")
    capture_images()
    calculate_calibration()
    print("\n🎉 流程结束！后续双机相对位姿脚本请读取 calib_result2.json。")