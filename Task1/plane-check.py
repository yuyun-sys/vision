import pyrealsense2 as rs
import numpy as np
import cv2
import math

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align = rs.align(rs.stream.color)
profile = pipeline.start(config)
pc = rs.pointcloud()

clicked_points = []
latest_vtx = None
latest_color_image = None

def plane_fitting_test(points_3d):
    """使用 SVD 分解拟合平面，并计算点到平面的距离残差"""
    points = np.array(points_3d)
    # 计算质心
    center = np.mean(points, axis=0)
    # 去中心化
    centered = points - center
    # SVD 分解
    u, s, vh = np.linalg.svd(centered)
    # 法向量即最小奇异值对应的向量
    normal = vh[2]
    # 计算每个点到平面的距离
    dists = np.abs(np.dot(centered, normal))
    return np.mean(dists) * 1000 # 返回毫米级残差

def mouse_callback(event, x, y, flags, param):
    global clicked_points, latest_vtx
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 4:
            clicked_points = []  # 点满4个点自动重置
        clicked_points.append((x, y))
        print(f"已点击像素坐标: ({x}, {y})")

# 这段代码逻辑改一下，如果你点满4个点，它就自动算平面
        if len(clicked_points) == 4:
            pts_3d = []
            valid = True
            for pt in clicked_points:
                idx = pt[1] * 640 + pt[0]
                p3d = latest_vtx[idx]
                if p3d[2] <= 0: 
                    valid = False
                pts_3d.append(p3d)
            
            if valid:
                rmse_mm = plane_fitting_test(pts_3d)
                print("\n" + "="*50)
                print(f"✅ 拟合平面完成！")
                print(f"🏆 点到平面的平均残差: {rmse_mm:.3f} 毫米")
                print("说明：如果这个数值小于 2 毫米，就证明点云中该平面极其平整！")
                print("="*50 + "\n")
            else:
                print("❌ 点击区域包含无效深度，请重新点击板子上的点。")

cv2.namedWindow("Plane Check")
cv2.setMouseCallback("Plane Check", mouse_callback)

print("请在标定板上依次点击 4 个角点。按 Q 退出。")
try:
    for _ in range(30): pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        latest_vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        latest_color_image = np.asanyarray(color_frame.get_data()).copy()
        for pt in clicked_points:
            cv2.circle(latest_color_image, pt, 5, (0, 255, 0), -1)
        
        cv2.imshow("Plane Check", latest_color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()