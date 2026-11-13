import pyrealsense2 as rs
import numpy as np
import cv2
import math

# 1. 初始化相机流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 深度对齐到彩色（消除视差，确保点云和画面严格重合）
align = rs.align(rs.stream.color)
profile = pipeline.start(config)
pc = rs.pointcloud()

# 2. 全局变量存放鼠标点击的坐标和点云数据
clicked_points = []
latest_vtx = None
latest_color_image = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 2:
            clicked_points = []  # 点满两个后自动重置
        clicked_points.append((x, y))
        print(f"已点击像素坐标: ({x}, {y})")
        
        if len(clicked_points) == 2:
            p1_px = clicked_points[0]
            p2_px = clicked_points[1]
            
            # 从缓存的点云数据中提取 3D 坐标
            idx1 = p1_px[1] * 640 + p1_px[0]
            idx2 = p2_px[1] * 640 + p2_px[0]
            
            p1_3d = latest_vtx[idx1]
            p2_3d = latest_vtx[idx2]
            
            # 检查深度是否有效（Z>0）
            if p1_3d[2] <= 0 or p2_3d[2] <= 0:
                print("❌ 警告：您点击的位置深度值无效（Z=0）！请重新点击标定板上的有效区域。")
            else:
                # 计算三维欧氏距离
                dist_3d = math.sqrt((p1_3d[0]-p2_3d[0])**2 + 
                                    (p1_3d[1]-p2_3d[1])**2 + 
                                    (p1_3d[2]-p2_3d[2])**2)
                
                print("\n" + "="*40)
                print(f"点1的3D坐标: [{p1_3d[0]:.4f}, {p1_3d[1]:.4f}, {p1_3d[2]:.4f}]")
                print(f"点2的3D坐标: [{p2_3d[0]:.4f}, {p2_3d[1]:.4f}, {p2_3d[2]:.4f}]")
                print(f"✅ 两点三维距离: {dist_3d:.4f} 米 ({dist_3d*100:.2f} 厘米)")
                print(f"📏 Z轴高度差: {abs(p1_3d[2]-p2_3d[2])*100:.2f} 厘米")
                print("="*40 + "\n")

# 3. 创建窗口并绑定鼠标事件
cv2.namedWindow("Point Cloud Measure")
cv2.setMouseCallback("Point Cloud Measure", mouse_callback)

print("程序已启动。请在画面中点击标定板的两个角点。按 Q 键退出。")

try:
    # 预热相机
    for _ in range(30):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 生成点云
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        # 将点云转换为 numpy 数组以便访问
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        latest_vtx = vtx
        
        # 获取彩色图像
        color_image = np.asanyarray(color_frame.get_data())
        latest_color_image = color_image.copy()
        
        # 绘制点击点
        for pt in clicked_points:
            cv2.circle(latest_color_image, pt, 5, (0, 0, 255), -1)
            
        # 显示画面
        cv2.imshow("Point Cloud Measure", latest_color_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
