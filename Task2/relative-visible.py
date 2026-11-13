import pyrealsense2 as rs
import cv2
import numpy as np
import json
import time
import cv2.aruco as aruco

# ================= 参数配置 =================
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
SQUARES_X, SQUARES_Y = 14, 10
SQUARE_LENGTH, MARKER_LENGTH = 0.02, 0.015

SMOOTH_WINDOW = 15
# ===========================================

board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
detector = aruco.CharucoDetector(board)

# ================= 1. 加载内参 =================
print("📂 正在加载两台相机的内参文件...")
with open("calib_result.json", "r") as f:
    r1 = json.load(f)
K1 = np.array(r1["K"], dtype=np.float64)
D1 = np.array(r1["D"], dtype=np.float64).reshape(1, -1)

with open("calib_result2.json", "r") as f:
    r2 = json.load(f)
K2 = np.array(r2["K"], dtype=np.float64)
D2 = np.array(r2["D"], dtype=np.float64).reshape(1, -1)
print("✅ 内参加载完成！")

# ================= 2. 自动获取序列号 =================
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    print(f"❌ 检测到 {len(devices)} 台相机，请插好两台相机！")
    exit()

serials = sorted([dev.get_info(rs.camera_info.serial_number) for dev in devices])
SERIAL_CAM1 = serials[0]
SERIAL_CAM2 = serials[1]
print(f"📷 Camera 1 序列号: {SERIAL_CAM1}")
print(f"📷 Camera 2 序列号: {SERIAL_CAM2}")

# ================= 3. 启动两台相机 =================
pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device(SERIAL_CAM1)
config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device(SERIAL_CAM2)
config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("\n🎬 正在启动两台相机...")
try:
    pipeline1.start(config1)
    time.sleep(1.5)
    pipeline2.start(config2)
    print("✅ 两台相机启动成功！")
except RuntimeError as e:
    print(f"❌ 相机启动失败: {e}")
    print("👉 请执行: sudo pkill -f python3 并拔插相机")
    exit()

# ================= 4. 工具函数 =================
def get_pose_from_frame(color_image, K, D):
    """从单帧图像中检测标定板并返回 T_board_to_cam"""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def rotation_matrix_to_euler(R):
    """旋转矩阵 -> 欧拉角（度），返回 (roll, pitch, yaw)"""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


def project_3d_to_pixel(point_3d, K, D, img_w=640, img_h=480, margin=1000):
    """把相机坐标系下的3D点投影到图像像素坐标，带安全防护"""
    if point_3d[2] <= 0.05:
        return None
    pts = np.array([point_3d], dtype=np.float64).reshape(-1, 1, 3)
    try:
        img_pts, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    except cv2.error:
        return None
    u = float(img_pts[0][0][0])
    v = float(img_pts[0][0][1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    if abs(u) > img_w + margin or abs(v) > img_h + margin:
        return None
    return int(round(u)), int(round(v))


def draw_optical_axis(disp, other_center, other_axis_dir, K, D,
                      center_color=(0, 0, 255), axis_color=(0, 255, 0)):
    """在当前相机画面中绘制另一台相机的光心和光轴"""
    h, w = disp.shape[:2]
    center_px = project_3d_to_pixel(other_center, K, D, w, h, margin=300)
    if center_px is not None:
        cv2.circle(disp, center_px, 8, center_color, -1)
        cv2.circle(disp, center_px, 8, (255, 255, 255), 2)
        cv2.putText(disp, "Optical Center", (center_px[0] + 12, center_px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    else:
        cv2.putText(disp, "Optical Center (out of view)", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    axis_dir = other_axis_dir / np.linalg.norm(other_axis_dir)
    s_values = np.linspace(0.1, 3.0, 60)
    candidates = []
    for s in s_values:
        p = other_center + s * axis_dir
        if p[2] > 0.5:
            candidates.append(p)

    if len(candidates) >= 2:
        p_near = candidates[0]
        p_far = candidates[-1]
        px_near = project_3d_to_pixel(p_near, K, D, w, h, margin=2000)
        px_far = project_3d_to_pixel(p_far, K, D, w, h, margin=2000)

        if px_near is not None and px_far is not None:
            def clamp(p):
                return (max(-w, min(2 * w, p[0])), max(-h, min(2 * h, p[1])))
            pn = clamp(px_near)
            pf = clamp(px_far)
            if pn != pf:
                cv2.line(disp, pn, pf, axis_color, 2, cv2.LINE_AA)
                if 0 <= pf[0] < w and 0 <= pf[1] < h:
                    cv2.putText(disp, "Optical Axis", (pf[0] + 8, pf[1] + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, axis_color, 1)


def draw_3d_scene(t_mean, R_mean, K1, K2, img_w=640, img_h=480,
                  panel_size=(800, 720), has_valid=True):
    """
    在面板上绘制3D场景：
    - Camera 1 和 Camera 2 用四棱锥表示
    - 两者连线代表平移向量 t
    - 每台相机有光轴延长线（虚线）和坐标轴
    """
    W, H = panel_size
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    panel[:] = (20, 22, 30)  # 深色背景

    # ============ 1. 虚拟观察相机设置 ============
    baseline = float(np.linalg.norm(t_mean)) if has_valid else 0.15
    scene_scale = max(baseline * 2.5, 0.35)

    # 场景中心：两台相机中点，稍微朝前（Z正方向）
    center = np.array([
        t_mean[0] / 2.0,
        t_mean[1] / 2.0,
        scene_scale * 0.35
    ])

    # 虚拟相机位置：从右后上方看过去
    virtual_cam_pos = center + scene_scale * np.array([0.8, -0.9, -1.1])

    # 看向场景中心
    target = center

    # "上" 方向（相机坐标系Y向下，所以"上"是 -Y）
    up_hint = np.array([0.0, -1.0, 0.0])

    forward = target - virtual_cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up_hint)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    up = np.cross(right, forward)

    # 焦距（像素）
    fov_deg = 60.0
    f_pixels = (W / 2.0) / np.tan(np.radians(fov_deg / 2.0))

    def project_3d(p):
        """把世界坐标系的3D点投影到面板像素坐标"""
        p_rel = p - virtual_cam_pos
        x_cam = float(np.dot(p_rel, right))
        y_cam = -float(np.dot(p_rel, up))
        z_cam = float(np.dot(p_rel, forward))
        if z_cam <= 0.05:
            return None
        u = W / 2.0 + f_pixels * x_cam / z_cam
        v = H / 2.0 + f_pixels * y_cam / z_cam
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    def draw_line_3d(p1, p2, color, thickness=2, dashed=False):
        if dashed:
            n = 24
            for i in range(0, n, 2):
                a = p1 + (p2 - p1) * (i / n)
                b = p1 + (p2 - p1) * ((i + 1) / n)
                pa = project_3d(a)
                pb = project_3d(b)
                if pa is not None and pb is not None:
                    cv2.line(panel, pa, pb, color, thickness, cv2.LINE_AA)
        else:
            pp1 = project_3d(p1)
            pp2 = project_3d(p2)
            if pp1 is not None and pp2 is not None:
                cv2.line(panel, pp1, pp2, color, thickness, cv2.LINE_AA)

    def draw_point_3d(p, color, radius=7):
        pp = project_3d(p)
        if pp is not None:
            cv2.circle(panel, pp, radius, color, -1)
            cv2.circle(panel, pp, radius, (255, 255, 255), 1)
            return pp
        return None

    def draw_text_3d(p, text, color, offset=(8, -8)):
        pp = project_3d(p)
        if pp is not None:
            tx = pp[0] + offset[0]
            ty = pp[1] + offset[1]
            cv2.putText(panel, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def make_pyramid(origin, R_cam, K, d=0.12):
        """生成相机四棱锥（顶点 + 底面四角）"""
        fx = K[0, 0]
        fy = K[1, 1]
        half_w = d * img_w / (2.0 * fx)
        half_h = d * img_h / (2.0 * fy)
        base_pts_cam = np.array([
            [-half_w, -half_h, d],
            [ half_w, -half_h, d],
            [ half_w,  half_h, d],
            [-half_w,  half_h, d]
        ])
        base_pts_world = (R_cam @ base_pts_cam.T).T + origin
        return origin, base_pts_world

    # ============ 2. 画地面网格 ============
    grid_range = scene_scale * 1.0
    for gx in np.linspace(-grid_range, grid_range, 7):
        draw_line_3d(np.array([gx, 0, -grid_range * 0.5]),
                     np.array([gx, 0,  grid_range * 1.5]),
                     (45, 50, 60), 1)
    for gz in np.linspace(-grid_range * 0.5, grid_range * 1.5, 7):
        draw_line_3d(np.array([-grid_range, 0, gz]),
                     np.array([ grid_range, 0, gz]),
                     (45, 50, 60), 1)

    # ============ 3. 相机四棱锥 ============
    apex1, base1 = make_pyramid(np.array([0.0, 0.0, 0.0]), np.eye(3), K1)
    apex2, base2 = make_pyramid(t_mean.copy(), R_mean.copy(), K2)

    # Camera 1：蓝色/青色
    for i in range(4):
        j = (i + 1) % 4
        draw_line_3d(base1[i], base1[j], (255, 200, 0), 2)
    for i in range(4):
        draw_line_3d(apex1, base1[i], (255, 200, 0), 2)

    # Camera 2：橙色
    for i in range(4):
        j = (i + 1) % 4
        draw_line_3d(base2[i], base2[j], (0, 130, 255), 2)
    for i in range(4):
        draw_line_3d(apex2, base2[i], (0, 130, 255), 2)

    # ============ 4. 基线 ============
    draw_line_3d(apex1, apex2, (0, 255, 255), 3)

    # ============ 5. 光轴延长线（虚线） ============
    axis_len = scene_scale * 0.9
    cam1_axis_end = apex1 + np.eye(3) @ np.array([0.0, 0.0, axis_len])
    cam2_axis_end = apex2 + R_mean @ np.array([0.0, 0.0, axis_len])
    draw_line_3d(apex1, cam1_axis_end, (0, 255, 0), 2, dashed=True)
    draw_line_3d(apex2, cam2_axis_end, (0, 255, 0), 2, dashed=True)

    # ============ 6. 世界坐标轴（在 Camera 1 处） ============
    coord_len = scene_scale * 0.35
    x_end = apex1 + np.array([coord_len, 0, 0])
    y_end = apex1 + np.array([0, coord_len, 0])
    z_end = apex1 + np.array([0, 0, coord_len])
    draw_line_3d(apex1, x_end, (0, 0, 255), 2)
    draw_line_3d(apex1, y_end, (0, 255, 0), 2)
    draw_line_3d(apex1, z_end, (255, 0, 0), 2)
    draw_text_3d(x_end, "X", (0, 0, 255), offset=(5, 5))
    draw_text_3d(y_end, "Y", (0, 255, 0), offset=(5, 5))
    draw_text_3d(z_end, "Z", (255, 0, 0), offset=(5, 5))

    # ============ 7. 光心和标签 ============
    draw_point_3d(apex1, (255, 200, 0), 8)
    draw_point_3d(apex2, (0, 130, 255), 8)
    draw_text_3d(apex1, "Cam1", (255, 200, 0), offset=(-50, 20))
    draw_text_3d(apex2, "Cam2", (0, 130, 255), offset=(15, 20))

    # 基线上标注距离
    mid = (apex1 + apex2) / 2.0
    draw_text_3d(mid, f"|t|={baseline*100:.1f}cm", (0, 255, 255), offset=(10, 20))

    # ============ 8. 面板文字信息 ============
    cv2.putText(panel, "3D View: Camera Relative Pose (Pyramid = Camera Frustum)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    if has_valid:
        cv2.putText(panel,
                    f"Cam2 in Cam1 frame:  t = [{t_mean[0]:+.3f}, {t_mean[1]:+.3f}, {t_mean[2]:+.3f}] m",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Baseline |t| = {baseline*100:.2f} cm",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(panel, "Waiting for valid detection in both cameras...",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ============ 9. 图例 ============
    ly = H - 110
    cv2.putText(panel, "Legend:", (20, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    ly += 24
    cv2.line(panel, (20, ly), (60, ly), (255, 200, 0), 3)
    cv2.putText(panel, "Camera 1 (origin, +Z forward)", (70, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ly += 22
    cv2.line(panel, (20, ly), (60, ly), (0, 130, 255), 3)
    cv2.putText(panel, "Camera 2 (position t, rotation R)", (70, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ly += 22
    cv2.line(panel, (20, ly), (60, ly), (0, 255, 255), 3)
    cv2.putText(panel, "Baseline (translation t)", (70, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ly += 22
    cv2.line(panel, (20, ly), (60, ly), (0, 255, 0), 2)
    cv2.putText(panel, "Optical Axis (dashed)", (70, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return panel


# ================= 5. 实时计算主循环 =================
t_history = []
rvec_history = []

print("\n🎯 实时相对位姿计算已启动！")
print("👉 请把标定板同时放到两台相机的视野中。按 Q 退出。")
print("👉 会同时弹出三个窗口：视频画面 + 数据面板 + 3D 相机姿态视图。")

while True:
    try:
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()

        color1 = frames1.get_color_frame()
        color2 = frames2.get_color_frame()
        if not color1 or not color2:
            continue

        img1 = np.asanyarray(color1.get_data())
        img2 = np.asanyarray(color2.get_data())

        T1 = get_pose_from_frame(img1, K1, D1)
        T2 = get_pose_from_frame(img2, K2, D2)

        disp1 = img1.copy()
        disp2 = img2.copy()

        has_valid = False
        t_mean = np.zeros(3)
        R_mean = np.eye(3)
        roll = pitch = yaw = 0.0
        distance = 0.0

        if T1 is not None and T2 is not None:
            has_valid = True
            T_cam2_to_cam1 = T1 @ np.linalg.inv(T2)
            t_current = T_cam2_to_cam1[:3, 3]
            R_current = T_cam2_to_cam1[:3, :3]

            rvec_current, _ = cv2.Rodrigues(R_current)
            rvec_current = rvec_current.flatten()

            t_history.append(t_current)
            rvec_history.append(rvec_current)
            if len(t_history) > SMOOTH_WINDOW:
                t_history.pop(0)
                rvec_history.pop(0)

            t_mean = np.mean(t_history, axis=0)
            rvec_mean = np.mean(rvec_history, axis=0)
            R_mean, _ = cv2.Rodrigues(rvec_mean)

            roll, pitch, yaw = rotation_matrix_to_euler(R_mean)
            distance = np.linalg.norm(t_mean)

            # 两台相机画面中互相画光心和光轴
            cam2_center_in_cam1 = t_mean
            cam2_axis_in_cam1 = R_mean @ np.array([0.0, 0.0, 1.0])
            draw_optical_axis(disp1, cam2_center_in_cam1, cam2_axis_in_cam1, K1, D1,
                              center_color=(0, 0, 255), axis_color=(0, 255, 0))

            R_inv = R_mean.T
            cam1_center_in_cam2 = -R_inv @ t_mean
            cam1_axis_in_cam2 = R_inv @ np.array([0.0, 0.0, 1.0])
            draw_optical_axis(disp2, cam1_center_in_cam2, cam1_axis_in_cam2, K2, D2,
                              center_color=(0, 0, 255), axis_color=(0, 255, 0))

            cv2.putText(disp1, "VALID - CAM1", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp2, "VALID - CAM2", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(disp1, f"Baseline: {distance*100:.2f} cm", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            print(f"\r📏 基线: {distance*100:.2f} cm | "
                  f"t = [{t_mean[0]:+.3f}, {t_mean[1]:+.3f}, {t_mean[2]:+.3f}] m | "
                  f"Yaw: {yaw:+.2f}° Pitch: {pitch:+.2f}° Roll: {roll:+.2f}°", end="")
        else:
            cv2.putText(disp1, "INVALID", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(disp2, "INVALID", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ===== 窗口 1：视频拼接画面 =====
        concat_disp = np.hstack((disp1, disp2))
        h, w = concat_disp.shape[:2]
        cv2.line(concat_disp, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        cv2.imshow("Dual Camera Realtime Pose", concat_disp)

        # ===== 窗口 2：数据面板 =====
        panel = build_info_panel(t_mean, R_mean, roll, pitch, yaw, distance, has_valid)
        cv2.imshow("Relative Pose Data Panel", panel)

        # ===== 窗口 3：3D 相机姿态视图 =====
        scene_3d = draw_3d_scene(t_mean, R_mean, K1, K2, has_valid=has_valid)
        cv2.imshow("3D Camera Poses", scene_3d)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n⚠️ 用户按 Q 退出。")
            break

    except cv2.error as e:
        print(f"\n⚠️ OpenCV 异常（已忽略继续）: {e}")
        continue
    except Exception as e:
        print(f"\n⚠️ 帧处理异常（已忽略继续）: {e}")
        continue

# ================= 6. 退出时保存最终结果 =================
print("\n\n✅ 实时相对位姿计算已结束。")

if len(t_history) > 0:
    t_final = np.mean(t_history, axis=0)
    rvec_final = np.mean(rvec_history, axis=0)
    R_final, _ = cv2.Rodrigues(rvec_final)
    roll, pitch, yaw = rotation_matrix_to_euler(R_final)
    distance_final = np.linalg.norm(t_final)

    print("=" * 50)
    print("📊 最终相对位姿结果：")
    print(f"平移向量 t (Camera 2 光心在 Camera 1 坐标系下的位置):")
    print(f"   t = [{t_final[0]:+.4f}, {t_final[1]:+.4f}, {t_final[2]:+.4f}] m")
    print(f"两相机基线距离: {distance_final*100:.2f} 厘米")
    print(f"Camera 2 相对于 Camera 1 的方位：")
    print(f"   Yaw   (绕 Z 轴): {yaw:+.2f}°")
    print(f"   Pitch (绕 Y 轴): {pitch:+.2f}°")
    print(f"   Roll  (绕 X 轴): {roll:+.2f}°")
    print("=" * 50)

    result = {
        "description": "Camera 2 relative to Camera 1 (realtime)",
        "translation_m": t_final.tolist(),
        "baseline_distance_m": distance_final,
        "rotation_rvec_rad": rvec_final.tolist(),
        "rotation_matrix": R_final.tolist(),
        "euler_deg": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)},
        "smoothing_window": SMOOTH_WINDOW
    }
    with open("relative_pose_realtime_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("📁 结果已保存为: relative_pose_realtime_result.json")
