import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    print("相机已启动，正在预热，请稍候...")
    # 预热：丢弃前 30 帧，让深度稳定
    for _ in range(30):
        pipeline.wait_for_frames()
    
    # 🔴 1. 先询问实测距离
    print("\n" + "-"*40)
    while True:
        try:
            user_input = input("📏 请先用卷尺量好距离，并输入实测距离(米)，例如 0.80: ").strip()
            actual_dist = float(user_input)
            if actual_dist <= 0:
                print("❌ 距离必须是大于 0 的数字！")
                continue
            break
        except ValueError:
            print("❌ 输入无效，请输入数字（例如 0.80）。")

    # 🔴 2. 等待用户对准画面
    print("\n✅ 距离已记录！")
    print("👉 请现在将标定板摆放到刚才测量的距离，并让标定板中心对准相机画面正中心。")
    print("👉 摆好后，在终端按【回车键】开始测量。")
    input("按回车键执行测量...")

    # 🔴 3. 此时才真正抓取一帧数据进行计算
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()

    w, h = depth.get_width(), depth.get_height()
    vals = []
    
    # 取画面中心 20x20 区域
    for x in range(w//2-10, w//2+10):
        for y in range(h//2-10, h//2+10):
            d = depth.get_distance(x, y)
            if d > 0:          # 过滤无效值
                vals.append(d)

    # 🔴 4. 计算与验证
    if len(vals) == 0:
        print("❌ 中心区域没有有效深度值，请检查反光和距离！")
    else:
        mean_d = np.mean(vals)
        print(f"\n📸 拍照成功！")
        print(f"有效像素数: {len(vals)} / 400")
        print(f"中心平均深度: {mean_d:.3f} m")
        print(f"深度标准差: {np.std(vals):.4f} m")

        error_cm = abs(mean_d - actual_dist) * 100
        rel_error = (error_cm / (actual_dist * 100)) * 100
        
        print(f"📏 实测距离: {actual_dist:.3f} m")
        print(f"📐 深度误差: {error_cm:.2f} cm")
        print(f"📊 相对误差: {rel_error:.2f}%")
        
        if error_cm < 2.0:
            print("✅ 深度测量合格（误差 < 2cm）")
        else:
            print("❌ 误差偏大，请检查是否对准标定板中心或距离过远")

except RuntimeError as e:
    print(f"❌ 相机连接失败: {e}")
    print("👉 请检查 USB 是否松动或被其他程序占用。")
except Exception as e:
    print(f"❌ 发生错误: {e}")
finally:
    try:
        pipeline.stop()
    except:
        pass

