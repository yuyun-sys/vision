# ROBOCON Vision Assignment 1 - Starter Projects

本仓库包含 Assignment 1 使用的三个代码项目：

```text
ROBOCON-Vision-Assignment1-Starter/
├── README.md
├── VERSION_REQUIREMENTS.md
├── python_A/
│   ├── camera.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
├── python_B/
│   ├── analyze_video.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
└── cpp/
    ├── include/
    │   └── transform.hpp
    ├── src/
    │   ├── main.cpp
    │   └── transform.cpp
    └── README.md
```

## 数据流

```text
摄像头
  │
  ▼
Python Project A
  │  保存未经处理的视频 raw_capture.mp4
  ├──────────────────────┐
  ▼                      ▼
Python Project B         C++ Task
离线分析视频              二值化 + Canny
  │                      │
  ▼                      ▼
advanced_analysis.mp4    cpp_processed.mp4
```

## 环境隔离要求

Project A 与 Project B 的 Python 版本要求故意不兼容：

- Project A: `Python >= 3.9, < 3.11`
- Project B: `Python >= 3.12, < 3.14`

因此不能使用同一个 Python/Conda 环境完成两个项目。请分别创建环境，并以各项目的 `pyproject.toml` 为最终版本要求依据。

## C++ 部分

C++ 项目只提供源码和依赖要求，**故意不提供 `CMakeLists.txt`**。

学生需要：

1. 先理解 `main.cpp`、`transform.cpp`、`transform.hpp`、OpenCV、Eigen 之间的关系；
2. 手动完成一次编译；
3. 再自行编写 `CMakeLists.txt`；
4. 最终使用 CMake 构建。

更详细的版本要求见 [`VERSION_REQUIREMENTS.md`](VERSION_REQUIREMENTS.md)。
