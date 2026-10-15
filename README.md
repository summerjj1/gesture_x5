# Gesture_X5

基于 RDK X5 的双目 USB 相机预览与 MediaPipe 手势识别项目。项目支持直接
使用 OpenCV 读取相机，也支持调用 X5 BPU 运行手掌检测和 21 点手部关键点
模型；手势识别主流程不需要启动 ROS。

## 1. 功能概览

- 读取 `/dev/video0` 的 MJPEG 相机画面。
- 支持完整画面、左画面、右画面和左右双路识别。
- 使用 X5 BPU 运行 MediaPipe 手掌检测模型。
- 使用 X5 BPU ROI 推理运行 MediaPipe 21 点手部关键点模型。
- 绘制手掌框、7 个手掌检测点、21 点骨架和置信度。
- 默认使用即时几何规则分类手势，无 warmup；官方 `gestureDet` 时序模型仅作可选对比。
- 双手 `Palm` 定时开始录像、双手消失报警、任意单手 `Thumb Up` 定时结束录像。
- 录像保存无识别标注的原始画面，截图保留识别标注便于调试。
- 提供本地 OpenCV 窗口、无 ROS 网页预览和可选 ROS 2/TROS 预览。

当前相机在板端枚举为：

```text
/dev/video0
MJPEG 4000x1200 @ 30/60 FPS
```

它是左右拼接的双目画面：

```text
4000x1200 = 左画面 2000x1200 + 右画面 2000x1200
```

## 2. 计算资源与数据流

手势识别默认不运行 ROS 节点，各部分资源分工如下：

```text
USB 双目相机
    │  MJPEG 4000x1200
    ▼
CPU / OpenCV：解码、左右拆分
    │
    ├─► X5 BPU：手掌检测 192x192
    │
    ├─► X5 BPU：每只手的 21 点 ROI 推理 224x224
    │
             │
             ▼
CPU / OpenCV：关键点几何规则分类、绘图、显示、录像
```

ROS 是软件通信框架，不是计算硬件。手势脚本只是复用了安装在
`/opt/tros/humble/lib/` 下的 BPU 模型文件，运行时不需要执行
`source /opt/tros/humble/setup.bash`。

## 3. 项目结构

```text
Gesture_X5/
├── gesture_x5/                  # 可安装的 Python 包
│   ├── apps/                    # 手势、相机和网页三个应用入口
│   └── core/                    # 不依赖 BPU/OpenCV 的状态机核心
├── native/                      # hbDNNRoiInfer C 桥接源码
├── scripts/                     # 安装、构建与代码检查脚本
├── launch/                      # ROS 2/TROS launch 文件
├── tests/                       # 硬件无关单元测试
├── build/                       # 板端动态库构建产物（自动生成）
├── pyproject.toml               # Python 包与命令行入口定义
├── Makefile                     # check/test/build/install 快捷目标
├── README.md                    # 使用和维护文档
├── run_gesture_opencv.sh        # 推荐的手势识别入口
├── run_camera_opencv.sh         # OpenCV 相机入口
├── run_camera_web.sh            # 无 ROS 网页入口
├── run_ros_camera_web.sh        # 可选 ROS 2/TROS 入口
└── *_opencv.py                  # 兼容旧调用方式的轻量入口
```

运行后可能生成：

```text
build/libhand_roi_bridge.so      # 板端编译产物
recordings/*.{avi,mpeg,mp4}      # 录像
recordings/*.jpg                 # 截图
__pycache__/                     # Python 缓存
```

这些生成物不会提交到 Git。

根目录启动脚本保持向后兼容；业务实现统一放在 `gesture_x5/`，不要再向兼容入口
堆叠新逻辑。状态机核心可以在普通电脑上测试，不要求安装 OpenCV 或连接 X5。

开发检查命令：

```bash
make check       # Python 编译、单元测试、Shell 语法检查
make test        # 只运行状态机单元测试
make build       # 在 X5 上编译原生桥接层
```

## 4. 环境要求

本项目针对当前 RDK X5 系统环境编写：

- Ubuntu / aarch64
- Python 3.10：`/usr/bin/python3`
- OpenCV Python：`cv2`
- NumPy
- D-Robotics `hobot_dnn.pyeasy_dnn`
- X5 DNN SDK：`/usr/include/dnn`、`/usr/lib/libdnn.so`
- GCC，用于编译 `libhand_roi_bridge.so`
- 本地桌面显示时需要 Xorg/XFCE，当前桌面为 `DISPLAY=:0`

检查基础依赖：

```bash
/usr/bin/python3 -c "import cv2, numpy; from hobot_dnn import pyeasy_dnn; print('Python dependencies OK')"
gcc --version
ls /usr/include/dnn/hb_dnn.h /usr/lib/libdnn.so
```

使用的 BPU 模型：

```text
/opt/tros/humble/lib/palm_detection_mediapipe/config/palm_det_192_192.bin
/opt/tros/humble/lib/hand_landmarks_mediapipe/config/hand_224_224.bin
/opt/tros/humble/lib/hand_gesture_detection/config/gestureDet_8x21.hbm  # 可选
```

### 安装手势模型包

在已经配置好 D-Robotics/TROS apt 软件源的 RDK X5 上，执行：

```bash
cd /home/sunrise/code/Gesture_X5
chmod +x install_dependencies.sh
./install_dependencies.sh
```

脚本会安装并检查：

```text
gcc
python3-numpy
python3-opencv
tros-humble-palm-detection-mediapipe
tros-humble-hand-landmarks-mediapipe
```

也可以手动安装：

```bash
sudo apt update
sudo apt install -y \
  gcc \
  python3-numpy \
  python3-opencv \
  tros-humble-palm-detection-mediapipe \
  tros-humble-hand-landmarks-mediapipe
```

这两个 TROS 包分别提供手掌检测和 21 点关键点模型。
安装完成后，脚本还会检查 `hobot_dnn` Python 接口、模型文件和 DNN SDK，
并编译 `libhand_roi_bridge.so`。

默认规则分类不需要 `tros-humble-hand-gesture-detection`。如需测试官方时序模型，
再运行 `./install_dependencies.sh --with-official-gesture` 安装它；脚本只直接加载
其中的模型文件，仍然不需要启动 ROS。

检查模型包是否已经安装：

```bash
dpkg-query -W \
  tros-humble-palm-detection-mediapipe \
  tros-humble-hand-landmarks-mediapipe
```

## 5. 快速开始：左右双路手势识别

进入项目目录：

```bash
cd /home/sunrise/code/Gesture_X5
```

默认显示左右画面，但为了降低延迟，只在左侧运行 BPU 识别：

```bash
./run_gesture_opencv.sh
```

等价的完整参数为：

```bash
./run_gesture_opencv.sh \
  --eye both \
  --compute-eye left \
  --fps 60 \
  --detect-every 2 \
  --max-hands 2 \
  --palm-tiles 2
```

第一次运行或 `native/hand_roi_bridge.c` 更新后，启动器会自动执行：

```bash
./build_hand_roi_bridge.sh
```

窗口快捷键：

| 按键 | 功能 |
|---|---|
| `E` | 左 → 右 → 双路 → 完整画面循环切换 |
| `R` | 开始或停止无识别标注的原始画面录像 |
| `S` | 保存当前带识别标注的截图 |
| `Q` / `Esc` | 退出 |

自动录像默认启用，并使用左镜头作为控制判定源：

```text
两只手同时保持 Palm 3 秒       → 开始录像
录制中两只手同时消失 5 秒      → 红色画面报警 + 终端报警
任意单手保持 Thumb Up 4 秒     → 停止并保存录像
```

录像和截图默认保存到：

```text
/home/sunrise/code/Gesture_X5/recordings/
```

## 6. 识别模式

### 左右显示、只计算左侧（默认，推荐）

```bash
./run_gesture_opencv.sh --eye both --compute-eye left
```

程序将 4000×1200 拆成左右两个 2000×1200 画面，窗口中仍然显示两侧，
但只有左侧标记为 `LEFT [BPU]` 并运行手掌和 21 点模型；右侧标记为
`RIGHT [VIEW]`，只做预览。自动录像也默认使用左侧结果判定。

如果需要恢复左右两侧都运行模型：

```bash
./run_gesture_opencv.sh --eye both --compute-eye both
```

左右同时计算的负载约为单侧的两倍。

只计算右侧但仍显示两侧：

```bash
./run_gesture_opencv.sh \
  --eye both \
  --compute-eye right \
  --control-eye right
```

### 只识别左路或右路

```bash
./run_gesture_opencv.sh --eye left
./run_gesture_opencv.sh --eye right
```

### 未拆分的完整画面

```bash
./run_gesture_opencv.sh --eye full
```

`full` 会把 4000×1200 超宽画面作为一张图进行推理，不等同于 `both`。
双目相机通常应使用 `both`。

## 7. 自动录像、报警与手动录像

### 自动录像状态机

双目左右画面会看到同一场景，因此不能把左镜头中的一只手和右镜头中的同一只
手计算成两只。程序默认只使用左镜头的识别结果判断自动录像条件，但保存的是
左右拼接的原始相机画面，不包含手掌框、关键点、倒计时、报警或 `REC` 标记。

默认条件：

1. 左镜头内至少检测到两只手，且两只手都是 `Palm`，连续保持 3 秒后开始录像。
2. 录像期间完全检测不到手，连续 5 秒后显示红色
   `ALARM: BOTH HANDS LOST`，同时在运行终端输出报警并触发终端响铃。
3. 任意一只手重新出现后解除“同时消失”报警。
4. 左镜头内任意一只手为 `Thumb Up`，连续保持 4 秒后结束并保存录像。

`Palm` 和 `Thumb Up` 计时默认都允许最多 1 秒的短暂漏检，因此单帧或少量帧
识别为 `Unknown` 不会立即清空整个进度；超过容错时间才会重新计时。

使用右镜头作为控制判定源：

```bash
./run_gesture_opencv.sh --eye both --control-eye right
```

自定义时间：

```bash
./run_gesture_opencv.sh \
  --palm-start-seconds 3 \
  --missing-alarm-seconds 5 \
  --thumb-stop-seconds 4 \
  --gesture-grace-seconds 1
```

关闭自动录像，只保留手动按键：

```bash
./run_gesture_opencv.sh --no-auto-record
```

为了避免误触发，自动录像要求同一控制镜头内检测到两只手，并且手势连续满足
完整时长；中途任何一次识别结果不满足都会重新计时。切换到不包含控制镜头的
视图时自动计时暂停，不会产生“手消失”的误报警。

### 手动录像与截图

启动后按 `R` 可以开始或停止录像，也可以在启动时直接录像：

```bash
./run_gesture_opencv.sh --record
```

指定录像文件：

```bash
./run_gesture_opencv.sh --record recordings/gesture_test.avi
```

保存为 MPEG：

```bash
./run_gesture_opencv.sh --record-format mpeg
```

也可以直接用扩展名选择格式：

```bash
./run_gesture_opencv.sh --record recordings/gesture_test.mpeg
./run_gesture_opencv.sh --record recordings/gesture_test.mp4
```

默认使用 MJPEG AVI；`mpeg` 使用 MPEG-2，`mp4` 使用 MPEG-4。显示窗口继续绘制
关键点、手势名称、置信度、FPS、自动
控制状态和报警，但这些内容不会写入视频。如果在录像过程中按 `E` 改变画面
尺寸，程序会先结束并保存当前录像。截图仍保存当前带识别标注的显示画面。

MPEG/MP4压缩比更高，但CPU编码开销通常大于MJPEG；如果识别速度明显下降，建议
继续使用默认AVI，或降低 `--record-width`、`--record-fps`。

为避免 4000×1200 MJPEG 编码再次拖慢识别，录像默认等比例缩放到 1280 宽，
并使用开始录像时测得的实际显示帧率。保存原始分辨率或指定录像帧率：

```bash
./run_gesture_opencv.sh --record-width 0 --record-fps 15
```

`--record-width 0` 会明显增加 CPU 编码负载，通常建议保持默认的 1280。

## 8. 性能调节

`--fps 60` 表示向相机申请 60 FPS，并不代表左右两路神经网络也能每秒运行
60 次。当前 OpenCV 显示与推理位于同一线程，推理期间窗口会等待。

双路测试图中，两个手掌和两套 21 点的总推理时间约为 48 ms；真实速度还会
受到手的数量、MJPEG 解码、4000×1200 图像复制、缩放和桌面绘制影响。

如果桌面只有约 7 FPS，可使用较轻的配置：

```bash
./run_gesture_opencv.sh \
  --eye both \
  --fps 60 \
  --detect-every 4 \
  --max-hands 1
```

参数含义：

- `--detect-every N`：每 N 帧运行一次模型，其他帧复用上一次结果。
- `--max-hands N`：每个画面最多处理 N 只手；`both` 模式左右分别计算。
- `--palm-tiles N`：整图漏掉手时自动分块补充检测；`1` 表示关闭。
- `--display-width N`：桌面显示的最大宽度，不改变相机采集分辨率。

若需要让桌面保持高刷新率，应进一步将相机显示与 BPU 推理拆成两个线程，
使桌面持续显示最新帧、识别线程异步更新关键点。

### 实测资源占用

以下数据于 2026-10-14 在当前 RDK X5 上短时采样，条件为
`4000×1200 @ 60 FPS`、`--compute-eye left`、`--max-hands 2`、
`--detect-every 2`，手势识别已运行但尚未触发录像编码。板卡有 8 个 CPU
核心和约 `3062 MiB` 内存。

Linux 进程 CPU 可以超过 `100%`：`100%` 表示占满一个核心，表中的“折合整机
CPU”按进程 CPU 除以 8 计算。RSS 是该进程的常驻物理内存，不是整机已用内存。

| 模式 | 启动参数 | 进程CPU | 折合整机CPU | RSS | BPU瞬时占用 | CPU/BPU温度 |
|---|---|---:|---:|---:|---:|---:|
| OpenCV可视化 | 默认 | `149–180%` | `18.6–22.5%` | `156–181 MiB` | 约 `0–3%` | 约 `74–75/73°C` |
| 无可视化 | `--headless --record-format mpeg` | `107–118%` | `13.4–14.8%` | `144.6–144.8 MiB` | `0–1%` | `74.9–75.1/73.1–73.2°C` |

本轮短时样本中，无可视化模式相对可视化模式大约减少 `20–40%` 的进程 CPU
和 `7–20%` 的 RSS，同时保留手势识别、自动录像、报警及短暂漏检容错：

```bash
./run_gesture_opencv.sh --headless --record-format mpeg
```

`--record-format mpeg` 只选择录像格式；在 Palm 尚未触发录像时不会执行 MPEG
编码。录像开始后，缩放和视频编码会使 CPU 占用高于表中基线，输出原始宽度
（`--record-width 0`）时尤其明显，建议保留默认 `1280`。

BPU 负责手掌检测和 21 点推理；MJPEG 解码、预处理、几何规则、画面绘制和视频
编码主要消耗 CPU。`hrut_somstatus` 显示的是采样瞬间的 BPU 利用率，推理是间歇
执行的，因此可能读到 `0%`，不能把单次读数理解为整段运行的平均值。整机负载
还会受桌面、VS Code Remote 和其他后台服务影响，表格主要用于同一板卡上的
相对比较。

可用以下命令复测当前进程：

```bash
pid=$(pgrep -o -f '^/usr/bin/python3 -m gesture_x5.apps.gesture')
top -p "$pid"
ps -p "$pid" -o pid,etimes,%cpu,%mem,rss,args
hrut_somstatus
```

如果仍需降低占用，优先使用 `--headless`，其次增大 `--detect-every`；不要把
`--max-hands` 降为 `1`，否则双手 Palm 启动录像将无法成立。例如：

```bash
./run_gesture_opencv.sh \
  --headless \
  --compute-eye left \
  --detect-every 3 \
  --max-hands 2 \
  --record-width 1280 \
  --record-format mpeg
```

## 9. 手势类别

默认根据单帧 21 个关键点的伸展状态和夹角进行即时几何规则分类，不需要等待
连续帧，也不会显示 `Warmup`：

- `Palm`
- `Victory`
- `Okay`
- `Awesome`
- `Thumb Up`
- `Thumb Left`
- `Thumb Right`
- `Mute`
- `Unknown`

若需要单独测试 D-Robotics 官方 `gestureDet_8x21.hbm` 时序模型，可显式运行：

```bash
./run_gesture_opencv.sh --gesture-backend official
```

官方模式需要为每只手缓存连续 8 次关键点，刚检测到手时会显示
`Warmup 1/8` 到 `Warmup 7/8`；轨迹中断后会重新缓存，因此不建议用于当前的
实时双手录像控制。

## 10. 主要命令行参数

查看全部参数：

```bash
./run_gesture_opencv.sh --help
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--device` | `/dev/video0` | V4L2 相机设备 |
| `--width` | `4000` | 请求采集宽度 |
| `--height` | `1200` | 请求采集高度 |
| `--fps` | `60` | 请求相机帧率 |
| `--eye` | `both` | `left`、`right`、`both` 或 `full` |
| `--compute-eye` | `left` | 双路显示时在 `left`、`right` 或 `both` 运行 BPU |
| `--display-width` | `1280` | 桌面显示最大宽度 |
| `--detect-every` | `2` | 每隔多少帧进行一次推理 |
| `--max-hands` | `2` | 每路最多处理的手数 |
| `--palm-score` | `0.5` | 手掌检测阈值 |
| `--hand-score` | `0.35` | 21 点结果阈值 |
| `--palm-tiles` | `2` | 整图不足两只手时进行水平重叠补检；`1` 关闭 |
| `--gesture-backend` | `rules` | `rules` 即时几何规则；`official` 可选时序 BPU 分类 |
| `--gesture-model` | 官方模型路径 | `gestureDet_8x21.hbm` 路径 |
| `--gesture-score` | `0.5` | 官方手势分类置信度阈值 |
| `--control-eye` | `left` | 自动录像使用的控制镜头：`left` 或 `right` |
| `--palm-start-seconds` | `3` | 双手 Palm 触发录像所需秒数 |
| `--missing-alarm-seconds` | `5` | 双手同时消失触发报警所需秒数 |
| `--thumb-stop-seconds` | `4` | 任意单手 Thumb Up 停止录像所需秒数 |
| `--gesture-grace-seconds` | `1` | Palm 和 Thumb Up 计时允许的短暂漏检秒数 |
| `--no-auto-record` | 关闭 | 禁用自动录像和自动报警状态机 |
| `--record [FILE]` | 关闭 | 启动时开始录制无标注原始画面，可指定文件 |
| `--record-format` | `avi` | 自动录像格式：`avi`、`mpeg` 或 `mp4` |
| `--record-dir` | `recordings` | 自动录像和截图目录 |
| `--record-width` | `1280` | 录像最大宽度；`0` 表示相机原始宽度 |
| `--record-fps` | 自动 | `0` 使用开始录像时测得的显示帧率 |
| `--image FILE` | 无 | 处理单张图片，不打开相机 |
| `--output FILE` | `gesture_result.jpg` | 单图处理输出文件 |
| `--no-display` / `--headless` | 关闭 | 无窗口运行，并跳过画面复制、绘制和显示缩放 |

## 11. 仅查看 OpenCV 相机

不运行手势模型，只查看相机：

```bash
./run_camera_opencv.sh --fps 60
```

快捷键：

| 按键 | 功能 |
|---|---|
| `1` | 完整画面 |
| `2` | 左半画面 |
| `3` | 右半画面 |
| `R` | 开始或停止录像 |
| `S` | 截图 |
| `Q` / `Esc` | 退出 |

启动时直接录像：

```bash
./run_camera_opencv.sh --fps 60 --record
./run_camera_opencv.sh --fps 60 --record recordings/raw_camera.avi
./run_camera_opencv.sh --fps 60 --record-format mpeg
```

## 12. 无 ROS 网页预览

网页预览只使用 Python、OpenCV 和标准库：

```bash
./run_camera_web.sh --source-fps 60
```

在同一局域网的电脑上打开：

```text
http://192.168.0.160:8088
```

网页可以切换完整、左半和右半画面。按 `Ctrl+C` 停止服务。

此模式目前只预览原始相机，不运行手势识别。

## 13. 可选 ROS 2/TROS 网页预览

如果确实需要 ROS 话题和官方 websocket 页面，可以运行：

```bash
./run_ros_camera_web.sh
```

启动器会先加载：

```text
/opt/tros/humble/setup.bash
```

然后启动 `hobot_usb_cam`、共享内存组件和 websocket。浏览器地址通常为：

```text
http://192.168.0.160:8000
```

这部分才需要 ROS 2/TROS；`run_gesture_opencv.sh` 不需要。

## 14. 无可视化运行

只运行相机采集、BPU识别、自动录像和报警，不创建OpenCV窗口，也不复制或绘制
显示画面：

```bash
./run_gesture_opencv.sh --headless --record-format mpeg
```

无可视化模式没有键盘快捷键，需要在终端按 `Ctrl+C` 退出；自动手势录像仍然有效。
也可以使用兼容参数 `--no-display`，行为完全相同。

## 15. 单张图片测试

使用板上自带的官方示例图验证完整 BPU 链路：

```bash
./run_gesture_opencv.sh \
  --image /opt/tros/humble/lib/palm_detection_mediapipe/config/example.jpg \
  --eye full \
  --output /tmp/gesture_result.jpg \
  --no-display
```

正常结果应包含一个手掌和一套 21 点关键点。

## 16. 对接接口与返回字段

以下是**当前代码实际提供的接口**。手势识别主程序是本地 Python/命令行应用；
`run_camera_web.sh` 的 HTTP 服务只提供原始相机预览，不返回手势、关键点或录像
事件 JSON。需要向其他进程推送识别结果时，可在 Python 调用处按下述字段自行
序列化；目前没有版本化的手势 HTTP 协议。

### 命令行：单图与持续运行

第 10 节列出全部参数及默认值。单图调用示例：

```bash
./run_gesture_opencv.sh --image /path/to/image.jpg --eye full \
  --output /tmp/gesture_result.jpg --headless
```

成功时写入带标注的 JPEG，并在标准输出打印
`palms=N hands=N time=XX.XXms output=/tmp/gesture_result.jpg`；这行文字
用于人工检查，不是稳定的机器解析格式，也不会输出各手的 21 点坐标。持续运行
模式将无标注录像写到 `--record-dir`（默认 `recordings/`）；按键或自动状态机
控制录像，录像事件也只打印到终端。需要逐帧结构化结果时使用下面的 Python
接口。

### Python：单帧手势推理

在 RDK X5 上安装第 4 节依赖并编译 `build/libhand_roi_bridge.so` 后，可以直接
调用 `BpuHandPipeline`。输入必须是 OpenCV 解码的 `uint8` BGR 图像，形状为
`(高度, 宽度, 3)`；对双目拼接图像，应先取左半或右半再调用。

```python
import cv2
from gesture_x5.apps.gesture import BpuHandPipeline

image = cv2.imread("/path/to/left_eye.jpg")
if image is None:
    raise RuntimeError("无法读取图片")

pipeline = BpuHandPipeline(
    palm_score=0.5,
    hand_score=0.35,
    max_hands=2,
    palm_tiles=2,
    gesture_backend="rules",
    gesture_model="",       # rules 模式不会加载此模型
    gesture_score=0.5,       # 仅 official 模式使用
)
try:
    palms, hands = pipeline.process(image, stream_id="left")
    for hand in hands:
        print(hand.gesture, hand.landmarks[8, :2], hand.score)
finally:
    pipeline.close()
```

`process(image, stream_id="default", bootstrap_gesture=False)` 返回
`(palms, hands)`，两者均为列表；没有检测结果时返回空列表，不返回 `None`。
`stream_id` 用于区分 `official` 时序分类器的不同画面；默认 `rules` 模式不需要
跨帧跟踪。`bootstrap_gesture` 仅用于官方分类器的单图测试。

| 返回对象 | 字段 | 类型与含义 |
|---|---|---|
| `Palm` | `box` | `(x1, y1, x2, y2)`，图像像素坐标，右下角为边界坐标 |
| `Palm` | `score` | `float`，手掌检测分数 |
| `Palm` | `keypoints` | `numpy.ndarray`，形状 `(7, 2)`，手掌检测点的像素 `x, y` |
| `Hand` | `landmarks` | `numpy.ndarray`，形状 `(21, 3)`，每点的 `x, y, z`；`x/y` 是像素坐标，`z` 是模型深度值按 ROI 尺寸缩放后的量，不能直接当作物理距离 |
| `Hand` | `score` | `float`，21 点手部模型的分数；低于 `hand_score` 的手不会进入 `hands` |
| `Hand` | `handedness` | `float`，模型原始左右手分数；本项目没有据此输出确定的左/右手标签 |
| `Hand` | `roi` | `(x1, y1, x2, y2)`，用于 21 点推理的图像像素区域 |
| `Hand` | `gesture` | `str`，第 9 节列出的手势名称；无法分类为 `Unknown`；官方模式预热时为 `Warmup n/8` |
| `Hand` | `gesture_score` | `float`；`rules` 模式固定为 `1.0`，**不是概率**；`official` 模式为模型分类分数，预热时为 `0.0` |
| `Hand` | `track_id` | `int`；`rules` 模式为 `-1`；`official` 模式的临时轨迹编号，不能用作持久身份 ID |

所有坐标均相对于传给 `process()` 的图像左上角。若输入是右半画面，想映射回
`4000×1200` 原始拼接帧，需将 `box`、`roi`、`keypoints` 和 `landmarks` 的
`x` 坐标加上半幅宽度（当前为 `2000`）。`palms` 是手掌候选，`hands` 是通过
21 点分数阈值的结果，两个列表不保证一一对应。MediaPipe 的 21 点索引遵循
常用顺序：`0` 为手腕，拇指指尖 `4`、食指指尖 `8`、中指指尖 `12`、无名指
指尖 `16`、小指指尖 `20`。

### Python：自动录像控制事件

`GestureRecordController` 只计算状态和事件，不负责打开相机或写视频文件。
调用方需要保存实际录像状态，并在每次更新后处理返回的 `(start, stop)`：

```python
import time
from gesture_x5.core.control import GestureRecordController

controller = GestureRecordController(3.0, 5.0, 4.0, 1.0)
recording = False

# 每帧用 pipeline.process(...) 得到当前控制视角的 hands。
start, stop = controller.update(hands, recording, time.monotonic())
if start:
    recording = True   # 调用方在此创建录像文件
if stop:
    recording = False  # 调用方在此关闭并保存录像
print(controller.alarm, controller.status)
```

| 输入/输出 | 类型 | 约定 |
|---|---|---|
| `hands` | `list` 或 `None` | 当前控制视角的手势对象列表，元素只需有 `gesture: str` 属性；`[]` 表示本帧检测到 0 只手，`None` 表示控制视角未参与计算且暂停状态机 |
| `recording` | `bool` | 调用方当前是否在录像；状态机不会自行修改录像器 |
| `now` | `float` | `time.monotonic()` 的秒数，连续调用必须使用同一时间基准 |
| `start` | `bool` | 双手 `Palm` 满 3 秒时单次返回 `True`，调用方应启动录像 |
| `stop` | `bool` | 录制中任意一手 `Thumb Up` 满 4 秒时单次返回 `True`，调用方应结束录像 |
| `controller.alarm` | `bool` | 录制中 `hands == []` 满 5 秒后为 `True`；重新检测到手后恢复 `False` |
| `controller.status` | `str` | 人类可读的进度/报警文字，不建议解析它来判断事件 |

Palm 和 Thumb Up 的短暂漏检容差默认为 1 秒。调用方应持续传入每帧的检测
结果；若推理隔帧执行，可复用最近一次结果，与主程序 `--detect-every` 的行为
一致。`hands=None` 会重置计时与报警状态，不能用它表示“没有检测到手”。

### HTTP：原始相机预览

运行 `./run_camera_web.sh` 后，可从同一局域网访问：

| 请求 | 返回内容 |
|---|---|
| `GET /` | HTML 预览页面，`Content-Type: text/html` |
| `GET /status` | JSON，相机状态；`Content-Type: application/json` |
| `GET /stream.mjpg?view=left&width=1280&fps=15&quality=80` | 持续的 MJPEG 流，`Content-Type: multipart/x-mixed-replace; boundary=frame`，每段为一张 JPEG |

`/stream.mjpg` 的 `view` 可取 `full`（默认）、`left`、`right`；`width`
默认 `1280`，限制在 `320–1920`；`fps` 默认 `15`，限制在 `1–30`；
`quality` 默认 `80`，限制在 `40–95`。这些参数影响输出流，不改变相机
采集参数。`/status` 的返回示例：

```json
{"opened": true, "error": "", "width": 4000, "height": 1200, "frame_id": 1234}
```

| JSON 字段 | 类型 | 含义 |
|---|---|---|
| `opened` | `bool` | 相机当前是否打开 |
| `error` | `str` | 最近的相机错误；正常时通常为空字符串 |
| `width` / `height` | `int` | 实际读取到的原始帧尺寸；首帧到达前为 `0` |
| `frame_id` | `int` | 已读取帧计数，从 `0` 开始递增；不是时间戳 |

这个 HTTP 服务不调用 BPU，也没有 `/gestures`、`/landmarks` 或录制控制端点。
调用 Python 推理入口是当前获取结构化手势结果的方式。

## 17. 常见问题

### 相机无法打开

同一时间通常只能有一个进程占用 `/dev/video0`。检查占用者：

```bash
fuser -v /dev/video0
pgrep -af "gesture_opencv|camera_opencv|camera_web|hobot_usb_cam"
```

先正常退出已有相机程序，再启动目标程序。不要同时启动 OpenCV 和 ROS 相机
节点。

### SSH 中出现 `could not connect to display` 或 Qt `xcb` 错误

优先通过项目启动脚本运行，它会自动绑定板载桌面：

```bash
./run_gesture_opencv.sh
```

如果仍然失败，可以手动设置：

```bash
export DISPLAY=:0
export XAUTHORITY=/home/sunrise/.Xauthority
./run_gesture_opencv.sh
```

如果板端没有正在运行的图形桌面，请使用 `--no-display` 或网页预览。

### `ros2: command not found`

只在使用 ROS 模式时需要：

```bash
source /opt/tros/humble/setup.bash
```

建议直接运行 `./run_ros_camera_web.sh`，手势识别模式不需要 `ros2`。

### `AMENT_TRACE_SETUP_FILES: unbound variable`

这是在 `set -u` 状态下加载 TROS 环境导致的。项目的
`run_ros_camera_web.sh` 已在加载环境后再启用 `set -u`，不要在其前面执行
`set -u`。

### 找不到 `libhand_roi_bridge.so`

重新编译：

```bash
./build_hand_roi_bridge.sh
ldd ./build/libhand_roi_bridge.so
```

该桥接层调用 `hbDNNRoiInfer`，因为 21 点模型是 X5 的 resizer/ROI 模型，
不能使用普通的 `pyeasy_dnn.forward()`。

### 有手掌框但没有 21 点

- 保证手掌完整进入画面，并避免严重运动模糊。
- 尝试将 `--hand-score` 从 `0.35` 调低到 `0.25`。
- 确认扩展后的手部区域没有被图像边界大量裁切。
- 使用 `--eye both`，不要对双目超宽图直接使用 `--eye full`。

### 画面有两只手但只识别出一只

- 先看状态栏：`P` 是手掌检测数量，`H` 是通过关键点模型的手数量。
- `P1/H1` 表示手掌模型漏检；默认 `--palm-tiles 2` 会自动进行重叠分块补检。
- `P2/H1` 表示第二只手已找到，但关键点置信度不足，可尝试 `--hand-score 0.25`。
- 双手尽量不要重叠，并让完整手腕和指尖都进入左视角。

### 桌面只有几 FPS

这是同步双路推理、MJPEG 解码和绘图的端到端 FPS，并非相机协商帧率。
可以先使用 `--detect-every 4 --max-hands 1`，或只识别一侧。

### 检查相机真实格式和帧率

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --get-fmt-video --get-parm
```

## 18. 已验证状态

- 相机：`4000x1200 MJPEG @ 60 FPS` 可正常协商。
- 手掌模型：X5 BPU 推理正常。
- 21 点模型：`hbDNNRoiInfer` 推理正常。
- 官方手势分类模型：`gestureDet_8x21.hbm` 通过原生 `hbDNNInfer` 调用。
- 官方单图：检测到 1 个手掌和 1 套 21 点。
- 左右双路测试：检测到 2 个手掌和 2 套 21 点，总推理约 48 ms。
- OpenCV 本地窗口、截图和 MJPEG AVI 录像可用。
- 主手势流程不启动 ROS 节点。
