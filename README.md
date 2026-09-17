# MaleCNS 果蝇全连接组本地实验

在本机运行 **166,700 个神经元、25,582,938 条有向连接**的固定权重脉冲神经网络。第一阶段检查完整数据、神经活动、可复现性和本机资源需求。

实际结果见 [本机测试报告](reports/local-readiness.md)。大脑数据保存在项目目录中，可以离线重复运行刺激实验和性能测试。

## 使用

**果蝇赛车：** 运行 `./flybrain race`（已有 `./flybrain web` 服务时直接打开网页），访问 [FLY CIRCUIT 赛车实验场](http://127.0.0.1:8787/race/)。支持 2／3／4 果蝇、机械对手、手动驾驶、跟车镜头与右侧驾驶视野，提供超车、氮气、漂移、过弯奖励，以及控制网络训练、记忆保存和导出。操作与接入规范见 [赛车说明](docs/racing.md)。

多车神经驾驶使用独立的赛车控制网络。右侧集成完整 3D 果蝇脑，保留脑区配色、拖动、缩放、神经元点选和骨架。默认随车观察已改为异步：赛车持续以 1/60 秒推进，大脑独立接收最新真实图像与身体／奖励反馈，界面显示采样时刻和延迟。「原生驾驶实验」保留严格锁步，运行速度由真实脑计算决定。奖励作为后续神经输入中的糖刺激，`frozen` 仍不修改脑内突触。卡顿修复前后实测见 [性能修复记录](reports/racing-performance-fix.md)。

**本地三维网页：** 运行 `./flybrain web` 后打开 [果蝇神经实验室](http://127.0.0.1:8787)。如果外部 3D agent 需要接管身体接口，可运行 `./flybrain web --allow-origin http://127.0.0.1:3000`。

右侧显示全部 166,700 个神经元的实际位置与 114 个官方脑区外形。颜色区分来源类别与脑区，亮度来自模拟器当前 100 ms 神经时间窗口内的实际放电计数。可旋转、缩放、全屏、切换脑区、只看放电细胞，点选细胞查看放电、电位和官方完整骨架。

左侧通过「刺激 / 脑区 / 细胞」切换控制与检查内容；活跃细胞列表可以直接点选，并可聚焦查看真实分支。提供自动轮播、全视野光、左侧光、右侧光、脉冲光和无外部驱动，以及强度、暂停、继续与重置。自动轮播每 2 个模拟秒切换一次。关闭或切到后台的页面会断开放电订阅；没有可见观看页面时，服务暂停神经计算，重新打开后继续。暂停且视角静止时不重绘，相机拖动和新活动到来时才更新画面。

全部位置和脑区外形已存本地。其中 139,662 个点来自胞体坐标，976 个来自通向胞体的标注坐标，26,062 个使用官方骨架上的真实锚点。后者表示神经元的实际分支位置，不是假定的胞体。点选一个尚未缓存的细胞时，会首次下载其完整骨架，之后可离线查看。

本次界面与性能优化的实测结果见 [优化报告](reports/frontend-backend-optimization.md)。全亮场短时对照中，后端推进速度从 1.22× 到 1.25×，改善约 2%；放电包体积减少 83.5%。这些是短时测量；本次继续运行时也观察到约 0.66–0.68×，不能保证持续实时。详细条件与此前十分钟基准见报告。

网页代码与服务的重建步骤：

```sh
.venv/bin/python -m pip install -r requirements-web.txt
npm --prefix web ci
npm --prefix web run build
# 仅在缺少三维缓存时执行以下两行；已完成时无需重跑。
.venv/bin/python scripts/web_data.py
.venv/bin/python scripts/fill_anchors.py
./flybrain web
```

`web/package-lock.json` 固定前端版本，`requirements-web-lock.txt` 记录网页阶段完整 Python 环境。服务只绑定本机回环地址。原始官方表面网格仍保留在 `web-data/source/`；显示外形使用较少三角形，全部神经元位置与模拟连接保持完整。

**身体接口：** [Body API v1 交接说明](docs/body-api.md) 已为外部游戏保留感觉输入、动作读出、奖励、episode 重置、session 独占、OpenAPI、WebSocket 活动流和 JSON Lines 实验记录。Python、JavaScript/Node 客户端分别在 `sdk/`，3D 场景接入边界见 `examples/game_loop.mjs`。当前 `learning_mode` 只有 `frozen`：奖励可记录、可回放，但不会修改权重；后续可在 `scripts/body_plasticity.py` 注册经验证的学习规则。

运行最小身体接线示例：

```sh
PYTHONPATH=sdk .venv/bin/python examples/body_loop.py --steps 3
```

原有命令行实验仍然可用：

在此项目目录打开终端：

```sh
./flybrain smoke
./flybrain benchmark --wall-seconds 600
```

`smoke` 依次运行无外部驱动、暗场、全亮、左半视野亮、右半视野亮、重复全亮六个实验。每组从同一个静息初态开始，持续 2 个模拟秒。它检查刺激是否影响下游活动，并逐神经元、逐 10 ms 时间段核对重复全亮实验的放电计数以及最终电位。

`benchmark` 连续运行指定的真实时间。暗、亮、左、右刺激各持续 1 个模拟秒，循环切换；切换刺激时保留神经状态。默认运行 600 秒，每 30 秒输出进度。速度 `0.5×` 表示真实过去 2 秒，大脑推进 1 秒。按 Ctrl+C 停止；未完成的运行不算通过十分钟验收。

两条命令均调用完整网络，默认关闭学习，不需要联网。日志、数据表和图片写入 `reports/`，同名报告在下一次运行时更新。需要保留某次实验时，先复制其报告。

## 数据准备与环境重建

已完成准备时不需要重复下载或导入。需要重建缓存时运行：

```sh
./flybrain prepare
```

下载会检查已有文件，续传未完成的数据，核对官方文件大小与固定上游版本记录的 SHA-256；随后分批导入、核对全部保留及排除统计、编译本机 C++ 内核。大文件使用八个 HTTP 分段，整个文件通过校验后清理分段文件。

从空目录重建环境的顺序为：

```sh
mkdir -p vendor
git clone https://github.com/nftechie/doomfly.git vendor/doomfly
git -C vendor/doomfly checkout --detach 71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt \
  --build-constraint vendor/doomfly/neural-build-constraints.txt
./flybrain prepare
```

上述命令配合本项目的启动脚本、`requirements.txt` 和 `scripts/` 使用。`requirements-lock.txt` 记录首次验收环境的精确版本。科学计算运行在项目独立环境中；启动器固定 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`VECLIB_MAXIMUM_THREADS=1`。

## 文件位置

| 内容 | 位置 |
| --- | --- |
| 固定版本模拟器源码 | `vendor/doomfly/` |
| 三份官方原始数据、来源校验清单 | `vendor/doomfly/connectome_data/malecns_v1/` |
| 标准化的全部保留神经元和连接 | 原始数据目录下的 `normalized/` |
| 可直接加载的稀疏网络 | `vendor/doomfly/outputs/doom/malecns_v1/graph.npz` |
| 本机编译的内核 | `vendor/doomfly/outputs/doom/libneural.dylib` |
| 数据完整性结果 | `reports/integrity.json`、`reports/data-manifest.json` |
| 刺激实验结果 | `reports/smoke.json`、`reports/stimulus.csv`、`reports/stimulus.png` |
| 性能结果 | `reports/benchmark.json`、`reports/benchmark.csv`、`reports/benchmark.png` |
| 准备阶段与测试日志 | `reports/logs/` |

## 模型与本阶段结果的含义

MaleCNS 发布的是生物连接组。原始连接表的 `weight` 表示突触接触数量；当前模拟器的有效权重为接触数量乘以递质符号和上游固定系数 `0.275`。它不是已经训练好的赛车策略。

运行沿用 DOOMFLY 固定基线：LIF 神经时间步长 0.1 ms，膜时间常数 20 ms，突触时间常数 5 ms，传输延迟 1.8 ms，不应期 2.2 ms。乙酰胆碱使用正号，GABA、谷氨酸和组胺使用负号，未确定及仅调质递质使用上游默认符号。这些动力学与符号是明确的建模假设。

视野位置使用上游根据连接推断的投影。直接视觉输入注入 3,335 个已映射的 R1–R6 神经元；另外 42 个未映射的 R1–R6 及其他神经元仍保留在网络中。`no_drive` 关闭所有外部电流；`dark` 及其他视觉实验保留上游 12 mV 的 lamina 驱动，因此暗场也可能有活动。

模型以紧凑稀疏数组存储全图，并按神经事件计算传播。它没有为了提高速度删去弱连接、自连接或某个脑区。原始表还包括胶质细胞和未解析分割对象；这些对象保留在原文件中，导入报告记录其未作为神经元进入仿真的原因。

运行、刺激响应和数值对照验证的是这套计算模型。本阶段学习关闭；后续需要加入可塑性规则，并用冻结权重对照、训练前后表现和记忆保留实验检验学习是否成立。

## 复现数值验证

在项目根目录运行：

```sh
PYTHONPATH="$PWD/vendor/doomfly" OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  vendor/doomfly/tests/test_connectome.py \
  vendor/doomfly/tests/test_doom_reference.py -q
```

这 15 项测试覆盖连接保留与排除、神经元 ID、递质符号，以及原生内核与 Brian2 对照中的兴奋和抑制连接、延迟、不应期、自连接和变化输入。它们使用小规模数值对照网络；`./flybrain smoke` 另行验证完整网络的刺激响应与重复性。

`requirements-dev.txt` 记录本地入口的代码检查工具版本；安装后可使用 Black、Ruff 和 `mypy --strict scripts` 检查本项目的 Python 入口。身体接口测试位于 `tests/test_body.py`，覆盖协议约束、完整图逐步对照、幂等重试、独占控制、奖励不改权重和 journal。

## 来源

- [Google Research，2026-09-03 公告](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [MaleCNS 官方下载](https://male-cns.janelia.org/download/)；[v1.0 版本记录](https://male-cns.janelia.org/release/)
- [固定版本 DOOMFLY](https://github.com/nftechie/doomfly/tree/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33)
- [Shiu 等，2024，果蝇计算脑模型](https://doi.org/10.1038/s41586-024-07763-9)

MaleCNS 数据按 CC BY 4.0 提供，DOOMFLY 原创代码按 MIT 提供；完整来源说明保留在上游仓库的 `THIRD_PARTY.md`、`THIRD_PARTY_NOTICES.md` 和 `licenses/` 中。
