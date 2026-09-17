# FLYLAB Body API v1

这套接口把完整 MaleCNS 模型接到外部身体或游戏。游戏负责世界、相机、物理、碰撞和奖励；本地大脑服务负责感觉输入、神经推进和动作读出。接口只绑定 `127.0.0.1`，没有公网服务。

启动服务：

```sh
./flybrain web --allow-origin http://127.0.0.1:3000
```

外部 agent 可以先读取机器契约：

```text
GET http://127.0.0.1:8787/api/v1/body/capabilities
GET http://127.0.0.1:8787/api/v1/body/ports
GET http://127.0.0.1:8787/openapi.json
```

`/openapi.json` 是 FastAPI 生成的 OpenAPI 文档；实际请求和响应模型以它为准。所有写请求必须带 `X-Flybrain-Local: 1`。从另一个本地端口的浏览器发请求时，需要用启动参数 `--allow-origin` 显式声明该 origin。Python 客户端和无依赖的 browser/Node ESM 客户端分别在 [sdk/flybody.py](../sdk/flybody.py) 和 [sdk/flybody.mjs](../sdk/flybody.mjs)。

## 锁步控制流程

身体控制器一次只能有一个。流程固定为：

```text
POST /sessions
POST /sessions/{session_id}/reset
POST /sessions/{session_id}/step       ← 给感觉输入，推进大脑
POST /sessions/{session_id}/reward     ← 世界推进后反馈奖励
POST /sessions/{session_id}/release
```

`step` 是唯一推进神经时间的接口。`dt_ms` 允许 10–100 ms、且必须是 10 的倍数；内部仍使用 0.1 ms 的原生神经步长。没有新的 `step` 请求时，大脑不会偷偷向前运行。`step_index`、`episode_id` 和请求缓存使超时重试幂等：相同请求不会把神经时间推进两次。

这里的锁步是神经接口的顺序约束。原生大脑实际控制身体时，世界等待返回的动作再推进；纯观察场景可以用独立的世界时钟持续运行，按实际处理能力采样输入，不能用缓慢的神经计算阻塞游戏绘制。这样的输入必须标明两个时钟，不声称神经模拟已达到墙钟实时。

一个最小 Python 循环：

```python
from flybody import FlyBody

with FlyBody(controller_name="racing-scene", preset="visual_bci") as body:
    body.reset(seed=7)
    for _ in range(100):
        frame = {"width": 64, "height": 32, "pixels": [0.0] * (64 * 32)}
        neural = body.step({"vision": frame}, dt_ms=20)
        # 世界把动作映射成自己的转向、油门和刹车单位。
        feedback = world.advance(neural["actions"], seconds=0.020)
        body.reward(feedback["reward"], components=feedback.get("components", {}),
                    terminated=feedback.get("terminated", False),
                    truncated=feedback.get("truncated", False))
```

JavaScript/TypeScript 场景可以直接导入 `sdk/flybody.mjs`。`examples/game_loop.mjs` 展示了从 WebGL RGBA framebuffer 转成线性亮度图、调用神经网络、推进场景和提交奖励的完整边界。`visionFromRGBA` 会处理 WebGL 的上下翻转选项和 sRGB 线性化；它不解释场景，也不替游戏决定相机裁剪。

## 感觉输入

每个 `step` 的 `observation` 可以提供以下之一：

- `vision`：行优先的 `width × height` 线性亮度数组，范围 `[0, 1]`，每边最多 128 像素。服务按照 MaleCNS 固定的 3,335 个 R1–R6 UV 坐标做双线性采样。
- `retina`：长度必须正好为 3,335 的亮度数组。顺序可从 `/ports` 的 `retina[].slot` 查询。
- `sensors`：在创建 session 时声明的身体通道，键名到 `[-1, 1]` 标量。声明示例是触碰、速度、姿态或内部状态；服务按声明的 `gain_mv` 转为额外 LIF drive。未声明的通道和遗漏通道都会被拒绝或置零，不会把上一帧输入粘住。
- `sugar`：明确的内置糖刺激开关。奖励不会自动变成糖刺激。

`observation.context` 是可选的来源记录，兼容现有客户端：

```json
{
  "clock": "sampled",
  "source_time_ms": 12500,
  "source_interval_start_ms": 11800,
  "source_id": "car_1"
}
```

`clock` 为 `sampled` 或 `lockstep`；`source_time_ms` 是相机采样时的世界时间，`source_interval_start_ms` 可标出保留短暂感觉事件的区间起点，`source_id` 标识来源。这些字段仅写入 journal，不注入电流、不改变神经时间；区间起点不得晚于采样时间。神经时间仍以 `step.sim_time_ms` 为准。

自定义身体通道使用稳定的 MaleCNS `bodyId` 字符串，不使用会随数组变动的内部索引：

```json
{
  "name": "touch",
  "neuron_ids": ["10059"],
  "gain_mv": 20
}
```

`/neurons?cell_type=...&side=...` 可以先按官方注释查找稳定 ID。`/ports` 返回实际 retinal、lamina 和 sugar 映射及 UV 坐标。

## 动作输出

`step` 返回 `actions` 和 `readouts`。动作值当前统一为 `[-1, 1]`，`turn` 正值表示向右，`forward` 正值表示向前。它们是可替换的工程读出，不是已经训练好的赛车策略，也不声称等同于真实果蝇的运动行为。

内置两个预设：

- `descending`：DNa02 左右差分给 `turn`，DNp09 与 MDN 的差分给 `forward`。
- `visual_bci`：DNp20 左右差分给 `turn`，DNpe017 的平均放电给 `forward`。

游戏应自行把归一化动作换算成车辆的转向角、油门、刹车或其他身体自由度。服务同时返回这些读出的人口平均放电率、平滑差分和参与的真实神经元，便于记录和做冻结权重对照。

## 奖励与可塑性

`reward` 请求必须引用刚返回的 `episode_id` 和 `step_index`，可以携带总奖励、命名分量，以及 `terminated` / `truncated`。服务会把请求写入该 session 的 JSON Lines journal：

```text
GET /api/v1/body/sessions/{session_id}/events
```

当前唯一可用模式是 `learning_mode: "frozen"`。奖励会被记录和回传累计值，`applied_to_weights` 明确为 `false`；它不会修改连接权重、注入糖刺激或暗中改变神经状态。`reset_learning`、学习状态、权重更新计数和 `scripts/body_plasticity.py` 的 `PlasticityRule` 已作为扩展点保留。未来新增规则必须在 `FACTORIES` 中显式注册，并单独做训练前后、冻结权重和记忆保留验证，不能通过 HTTP 动态加载任意代码。

## 活动流与可视化

原有 `WebSocket /stream` 仍供本地三维网页使用，包格式是无损的 FLY2，代表最近一个 100 ms 观察窗口的全部 166,700 个放电计数。它是可视化活动流，不是身体控制时钟；游戏动作应只使用 `step` 返回的 `actions`。外部场景可同时打开网页，网页会显示“场景控制中”，不能抢占输入控制权。

一次 session 释放后，网页恢复为暂停状态；页面控制可以再次接管。每个 session 的 journal 记录固定上游 commit、完整图规模、配置、每步输入（如 `record_observations` 开启）、动作读出和奖励，便于另一个 agent 复现一个 episode。

## 最小自检

```sh
PYTHONPATH=sdk .venv/bin/python examples/body_loop.py --steps 3
PYTHONPATH=scripts:vendor/doomfly .venv/bin/pytest -q tests/test_body.py
```

自检不会声称模型已经学会游戏；它只确认身体协议、完整图推进、动作读出、奖励记录、重置和固定权重行为可重复。

## 本项目赛车接入

FLY CIRCUIT 位于 `/race/`，由 `./flybrain race` 或现有 `./flybrain web` 服务提供。操作、奖励分量、摄像机约定与后续原生训练路线见 [赛车说明](racing.md)。该说明从属于本文件；新增身体能力和学习模式仍先修改本协议与生成的 OpenAPI。

赛车复用 `sdk/flybody.mjs`，输入绑定赛车的真实 64×32 线性亮度画面；v1 仍只使用一个原生脑实例。2／3／4 果蝇选项使用独立的小型赛车控制网络，其浏览器训练记忆不属于 MaleCNS 突触权重。原生模式仍为 `frozen`，`applied_to_weights=false`。

赛车提供两种连接方式：默认“随车观察”让世界以 1/60 秒实时推进，后台每次仅处理最新一个采样；每次神经计算仍推进 100 ms，但世界不会等待。画面用 WebGL2 异步读回，所有请求保持 `step → reward` 串行，旧图像不排队。驾驶控制器保持不变，原生动作仅供检查。“原生驾驶实验”把原生读出应用到车辆，保留 100 ms `step → 世界推进 → reward` 的严格锁步，比赛速度取决于 CPU 的真实计算能力。

观察期间的奖励按分量累计，在当前神经步骤结束后提交一次；提交过程中新增的奖励留给下一次，避免重发或丢失。短暂氮气、触碰信号保留到下一次采样，并以 `context` 记录来源区间。比赛结束后可完成最后的奖励刺激，但不增加游戏时间。切换观察对象会释放旧 session，再为新对象重置神经状态，普通比赛持续运行。

赛车的神经反馈使用已有 `sensory_channels` 与 `observation`，不改变 `reward` 的冻结语义：

| 感觉通道 | 会话创建时的声明 | 每步输入 |
| --- | --- | --- |
| 图像 | `visual_bci`，既有 R1–R6 投影 | 同一车载相机的 64×32 线性亮度 |
| `body_left` / `body_right` | 查询官方 `SNpp30` 左／右稳定 ID，各 4 个，`gain_mv=30` | 实际速度归一化至 `[0,1]`，再按转向调整左右反馈 |
| `nitro` | 官方 `SApp10`，37 个稳定 ID，`gain_mv=30` | 原生驾驶用当前实际氮气状态；异步观察保留最近来源区间中生效过的氮气，值为 0 或 1 |
| `touch` | 官方 `SNta13`，6 个稳定 ID，`gain_mv=30` | 碰撞冷却中为 1，路外为 0.5，否则为 0；异步观察保留来源区间最大值 |
| 奖励反馈 | 既有 `observation.sugar`，23 个保留的糖感觉神经元 | 收到包含正向超车、氮气、漂移、过弯或圈数奖励的反馈后，下一个神经步骤 `sugar=true`，持续一个 100 ms 神经窗口；同批多个事件可合并 |

这是明确记录的工程感觉映射，不表示果蝇生物学中存在“氮气神经元”，也不把奖励刺激当作学习证明。逐米前进奖励不触发糖刺激。最后一次反馈如有正向行为奖励，会再推进一个带糖刺激的 100 ms 神经窗口，然后结束 episode；异步观察不改变已结束的世界，原生驾驶实验则按原有锁步规则推进。`reward` 本身仍不修改电位、drive 或权重；只有后续显式 `step.observation` 引起神经响应。

赛车右侧脑视图复用神经观察室的 `BrainView`、完整位置、脑区配色和 114 个官方外形。它通过 `/stream` 读取当前 session 的 FLY2 数据，核对 session ID 和 sequence，仅显示真实最新窗口的放电。显示模块自身不推进大脑；收起或离开可见区域会停止绘制并断开该显示订阅。
