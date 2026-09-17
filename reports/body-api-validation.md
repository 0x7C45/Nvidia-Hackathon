# Body API v1 验证记录

验证环境：本地 Apple Silicon、项目 `.venv`、MaleCNS v1.0 固定版本，完整图为 166,700 个神经元、25,582,938 条有向连接、124,177,617 个突触接触。

本次通过了：

- `tests/test_body.py` 与 `tests/test_web.py`：**36 passed**，身体协议代码覆盖率 **97%**。
- 上游 `test_connectome.py` 与 `test_doom_reference.py`：**15 passed**。
- 前端生产构建、Ruff、Black、严格 Mypy，以及 Python/JavaScript 客户端语法检查。
- OpenAPI 实际服务端点：10 个 `/api/v1/body/*` 路径，服务标题 `FLYLAB local body bridge`，版本 `1.0.0`。
- Python 客户端完整 session：connect、reset、两次 20 ms step、reward、release。
- JavaScript 客户端完整 session：RGBA → 线性亮度、10 ms step、reward、release。

数值对照中，身体接口以 2×1 视觉输入推进 20 ms，返回的 FLY2 活动包与新的原生模型逐细胞一致；电位和突触电导数组也一致。重试同一个 step 不重复推进；奖励重试返回同一 receipt。正负奖励在 `frozen` 模式下产生相同的后续神经状态，权重和 drive 数组保持不变。

接口的实际约束如下：外部控制器独占输入；`reset → step → reward → release` 是锁步顺序；step 是唯一推进神经时间的操作；动作范围为 `[-1, 1]`，转向正值为右、前进正值为前；视觉最多 128×128，使用固定 R1–R6 UV 双线性采样；遗漏的身体传感器每步归零；未声明通道、坏 ID、越序 step 和跨 origin 写请求都会被拒绝。

当前学习状态：`learning_mode=frozen` 是唯一注册模式，奖励只写 JSON Lines journal，`applied_to_weights=false`。未来规则只能通过 `scripts/body_plasticity.py` 的显式 factory 注册。这样可以先得到可复现的冻结基线，再单独比较加入可塑性后的训练前后表现、权重变化和记忆保留。

交接入口：

- 协议与接入边界：[docs/body-api.md](../docs/body-api.md)
- Python 客户端：[sdk/flybody.py](../sdk/flybody.py)
- Browser/Node 客户端：[sdk/flybody.mjs](../sdk/flybody.mjs)
- 3D 场景循环：[examples/game_loop.mjs](../examples/game_loop.mjs)
- 机器契约：`http://127.0.0.1:8787/openapi.json`
