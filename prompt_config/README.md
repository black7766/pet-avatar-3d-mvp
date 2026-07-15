# 动作提示词配置

需要调整动作时，只修改 [`actions.py`](actions.py)。当前可编辑动作：

- `idle.video`：静息循环动作。
- `fast_walk.state_frame`：快走首帧姿态。
- `fast_walk.video`：快走循环动作。
- `sleep.state_frame`：睡眠首帧姿态。
- `sleep.video`：睡眠循环动作。

不要把背景、绿幕、灯光、分辨率、时长、音频、固定镜头、完整身体或身份一致性要求写进 `actions.py`。这些生产约束统一保存在 `locked.py`，生成时自动拼接，避免改动作时破坏抠图条件。

修改后执行：

```powershell
python -m unittest tests.test_prompt_config -q
```

现有 `prompts.py` 是兼容入口，`poc.py` 无需修改导入方式。
