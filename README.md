# EmbodiGen

**任务描述进，带标签的真实画面出。**

具身数据引擎的参考实现：仿真引擎持有世界状态、导出精确真值；视频模型只负责把白模画得像真的；验证器证明两者没有走样。

![左：引擎导出的白模；右：照片级化之后](docs/assets/demo.gif)

<sub>左半是引擎的白模，右半是渲染后的结果。这是 `embodigen gen` 的一条真实产物，没有剪辑。</sub>

---

## 跑起来

```bash
pip install -e .
embodigen gen --task "把红色方块放进盒子" --out out --resolution 96
embodigen validate --episode out/episodes/franka-0000
```

零第三方依赖，Python ≥ 3.11，clone 下来就能跑，不需要 GPU。产物在 `out/episodes/franka-0000/`。

## 让 agent 替你装

把下面这段直接粘给 Codex 或 Claude Code：

> 克隆 `https://github.com/wellkilo/embodigen` 到本地，用 `python3 -m venv .venv` 建环境并激活，执行 `pip install -e .`。
> 然后运行 `embodigen gen --task "把红色方块放进盒子" --out out --resolution 96`，
> 读取 `out/episodes/franka-0000/meta/fidelity.json`，告诉我 p95 误差、阈值和是否通过，
> 并数一下 `out/episodes/franka-0000/frames/photoreal/` 里有多少帧。
> 任何一步失败，读完整报错再修，不要跳过。

它装完环境、跑通一条样本、把数字报回来——这就是这个项目的验收标准。仓库根目录有 `AGENTS.md`，agent 会自动读到安装与验证流程。

## 链路

```
任务 → 场景 → 轨迹 → 精确真值 → 照片级化 → 忠实度认证 → 数据集
```

| 阶段 | 现在 | 生产环境会换成 |
|---|---|---|
| `SceneGenerator` | 模板 + 参数化 | 代码智能体自由布置 |
| `TrajectoryGen` | 航点插值 | ManiSkill3 / MimicGen |
| `GTExporter` | 软件光线投射 | ManiSkill3（SAPIEN） |
| `Photorealizer` | 确定性替身 | Wan2.2-Fun-5B-Control |
| `FidelityValidator` | 颜色分割 + 重投影 | 关键点跟踪 / 逆动力学 |
| `DatasetPacker` | LeRobot 形状目录 | mp4 + parquet |

核心承诺只有一条：**深度、分割、位姿、动作是算出来的，不是估出来的**。渲染器只改外观，验证器负责证明它没把几何带偏。

## 验证器

只读渲染像素和相机内参：把夹爪从画面里找回来，与真值投到同一相机上比重投影误差，单位毫米。

| 条件 | p95 | 判定 |
|---|---|---|
| 正常渲染 96px | 2.3 mm | 通过（阈值 12 mm） |
| 注入 25 mm 几何漂移 | 28.1 mm | 拒绝，加严条件重渲 |

不达标不入库；每次重试的种子和条件权重都写进 `meta/provenance.json`，事后可审计。

## 产物

```
out/episodes/franka-0000/
├── meta/            info · provenance · fidelity · scene
├── data/chunk-000/  episode_000000.jsonl
└── frames/          photoreal · white · depth(.f32) · seg(.u16)
```

整个目录自包含：`embodigen validate` 只凭它就能重算忠实度，不需要再跑引擎。

## 展示页

[docs/index.html](docs/index.html) 是完整的展示页：六段流程的滚动演示、可拖动的白模/照片级对比、真实误差曲线。
[在线预览](https://wellkilo.github.io/embodigen/) · 本地 `python3 -m http.server -d docs`

## 边界

渲染器不是照片级的，验证器靠颜色分割，只有一个抓放模板，移动与人形只验证了接口和动作空间——这套代码的价值是**接口与数据契约**，不是画面本身。

## 开发

```bash
python3 -m unittest discover -s tests                   # 12 个测试，约 1 秒
python3 scripts/build_docs_assets.py --out docs/assets  # 重生成展示素材
python3 scripts/build_demo_gif.py                       # 重生成 README 动图
python3 scripts/build_favicon.py --out docs             # 重生成站点图标
```
