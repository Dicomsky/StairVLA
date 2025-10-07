# StarVLA (tarVLA)
一个让 Vision-Language-Action (VLA) 模型开发重新像搭积木一样灵活的模块化代码库  


> 核心理念：Top-Down 拆分、单一职责、高内聚低耦合、全部组件可插拔（Lego-like）。

<!-- 优先使用 PNG/SVG，可保留 PDF 作为补充 -->
![StarVLA Roadmap](assets/starVLA_arg.png)

[StarVLA Roadmap](assets/starVLA_arg.pdf)
**图例**：实线框 = 已支持 | 无边框 = Coming Soon | 欢迎 PR 扩展

- [ ] Various VLA Frameworks：基于 Qwen2.5-VL 的多种框架拼装  
- [ ] Various VLA Benchmarks：主流 Robot Manipulation benchmark 训练 + 评测打通  
- [ ] Various Training Strategies：更多混合 / 分阶段 / 课程式策略插件  

---

## 1. 为什么有 StarVLA？
当前 VLA 研发痛点：
- 代码耦合：数据、模型、推理、环境绑定到一起，复用成本高  
- 迭代慢：替换一个感知/控制模块需改多处  
- 训练 / 推理 pipeline 不透明：forward/inference 链路难追踪  
 

StarVLA 提供：
- Framework = 唯一对外模型入口，可单文件阅读/执行
- Modules = 感知、融合、决策、动作头等积木
- Dataloader 直接产出“原始语义模态”而非过度预处理
- 全局单一 config（集中 + 可 CLI 覆盖 + 运行期快照）
- Trainer 支持多数据源（dict 形式调度）
- Inference 一致走 websocket，Sim 通过 sim_interface 解耦
- Dict 作为参数/输出签名，易扩展 & 容错

---

## 2. 特性概览
- VLA 框架：多种架构（基于 Qwen2.5-VL，可插入控制/规划模块）
- Benchmark：打通主流机器人操作数据与评测（持续扩展）
- 训练策略：支持多源混合、阶段式、策略插拔
- 数据支持：对齐 LeRobot / GR00T 风格，可扩任意自定义源
- 推理统一：WebSocket + model2sim_interface.py
- 高可视化：单文件可跑的 Framework / Dataset 便于审查
- 极简扩展：新增模块无需侵入全局

---


## 3. 设计原则（Lego Philosophy）
1. Framework.py = 唯一外部入口（可 python framework/my_vla_framework.py 跑通 forward + inference demo）  
3. Dataloader 返回“最原始”对象：PIL Image / str / normed actions / state（预处理延后到 Framework）。  
4. 全局 config：读取 oxe.yaml → CLI 覆盖 → 运行时冻结副本存入 save_path。  
5. Trainer 维护 {name: dataloader} dict，多源策略化调度。  
6. Inference：一律 WebSocket，Sim 通过 adapter（model2sim_interface.py）桥接。  
7. 输入输出签名：dict，可包含冗余字段（向后兼容）。  
8. 可观测性：所有关键 forward path 应可单步打印/trace。  

---


## 4. 编写一个新 Framework
步骤：
1. 复制 base_framework.py → my_framework.py  
2. 定义 init：注册所需 modules（vision_encoder / policy_head / tokenizer 等）  
3. 实现 forward(self, batch: Dict) → Dict：含 loss / metrics / actions  
4. 实现 inference(self, obs: Dict) → action Dict  
5. 添加一个 demo main：构造假 batch，跑 forward 与 inference  
6. 在 trainer config 中引用：model.framework = my_framework  


---


## 5. 贡献指南
1. Fork & 新建分支：feat/<name>
2. 新模块：放入 framework/modules/xxx
3. 提供最小可运行 demo（python your_file.py）
4. 更新 tests/（若含公共逻辑）
5. PR 模板需含：
   - 背景
   - 变化点
   - 性能/收敛影响（若适用）
6. 代码风格：PEP8 + 黑格式化 (black) + isort

---

## 6. FAQ
Q: 为什么不把预处理放 dataloader?  
A: 统一到 Framework，使模型差异化处理自由化，减少数据侧分叉。

Q: 可以不用 Qwen2.5-VL 吗？  
A: 可以。实现一个新 vision+language 模块并在 Framework 中组合。

Q: 多 dataloader 的 loss 如何加权？  
A: Trainer 中可配置 weight dict，或在框架 forward 聚合。

Q: 推理速度如何优化？  
A: 关闭多余 debug 字段、开启模型半精度、缓存 tokenizer。

---

## 7. 引用 (Citation)
(即将添加 BibTeX，占位)
```
@misc{starvla2025,
  title  = {StarVLA: Modular Vision-Language-Action Codebase},
  author = {...},
  year   = {2025}
}
```

---

## 8. 致谢
参考与灵感：LeRobot, GR00T, DeepSpeed, 各类开源 VLM / 控制实践。  
Codeabse fork from InternVLA-M1

---


