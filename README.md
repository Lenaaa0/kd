# 轻量化可解释加密流量识别（蒸馏）

本工程实现一个可复现的最小闭环：**教师深度模型（DNN）→ 知识蒸馏 → 可解释学生模型**，并评估识别精度、推理延迟与可解释性输出。

## 你需要准备的数据

目前代码默认读取 **CICFlowMeter/NetFlow 风格的 CSV**（每行一个流，最后一列或指定列为标签）。常见数据集导出的特征表都能对上这个格式。

你需要提供一个或多个 CSV，并在 `configs/default.yaml` 里填写：

- `data.train_csv`: 训练集 CSV 路径
- `data.test_csv`: 测试集 CSV 路径
- `data.label_col`: 标签列名（字符串）

## 快速开始

1) 创建环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) 准备配置

```bash
cp configs/default.yaml configs/local.yaml
```

编辑 `configs/local.yaml`，填入你的 CSV 路径与标签列名。

3) 训练教师模型（MLP）

```bash
python -m src.train_teacher --config configs/local.yaml
```

4) 蒸馏到可解释学生模型（EBM / 可加性模型）

```bash
python -m src.distill_student_ebm --config configs/local.yaml
```

5) 评测（精度 + 延迟）

```bash
python -m src.evaluate --config configs/local.yaml
```

## 输出物

运行后会在 `runs/<timestamp>/` 产出：

- `teacher.pt`: 教师模型权重
- `student_ebm.pkl`: 学生模型
- `metrics.json`: Accuracy/Macro-F1/混淆矩阵等
- `latency.json`: 推理延迟统计
- `explanations/`: 若干样本的解释结果（特征贡献）

## 方法说明（对应任务书）

- **实时性**：学生模型参数少、推理快；并输出延迟统计。
- **可解释性**：学生使用 EBM（Explainable Boosting Machine，广义加性模型），可直接给出每个特征对预测的贡献。
- **蒸馏**：用教师的 soft logits 作为额外监督，使学生在保持可解释的前提下逼近教师决策边界。

