# 梅特勒托利多 AI 竞赛 - 最终提交

## 1. 项目简介

- 竞赛：Drug Prophet

- 团队：书香百味知多少 + 李泓奕、王子诺
  
  |     | Kaggle账号      | Kaggle昵称        | 姓名  | 电话          | 邮箱                |
  | --- | ------------- | --------------- | --- | ----------- | ----------------- |
  | 队长  | lihongyiainuo | li hongyi-ainuo | 李泓奕 | 18370297118 | 342032684@qq.com  |
  | 队员  | wangzinuo11li | Wang Zinuo      | 王子诺 | 13133820713 | 3241547973@qq.com |

- 方案介绍：见PPT (Project Overview.pptx)

## 2. 环境要求

- Python：3.10（已在 3.10.20 测试通过）
- 硬件：CPU
- 操作系统：Windows (已在 Windows 11 测试通过)

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 数据准备

从竞赛页面下载数据，放置于 `data/kaggle_upload/` 目录，包含以下文件：

```
data/kaggle_upload/
├── train_series.csv        # 训练集时间序列（SPC + 过程变量）
├── train_labels.csv        # 训练集标签
├── test_series.csv         # 测试集时间序列
└── sample_submission.csv   # 提交模板
```

无需额外预处理，脚本内部自动完成缺失值填充。

## 5. 训练

```bash
python train.py
```

- 预计耗时：< 30 秒（CPU）
- 产出：`models/` 目录（4 个 `.npz` 文件，总大小 < 10 KB）
- 交叉验证集平均误差 (CV MAE) ~3.70，决定系数 (OOF R²) 约 recipe=0.40, operator=0.48, apc=0

## 6. 推理

```bash
# 默认测试数据路径
python infer.py

# 自定义测试数据目录，如data/kaggle_upload
python infer.py data/kaggle_upload
```

- 产出：`raw_predictions.csv`（CT校准前的原始模型预测，作为中间文件）
- 说明：推理过程完全离线，不访问任何外部 API 或云端服务。

## 7. 生成提交文件

```bash
# 从 raw_predictions.csv 生成最终提交
python make_submission.py

# 自定义输入/输出
python make_submission.py raw_predictions.csv -o submission.csv
```

- 输出：`submission.csv`
- 格式：与 `sample_submission.csv` 一致（`sample_id, target_stable_mean`）

### 一键运行（可选）

也可用配套的 `run.py` 一次性完成全部流程：

```bash
python run.py
```

## 8. 目录结构

```
project/
├── train.py                # 训练脚本
├── infer.py                # 推理脚本
├── make_submission.py      # 生成提交文件
├── models/                 # 模型文件（总大小 < 10 KB）
│   ├── apc.npz
│   ├── operator.npz
│   ├── recipe.npz
│   └── train_info.npz
├── run.py                  # 一键运行（可选）
├── README.md
├── requirements.txt
└── data/kaggle_upload/     # 数据目录（需自行放置）
    ├── train_series.csv
    ├── train_labels.csv
    ├── test_series.csv
    └── sample_submission.csv
```

## 9. 依赖许可证

| 库名           | 版本    | 许可证          |
| ------------ | ----- | ------------ |
| numpy        | 2.2.6 | BSD-3-Clause |
| pandas       | 2.3.3 | BSD-3-Clause |
| scikit-learn | 1.7.2 | BSD-3-Clause |

## 10. 常见问题

- Q：没有 GPU 能否运行？
  A：是。本方案仅使用 Ridge 回归，纯 CPU 运行，1 分钟内完成全流程。

- Q：模型文件为什么这么小？
  A：Ridge 回归仅保存系数向量（64维浮点数），三个模型 + Scaler 参数总计 < 10 KB，远低于 200 MB 限制。

- Q：如何使用新测试集推理？
  A：将新测试集 `test_series.csv` 和 `sample_submission.csv` 放入新目录，执行 `python infer.py /path/to/new_dir; python make_submission.py`。

- Q：CT 校准参数是如何确定的？
  A：收缩比例（recipe 35%、operator 20%、apc 70%）通过离线交叉验证网格搜索确定。recipe 与 apc 向训练集 CT 均值收缩；operator 向 7 个 operator 训练样本的留一（LOO）预测均值收缩。

- Q：为什么 APC 收缩 70% 这么高？
  A：APC 训练样本极少（6 个），模型 OOF R²≈0（几乎无预测能力），大幅向 CT 均值收缩是最优策略。
