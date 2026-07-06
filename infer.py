"""
MTL AI Competition - 推理脚本
加载模型 → 对测试集推理 → 输出 raw_predictions.csv
用法: python infer.py [测试数据目录路径，如:data/kaggle_upload]
"""
import os, sys, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MTL AI Competition - 推理')
parser.add_argument('test_dir', nargs='?', default=r'data/kaggle_upload',
                    help='测试数据目录路径，默认 data/kaggle_upload')
parser.add_argument('--output', '-o', default='raw_predictions.csv',
                    help='输出文件名，默认 raw_predictions.csv')
args = parser.parse_args()

TRAIN_DIR = r'data/kaggle_upload'
TEST_DIR  = os.path.abspath(args.test_dir)
OUTPUT    = args.output
MODEL_DIR = 'models'

# ============================================================
# 1. 加载模型
# ============================================================
print('[1] 加载模型...')
apc_model = np.load(os.path.join(MODEL_DIR, 'apc.npz'))
op_model  = np.load(os.path.join(MODEL_DIR, 'operator.npz'))
rc_model  = np.load(os.path.join(MODEL_DIR, 'recipe.npz'))
info      = np.load(os.path.join(MODEL_DIR, 'train_info.npz'))

# ============================================================
# 2. 加载测试数据 & 预处理
# ============================================================
print('[2] 加载测试数据...')
print(f'  测试数据目录: {TEST_DIR}')
test_series  = pd.read_csv(os.path.join(TEST_DIR, 'test_series.csv'))
sample_sub   = pd.read_csv(os.path.join(TEST_DIR, 'sample_submission.csv'))

SPC_COLS = [c for c in test_series.columns if c.startswith('spc_')]

def preprocess(df):
    df = df.copy()
    cols = [c for c in df.columns if c not in ['sample_id','batch_id','t_rel','window_pct','control_type']
            and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df.groupby('sample_id')[cols].ffill().bfill()
    df[cols] = df[cols].fillna(0)
    return df

test_series = preprocess(test_series)

# ============================================================
# 3. 特征工程（仅测试集）
# ============================================================
print('[3] 特征工程...')
te_spc_mean = test_series.groupby('sample_id')[SPC_COLS].mean().values
te_spc_std  = test_series.groupby('sample_id')[SPC_COLS].std().fillna(0).values
te_spc_skew = test_series.groupby('sample_id')[SPC_COLS].apply(lambda x: x.skew()).fillna(0).values
te_spc_med  = test_series.groupby('sample_id')[SPC_COLS].median().values
te_spc_cv   = np.divide(te_spc_std, np.abs(te_spc_mean) + 1e-8)

te_ct   = test_series.groupby('sample_id')['control_type'].first().values
te_sids = test_series.groupby('sample_id').size().index.tolist()

# 加载特征排名
spc_rank  = info['spc_rank']
std_rank  = info['std_rank']
skew_rank = info['skew_rank']
med_rank  = info['med_rank']
cv_rank   = info['cv_rank']

def build_feat(indices, c):
    feats = []
    if c.get('spc_mean', 0) > 0: feats.append(te_spc_mean[indices][:, spc_rank[:c['spc_mean']]])
    if c.get('spc_std', 0) > 0:  feats.append(te_spc_std[indices][:, std_rank[:c['spc_std']]])
    if c.get('spc_skew', 0) > 0: feats.append(te_spc_skew[indices][:, skew_rank[:c['spc_skew']]])
    if c.get('spc_med', 0) > 0:  feats.append(te_spc_med[indices][:, med_rank[:c['spc_med']]])
    if c.get('spc_cv', 0) > 0:   feats.append(te_spc_cv[indices][:, cv_rank[:c['spc_cv']]])
    return np.hstack(feats) if feats else np.zeros((len(indices), 1))

# ============================================================
# 4. 推理
# ============================================================
print('[4] 推理...')

def predict_from_npz(model_npz, X):
    sc = StandardScaler()
    sc.mean_ = model_npz['scaler_mean']
    sc.scale_ = model_npz['scaler_std']
    m = Ridge(alpha=0.1)
    m.coef_ = model_npz['coef']
    m.intercept_ = model_npz['intercept']
    return m.predict(sc.transform(X))

te_preds = np.zeros(len(te_sids))

# CT配置（与 train.py 一致）
CFG_OP = {'spc_mean': 15, 'spc_std': 3, 'spc_skew': 4}
CFG_RC = {'spc_mean': 8, 'spc_std': 3, 'spc_skew': 3}

# --- APC ---
apc_feat_idx = apc_model['feat_idx']
X_apc_te = te_spc_mean[:, apc_feat_idx]
te_preds[te_ct == 'apc'] = predict_from_npz(apc_model, X_apc_te[te_ct == 'apc'])

# --- Operator ---
X_op_te = build_feat(np.arange(len(te_sids)), CFG_OP)
te_preds[te_ct == 'operator'] = predict_from_npz(op_model, X_op_te[te_ct == 'operator'])

# --- Recipe ---
X_rc_te = build_feat(np.arange(len(te_sids)), CFG_RC)
te_preds[te_ct == 'recipe'] = predict_from_npz(rc_model, X_rc_te[te_ct == 'recipe'])

print(f'  预测均值: rc={te_preds[te_ct=="recipe"].mean():.2f}, '
      f'op={te_preds[te_ct=="operator"].mean():.2f}, apc={te_preds[te_ct=="apc"].mean():.2f}')

# ============================================================
# 5. 保存原始预测
# ============================================================
print('[5] 保存原始预测...')
sub_sids = sample_sub['sample_id'].tolist()
sids_set = set(sub_sids)
y_mean = float(info['y_mean'])
sub_preds = np.array([te_preds[te_sids.index(s)] if s in sids_set else y_mean for s in sub_sids])

raw = pd.DataFrame({'sample_id': sub_sids, 'target_stable_mean': sub_preds})
raw.to_csv(OUTPUT, index=False)

print(f'\n{"="*60}')
print(f'  推理完成')
print(f'  输出: {OUTPUT}')
print(f'{"="*60}')
