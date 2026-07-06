"""
MTL AI Competition - 最终提交流水线
端到端：SPC特征聚合 → Ridge集成训练 → CT校准 → submission_best.csv

方法:
  1. SPC聚合特征 (mean/std/skew/med/cv) + CT特异性特征选择
  2. Ridge回归 + 30 seed Ensemble（全量训练 + CV评估）
  3. CT校准：向参考均值收缩，补偿Ridge的正则化偏差
      recipe=35%, operator=20%, apc=70%
"""
import os, sys, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

# 命令行参数: python run.py [测试数据目录路径]
#   无参数 → 默认使用 ../data/kaggle_upload
#   有参数 → 训练数据仍用默认路径，测试数据用指定目录
parser = argparse.ArgumentParser(description='MTL AI Competition - 端到端流水线')
parser.add_argument('test_dir', nargs='?', default=r'../data/kaggle_upload',
                    help='测试数据目录路径（test_series.csv, sample_submission.csv），默认 ../data/kaggle_upload')
parser.add_argument('--output', '-o', default='submission_best.csv',
                    help='输出文件名，默认 submission_best.csv')
args = parser.parse_args()

TRAIN_DIR = r'../data/kaggle_upload'
TEST_DIR  = os.path.abspath(args.test_dir)
OUTPUT    = args.output
np.random.seed(42)

# ============================================================
# 1. 加载数据 & 预处理
# ============================================================
print('[1] 加载数据...')
print(f'  训练数据: {TRAIN_DIR}')
print(f'  测试数据: {TEST_DIR}')
train_series = pd.read_csv(os.path.join(TRAIN_DIR, 'train_series.csv'))
train_labels = pd.read_csv(os.path.join(TRAIN_DIR, 'train_labels.csv'))
test_series  = pd.read_csv(os.path.join(TEST_DIR, 'test_series.csv'))
sample_sub   = pd.read_csv(os.path.join(TEST_DIR, 'sample_submission.csv'))

SPC_COLS = [c for c in train_series.columns if c.startswith('spc_')]

def preprocess(df):
    df = df.copy()
    cols = [c for c in df.columns if c not in ['sample_id','batch_id','t_rel','window_pct','control_type']
            and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df.groupby('sample_id')[cols].ffill().bfill()
    df[cols] = df[cols].fillna(0)
    return df

train_series = preprocess(train_series)
test_series  = preprocess(test_series)

# ============================================================
# 2. 特征工程
# ============================================================
print('[2] 特征工程...')
labels_dict = dict(zip(train_labels['sample_id'], train_labels['target_stable_mean']))
sample_ids  = train_series.groupby('sample_id').size().index.tolist()
y      = np.array([labels_dict[s] for s in sample_ids])
groups = train_series.groupby('sample_id')['batch_id'].first().values
ct     = train_series.groupby('sample_id')['control_type'].first().values

tr_spc_mean = train_series.groupby('sample_id')[SPC_COLS].mean().values
tr_spc_std  = train_series.groupby('sample_id')[SPC_COLS].std().fillna(0).values
tr_spc_skew = train_series.groupby('sample_id')[SPC_COLS].apply(lambda x: x.skew()).fillna(0).values
tr_spc_med  = train_series.groupby('sample_id')[SPC_COLS].median().values
tr_spc_cv   = np.divide(tr_spc_std, np.abs(tr_spc_mean) + 1e-8)

te_spc_mean = test_series.groupby('sample_id')[SPC_COLS].mean().values
te_spc_std  = test_series.groupby('sample_id')[SPC_COLS].std().fillna(0).values
te_spc_skew = test_series.groupby('sample_id')[SPC_COLS].apply(lambda x: x.skew()).fillna(0).values
te_spc_med  = test_series.groupby('sample_id')[SPC_COLS].median().values
te_spc_cv   = np.divide(te_spc_std, np.abs(te_spc_mean) + 1e-8)

te_ct   = test_series.groupby('sample_id')['control_type'].first().values
te_sids = test_series.groupby('sample_id').size().index.tolist()

# 特征排名 (基于与目标的相关性)
spc_corr = np.array([np.corrcoef(tr_spc_mean[:, i], y)[0, 1] for i in range(64)])
spc_rank = np.argsort(-np.abs(spc_corr))
std_corr = np.array([np.corrcoef(tr_spc_std[:, i], y)[0, 1] if np.std(tr_spc_std[:, i]) > 0 else 0 for i in range(64)])
std_rank = np.argsort(-np.abs(std_corr))
skew_corr = np.array([np.corrcoef(tr_spc_skew[:, i], y)[0, 1] if np.std(tr_spc_skew[:, i]) > 0 else 0 for i in range(64)])
skew_rank = np.argsort(-np.abs(skew_corr))
med_corr = np.array([np.corrcoef(tr_spc_med[:, i], y)[0, 1] for i in range(64)])
med_rank = np.argsort(-np.abs(med_corr))
cv_corr = np.array([np.corrcoef(tr_spc_cv[:, i], y)[0, 1] if np.std(tr_spc_cv[:, i]) > 0 else 0 for i in range(64)])
cv_rank = np.argsort(-np.abs(cv_corr))

# ============================================================
# 3. CT特异性配置
# ============================================================
print('[3] CT特异性配置...')
CT_NAMES = ['recipe', 'operator', 'apc']
CT_MEANS = {c: y[ct == c].mean() for c in CT_NAMES}
print(f'  训练集 CT 均值: {CT_MEANS}')

BEST_CFG = {
    'apc':      {'spc_mean': 3, 'alpha': 0.1},
    'operator': {'spc_mean': 15, 'spc_std': 3, 'spc_skew': 4, 'weight': 2, 'alpha': 0.1},
    'recipe':   {'spc_mean': 8, 'spc_std': 3, 'spc_skew': 3, 'weight': 2, 'alpha': 0.1},
}

def build_feat(indices, c, is_te=False):
    feats = []
    ms = te_spc_mean if is_te else tr_spc_mean
    ss = te_spc_std if is_te else tr_spc_std
    sk = te_spc_skew if is_te else tr_spc_skew
    md = te_spc_med if is_te else tr_spc_med
    cv = te_spc_cv if is_te else tr_spc_cv
    if c.get('spc_mean', 0) > 0: feats.append(ms[indices][:, spc_rank[:c['spc_mean']]])
    if c.get('spc_std', 0) > 0: feats.append(ss[indices][:, std_rank[:c['spc_std']]])
    if c.get('spc_skew', 0) > 0: feats.append(sk[indices][:, skew_rank[:c['spc_skew']]])
    if c.get('spc_med', 0) > 0: feats.append(md[indices][:, med_rank[:c['spc_med']]])
    if c.get('spc_cv', 0) > 0: feats.append(cv[indices][:, cv_rank[:c['spc_cv']]])
    return np.hstack(feats) if feats else np.zeros((len(indices), 1))

# ============================================================
# 4. 交叉验证评估
# ============================================================
print('[4] 交叉验证评估...')
oof_preds = np.full(len(y), np.nan)
n_cv_runs = 10

for run in range(n_cv_runs):
    np.random.seed(42 + run)
    folds = GroupKFold(n_splits=5)
    for trn, val in folds.split(np.zeros(len(y)), y, groups):
        val_batch = groups[val]
        selected = []
        for b in set(val_batch):
            b_mask = val[groups[val] == b]
            selected.append(np.random.choice(b_mask))
        selected = np.array(selected)

        for ct_val in CT_NAMES:
            c = BEST_CFG[ct_val]
            va_idx = np.where(ct[selected] == ct_val)[0]
            if len(va_idx) == 0:
                continue
            if ct_val == 'apc':
                tr_mask = (ct[trn] == ct_val)
                y_ct = y[trn][tr_mask]
                feats_tr, feats_va = [], []
                if c.get('spc_mean', 0) > 0:
                    cc = np.array([np.corrcoef(tr_spc_mean[trn][tr_mask][:, j], y_ct)[0, 1]
                                  if np.std(tr_spc_mean[trn][tr_mask][:, j]) > 0 else 0 for j in range(64)])
                    r = np.argsort(-np.abs(cc))[:c['spc_mean']]
                    feats_tr.append(tr_spc_mean[trn][tr_mask][:, r])
                    feats_va.append(tr_spc_mean[selected[va_idx]][:, r])
                if c.get('spc_skew', 0) > 0:
                    cc = np.array([np.corrcoef(tr_spc_skew[trn][tr_mask][:, j], y_ct)[0, 1]
                                  if np.std(tr_spc_skew[trn][tr_mask][:, j]) > 0 else 0 for j in range(64)])
                    r = np.argsort(-np.abs(cc))[:c['spc_skew']]
                    feats_tr.append(tr_spc_skew[trn][tr_mask][:, r])
                    feats_va.append(tr_spc_skew[selected[va_idx]][:, r])
                X_tr = np.hstack(feats_tr)
                X_va = np.hstack(feats_va)
            else:
                X_all = build_feat(np.arange(len(y)), c)
                X_tr = X_all[trn]
                X_va = X_all[selected[va_idx]]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)
            if c.get('weight', 1) != 1 and ct_val != 'apc':
                w = np.array([c['weight'] if cv == ct_val else 1.0 for cv in ct[trn]])
                m = Ridge(alpha=c.get('alpha', 0.1)).fit(X_tr_s, y[trn], sample_weight=w)
            elif ct_val == 'apc':
                m = Ridge(alpha=c.get('alpha', 0.1)).fit(X_tr_s, y[trn][tr_mask])
            else:
                m = Ridge(alpha=c.get('alpha', 0.1)).fit(X_tr_s, y[trn])
            oof_preds[selected[va_idx]] = m.predict(X_va_s)

mask = ~np.isnan(oof_preds)
cv_mae = mean_absolute_error(y[mask], oof_preds[mask])
print(f'  CV MAE: {cv_mae:.4f}')

OOF_R2 = {}
for ct_val in CT_NAMES:
    cm = mask & (ct == ct_val)
    if cm.sum() > 1:
        ss_res = np.sum((y[cm] - oof_preds[cm]) ** 2)
        ss_tot = np.sum((y[cm] - y[cm].mean()) ** 2)
        OOF_R2[ct_val] = max(1 - ss_res / ss_tot, 0) if ss_tot > 0 else 0
print(f'  OOF R2: {OOF_R2}')

# ============================================================
# 5. 全量训练 + 30 seed Ensemble
# ============================================================
print('[5] 全量训练 + 30 seed Ensemble...')
n_seeds = 30
te_preds = np.zeros(len(te_sids))

for seed in range(n_seeds):
    np.random.seed(seed)

    # --- APC: 仅用APC样本训练 (APC样本少，需CT特异性特征选择) ---
    apc_mask = ct == 'apc'
    y_apc = y[apc_mask]
    c_apc = BEST_CFG['apc']
    feats_apc_tr, feats_apc_te = [], []
    if c_apc.get('spc_mean', 0) > 0:
        cc = np.array([np.corrcoef(tr_spc_mean[apc_mask][:, j], y_apc)[0, 1]
                      if np.std(tr_spc_mean[apc_mask][:, j]) > 0 else 0 for j in range(64)])
        r = np.argsort(-np.abs(cc))[:c_apc['spc_mean']]
        feats_apc_tr.append(tr_spc_mean[apc_mask][:, r])
        feats_apc_te.append(te_spc_mean[:, r])
    if c_apc.get('spc_skew', 0) > 0:
        cc = np.array([np.corrcoef(tr_spc_skew[apc_mask][:, j], y_apc)[0, 1]
                      if np.std(tr_spc_skew[apc_mask][:, j]) > 0 else 0 for j in range(64)])
        r = np.argsort(-np.abs(cc))[:c_apc['spc_skew']]
        feats_apc_tr.append(tr_spc_skew[apc_mask][:, r])
        feats_apc_te.append(te_spc_skew[:, r])
    X_apc_tr = np.hstack(feats_apc_tr)
    X_apc_te = np.hstack(feats_apc_te)
    sc_apc = StandardScaler()
    m_apc = Ridge(alpha=0.1).fit(sc_apc.fit_transform(X_apc_tr), y_apc)

    # --- Operator: 全量训练 + operator样本加权 ---
    c_op = BEST_CFG['operator']
    X_op_all = build_feat(np.arange(len(y)), c_op)
    X_op_te  = build_feat(np.arange(len(te_sids)), c_op, is_te=True)
    sc_op = StandardScaler()
    weights_op = np.array([c_op['weight'] if c == 'operator' else 1.0 for c in ct])
    m_op = Ridge(alpha=0.1).fit(sc_op.fit_transform(X_op_all), y, sample_weight=weights_op)

    # --- Recipe: 全量训练 + recipe样本加权 ---
    c_rc = BEST_CFG['recipe']
    X_rc_all = build_feat(np.arange(len(y)), c_rc)
    X_rc_te  = build_feat(np.arange(len(te_sids)), c_rc, is_te=True)
    sc_rc = StandardScaler()
    weights_rc = np.array([c_rc['weight'] if c == 'recipe' else 1.0 for c in ct])
    m_rc = Ridge(alpha=0.1).fit(sc_rc.fit_transform(X_rc_all), y, sample_weight=weights_rc)

    # --- 预测 ---
    te_pred = np.zeros(len(te_sids))
    te_pred[te_ct == 'apc']      = m_apc.predict(sc_apc.transform(X_apc_te[te_ct == 'apc']))
    te_pred[te_ct == 'operator'] = m_op.predict(sc_op.transform(X_op_te[te_ct == 'operator']))
    te_pred[te_ct == 'recipe']   = m_rc.predict(sc_rc.transform(X_rc_te[te_ct == 'recipe']))
    te_preds += te_pred / n_seeds

print(f'  模型预测均值: rc={te_preds[te_ct=="recipe"].mean():.2f}, '
      f'op={te_preds[te_ct=="operator"].mean():.2f}, apc={te_preds[te_ct=="apc"].mean():.2f}')

# ============================================================
# 6. CT校准
# ============================================================
print('[6] CT校准...')

sub_sids = sample_sub['sample_id'].tolist()
ct_map_te = test_series.groupby('sample_id')['control_type'].first().to_dict()
ct_arr = np.array([ct_map_te[s] for s in sub_sids])
rc_mask = ct_arr == 'recipe'
op_mask = ct_arr == 'operator'
ap_mask = ct_arr == 'apc'

# 提取提交样本的模型预测
sids_set = set(sub_sids)
sub_preds = np.array([te_preds[te_sids.index(s)] if s in sids_set else float(y.mean()) for s in sub_sids])

# 校准参数（收缩比例由CV网格搜索确定）
SHRINK = {'recipe': 0.35, 'operator': 0.20, 'apc': 0.70}

# 校准目标
#   recipe/apc: 向训练集CT均值收缩（Ridge正则化会压缩预测方差）
#   operator: 向LOO留一批次预测均值收缩（CV估计值）
#   LOO_OP_MEAN = 7个operator样本的留一预测均值（与opt_v10一致，保留完整精度），原值LOO_OP_MEAN = 24.2171
LOO_OP_MEAN = np.mean([23.28, 20.11, 16.96, 32.69, 19.97, 28.38, 28.13])

CALIB_TARGET = {
    'recipe':   CT_MEANS['recipe'],
    'operator': LOO_OP_MEAN,
    'apc':      CT_MEANS['apc'],
}

p = sub_preds.copy()
p[rc_mask] = CALIB_TARGET['recipe']   + (p[rc_mask] - CALIB_TARGET['recipe'])   * (1 - SHRINK['recipe'])
p[op_mask] = CALIB_TARGET['operator'] + (p[op_mask] - CALIB_TARGET['operator']) * (1 - SHRINK['operator'])
p[ap_mask] = CALIB_TARGET['apc']      + (p[ap_mask] - CALIB_TARGET['apc'])      * (1 - SHRINK['apc'])

print(f'  校准目标: {CALIB_TARGET}')
print(f'  收缩比例: {SHRINK}')
print(f'  校准后均值: rc={p[rc_mask].mean():.2f}, op={p[op_mask].mean():.2f}, apc={p[ap_mask].mean():.2f}')

# ============================================================
# 7. 保存
# ============================================================
print('[7] 保存...')
sub = sample_sub.copy()
sub['target_stable_mean'] = p
sub.to_csv(OUTPUT, index=False)

print(f'\n{"="*60}')
print(f'  CV MAE:    {cv_mae:.4f}')
print(f'  总体均值:   {p.mean():.2f}')
print(f'  recipe:     {p[rc_mask].mean():.2f}')
print(f'  operator:   {p[op_mask].mean():.2f}')
print(f'  apc:        {p[ap_mask].mean():.2f}')
print(f'  输出:       {OUTPUT}')
print(f'{"="*60}')
