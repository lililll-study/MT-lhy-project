"""
MTL AI Competition - 训练脚本
特征工程 → 交叉验证评估 → 全量训练 → 保存模型到 models/
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

DATA_DIR = r'data/kaggle_upload'
MODEL_DIR = 'models'
np.random.seed(42)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据 & 预处理
# ============================================================
print('[1] 加载训练数据...')
train_series = pd.read_csv(os.path.join(DATA_DIR, 'train_series.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))

SPC_COLS = [c for c in train_series.columns if c.startswith('spc_')]

def preprocess(df):
    df = df.copy()
    cols = [c for c in df.columns if c not in ['sample_id','batch_id','t_rel','window_pct','control_type']
            and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df.groupby('sample_id')[cols].ffill().bfill()
    df[cols] = df[cols].fillna(0)
    return df

train_series = preprocess(train_series)

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
    ms = tr_spc_mean  # 仅训练阶段，无测试集
    ss = tr_spc_std
    sk = tr_spc_skew
    md = tr_spc_med
    cv = tr_spc_cv
    if c.get('spc_mean', 0) > 0: feats.append(ms[indices][:, spc_rank[:c['spc_mean']]])
    if c.get('spc_std', 0) > 0:  feats.append(ss[indices][:, std_rank[:c['spc_std']]])
    if c.get('spc_skew', 0) > 0: feats.append(sk[indices][:, skew_rank[:c['spc_skew']]])
    if c.get('spc_med', 0) > 0:  feats.append(md[indices][:, med_rank[:c['spc_med']]])
    if c.get('spc_cv', 0) > 0:   feats.append(cv[indices][:, cv_rank[:c['spc_cv']]])
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
# 5. 全量训练 & 保存模型
# ============================================================
print('[5] 全量训练 & 保存模型...')
np.random.seed(0)

# --- APC ---
apc_mask = ct == 'apc'
y_apc = y[apc_mask]
c_apc = BEST_CFG['apc']
feats_apc_tr = []
if c_apc.get('spc_mean', 0) > 0:
    cc = np.array([np.corrcoef(tr_spc_mean[apc_mask][:, j], y_apc)[0, 1]
                  if np.std(tr_spc_mean[apc_mask][:, j]) > 0 else 0 for j in range(64)])
    apc_feat_idx = np.argsort(-np.abs(cc))[:c_apc['spc_mean']]
    feats_apc_tr.append(tr_spc_mean[apc_mask][:, apc_feat_idx])
X_apc_tr = np.hstack(feats_apc_tr)
sc_apc = StandardScaler().fit(X_apc_tr)
m_apc = Ridge(alpha=0.1).fit(sc_apc.transform(X_apc_tr), y_apc)
np.savez(os.path.join(MODEL_DIR, 'apc.npz'),
         coef=m_apc.coef_, intercept=m_apc.intercept_,
         scaler_mean=sc_apc.mean_, scaler_std=sc_apc.scale_,
         feat_idx=apc_feat_idx, feat_type='spc_mean')
print(f'  APC: {len(apc_feat_idx)} features, R2={OOF_R2.get("apc", 0):.3f}')

# --- Operator ---
c_op = BEST_CFG['operator']
X_op_all = build_feat(np.arange(len(y)), c_op)
sc_op = StandardScaler().fit(X_op_all)
weights_op = np.array([c_op['weight'] if c == 'operator' else 1.0 for c in ct])
m_op = Ridge(alpha=0.1).fit(sc_op.transform(X_op_all), y, sample_weight=weights_op)
np.savez(os.path.join(MODEL_DIR, 'operator.npz'),
         coef=m_op.coef_, intercept=m_op.intercept_,
         scaler_mean=sc_op.mean_, scaler_std=sc_op.scale_)
print(f'  Operator: {X_op_all.shape[1]} features, R2={OOF_R2.get("operator", 0):.3f}')

# --- Recipe ---
c_rc = BEST_CFG['recipe']
X_rc_all = build_feat(np.arange(len(y)), c_rc)
sc_rc = StandardScaler().fit(X_rc_all)
weights_rc = np.array([c_rc['weight'] if c == 'recipe' else 1.0 for c in ct])
m_rc = Ridge(alpha=0.1).fit(sc_rc.transform(X_rc_all), y, sample_weight=weights_rc)
np.savez(os.path.join(MODEL_DIR, 'recipe.npz'),
         coef=m_rc.coef_, intercept=m_rc.intercept_,
         scaler_mean=sc_rc.mean_, scaler_std=sc_rc.scale_)
print(f'  Recipe: {X_rc_all.shape[1]} features, R2={OOF_R2.get("recipe", 0):.3f}')

# --- 特征排名 & 训练元信息 ---
np.savez(os.path.join(MODEL_DIR, 'train_info.npz'),
         spc_rank=spc_rank, std_rank=std_rank, skew_rank=skew_rank,
         med_rank=med_rank, cv_rank=cv_rank,
         CT_MEANS_recipe=CT_MEANS['recipe'],
         CT_MEANS_operator=CT_MEANS['operator'],
         CT_MEANS_apc=CT_MEANS['apc'],
         y_mean=y.mean())
print(f'  特征排名 & 元信息已保存')

print(f'\n{"="*60}')
print(f'  训练完成')
print(f'  CV MAE:  {cv_mae:.4f}')
print(f'  模型已保存至: {MODEL_DIR}/')
print(f'{"="*60}')
