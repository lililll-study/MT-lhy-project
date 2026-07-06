"""
MTL AI Competition - 生成提交文件
加载原始预测 → CT校准 → 输出 submission.csv
用法: python make_submission.py [原始预测CSV路径]
"""
import os, sys, warnings, argparse
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MTL AI Competition - 生成提交文件')
parser.add_argument('raw_csv', nargs='?', default='raw_predictions.csv',
                    help='原始预测CSV路径，默认 raw_predictions.csv')
parser.add_argument('--output', '-o', default='submission.csv',
                    help='输出文件名，默认 submission.csv')
parser.add_argument('--test-dir', default=r'data/kaggle_upload',
                    help='测试数据目录（需 sample_submission.csv），默认 data/kaggle_upload')
args = parser.parse_args()

RAW_CSV  = args.raw_csv
OUTPUT   = args.output
TEST_DIR = args.test_dir

# ============================================================
# 1. 加载数据
# ============================================================
print('[1] 加载数据...')
raw = pd.read_csv(RAW_CSV)
sample_sub = pd.read_csv(os.path.join(TEST_DIR, 'sample_submission.csv'))
test_series = pd.read_csv(os.path.join(TEST_DIR, 'test_series.csv'))

sub_sids = sample_sub['sample_id'].tolist()
ct_map_te = test_series.groupby('sample_id')['control_type'].first().to_dict()
ct_arr = np.array([ct_map_te[s] for s in sub_sids])
rc_mask = ct_arr == 'recipe'
op_mask = ct_arr == 'operator'
ap_mask = ct_arr == 'apc'

# ============================================================
# 2. CT校准
# ============================================================
print('[2] CT校准...')

# 加载训练元信息（CT均值等）
info = np.load(os.path.join('models', 'train_info.npz'))
CT_MEANS = {
    'recipe':   float(info['CT_MEANS_recipe']),
    'operator': float(info['CT_MEANS_operator']),
    'apc':      float(info['CT_MEANS_apc']),
}

# 原始模型预测
sub_preds = raw['target_stable_mean'].values

# 校准参数（收缩比例由离线CV网格搜索确定，代码中为固定常量）
SHRINK = {'recipe': 0.35, 'operator': 0.20, 'apc': 0.70}

# 校准目标
#   recipe/apc: 向训练集CT均值收缩（Ridge 正则化会压缩预测方差）
#   operator: 向LOO留一批次预测均值收缩（CV估计值，原值 LOO_OP_MEAN = 24.2171）
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

print(f'  收缩比例: {SHRINK}')
print(f'  校准后均值: rc={p[rc_mask].mean():.2f}, op={p[op_mask].mean():.2f}, apc={p[ap_mask].mean():.2f}')

# ============================================================
# 3. 保存
# ============================================================
print('[3] 保存...')
sub = sample_sub.copy()
sub['target_stable_mean'] = p
sub.to_csv(OUTPUT, index=False)

print(f'\n{"="*60}')
print(f'  生成完成')
print(f'  总体均值:   {p.mean():.2f}')
print(f'  recipe:     {p[rc_mask].mean():.2f}')
print(f'  operator:   {p[op_mask].mean():.2f}')
print(f'  apc:        {p[ap_mask].mean():.2f}')
print(f'  输出:       {OUTPUT}')
print(f'{"="*60}')
