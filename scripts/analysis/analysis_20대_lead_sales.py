# -*- coding: utf-8 -*-
"""
20대 이동인구 선행 → 매출 반응 분석
가설: 특정 행정동에서 20대 유입 증가(YoY)가 전체 매출 증가(YoY)를 1~2분기 선행한다
출력: data/20대_선행매출_분석.xlsx
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parent
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
RAW_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MAP_PATH  = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
OUT_PATH  = ROOT / 'data/20대_선행매출_분석.xlsx'

# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']

_raw = pd.read_csv(RAW_PATH, encoding='utf-8-sig')
raw_cols_idx = [0, 1, 3, 4, 5, 18, 51, 52, 53, 54, 55, 56]
_raw2 = _raw.iloc[:, raw_cols_idx].copy()
_raw2.columns = ['raw_code','dong','yr','q','yrq','sales',
                 'age10','age20','age30','age40','age50','age60']

# 코드 매핑 (migration 7자리 ↔ raw 8자리)
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()   # mig → raw
code_raw2mig = _map.set_index('raw_code')['mig_code'].to_dict()   # raw → mig
dong_name    = _map.set_index('raw_code')['dong_map'].to_dict()

def qnum(yr, q_): return int((yr - 2020) * 4 + q_)

_mig['qnum'] = _mig.apply(lambda r: qnum(r['yr'], r['q']), axis=1)
_raw2['qnum'] = _raw2.apply(lambda r: qnum(r['yr'], r['q']), axis=1)

# raw_code 기준으로 통일
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

# ──────────────────────────────────────────────
# 2. 이동인구 집계 (raw_code 기준)
# ──────────────────────────────────────────────
mig20 = (_mig[_mig['age'] == 20]
         .groupby(['raw_code','qnum'])['mig_total'].sum().reset_index()
         .rename(columns={'mig_total': 'mig20'}))

mig_all_age = _mig.groupby(['raw_code','qnum','age'])['mig_total'].sum().reset_index()

# ──────────────────────────────────────────────
# 3. 매출 집계 (raw_code 기준)
# ──────────────────────────────────────────────
sales_q = (_raw2.groupby(['raw_code','qnum'])['sales'].sum().reset_index())

AGE_SALES = ['age10','age20','age30','age40','age50','age60']
AGE_KR    = {'age10':'10대','age20':'20대','age30':'30대',
             'age40':'40대','age50':'50대','age60':'60이상'}

# ──────────────────────────────────────────────
# 4. YoY 계산
# ──────────────────────────────────────────────
def calc_yoy(df, key, val, new_col):
    df = df.sort_values([key, 'qnum']).copy()
    prev = df[[key, 'qnum', val]].rename(columns={val: 'prev'})
    prev['qnum'] = prev['qnum'] + 4
    df = df.merge(prev, on=[key, 'qnum'], how='left')
    df[new_col] = (df[val] - df['prev']) / df['prev'].replace(0, np.nan)
    return df.drop(columns=['prev'])

mig20   = calc_yoy(mig20,   'raw_code', 'mig20',  'mig20_yoy')
sales_q = calc_yoy(sales_q, 'raw_code', 'sales',  'sales_yoy')

print(f"공통 행정동 수: {len(set(mig20['raw_code']) & set(sales_q['raw_code']))}")

# ──────────────────────────────────────────────
# 5. CCF 분석
# ──────────────────────────────────────────────
LAGS     = range(-2, 5)
MIN_OBS  = 8
R_THRESH = 0.35

ccf_rows = []
common_codes = sorted(set(mig20['raw_code']) & set(sales_q['raw_code']))

for code in common_codes:
    s_s = sales_q[sales_q['raw_code'] == code].set_index('qnum')['sales_yoy'].dropna()
    m_s = mig20[mig20['raw_code'] == code].set_index('qnum')['mig20_yoy'].dropna()
    common_q = s_s.index.intersection(m_s.index)
    if len(common_q) < MIN_OBS:
        continue

    lag_rs = {}
    for lag in LAGS:
        if lag >= 0:
            idx_m = common_q
            idx_s = common_q + lag
        else:
            idx_m = common_q - lag
            idx_s = common_q
        valid = idx_m[idx_m.isin(m_s.index) & idx_s.isin(s_s.index)]
        if len(valid) < MIN_OBS:
            continue
        x = m_s[valid].values
        y = s_s[(valid + lag) if lag >= 0 else valid].values
        try:
            r, p = pearsonr(x, y)
            lag_rs[lag] = (round(float(r), 3), round(float(p), 4))
        except Exception:
            continue

    if not lag_rs:
        continue

    best_lag = max(lag_rs, key=lambda k: lag_rs[k][0])
    row = {'raw_code': code,
           'dong': dong_name.get(code, str(code)),
           'peak_lag': best_lag,
           'peak_r': lag_rs[best_lag][0],
           'peak_p': lag_rs[best_lag][1]}
    for lag in LAGS:
        row[f'r_lag{lag:+d}'] = lag_rs[lag][0] if lag in lag_rs else np.nan
    ccf_rows.append(row)

ccf_df = pd.DataFrame(ccf_rows)
print(f"CCF 완료 행정동: {len(ccf_df)}")

# ──────────────────────────────────────────────
# 6. 매출 성장률 & 가설 선별
# ──────────────────────────────────────────────
sales_yr = _raw2.groupby(['raw_code', 'yr'])['sales'].sum().reset_index()
s20 = sales_yr[sales_yr['yr'] == 2020].set_index('raw_code')['sales']
s24 = sales_yr[sales_yr['yr'] == 2024].set_index('raw_code')['sales']
growth = ((s24 - s20) / s20 * 100).round(1).rename('sales_growth').reset_index()

ccf_df = ccf_df.merge(growth, on='raw_code', how='left')

cand = ccf_df[
    (ccf_df['peak_lag'] >= 1) &
    (ccf_df['peak_r']   >= R_THRESH) &
    (ccf_df['sales_growth'] > 0)
].sort_values('peak_r', ascending=False).reset_index(drop=True)

print(f"\n가설 부합 행정동 {len(cand)}개:")
for _, r in cand.iterrows():
    print(f"  {r['dong']:10s}  선행={r['peak_lag']}분기  r={r['peak_r']}  매출성장={r['sales_growth']}%")

cand_codes = cand['raw_code'].tolist()

# ──────────────────────────────────────────────
# 7. 세대별 이동인구 분기별
# ──────────────────────────────────────────────
AGE_LABEL = {0:'0대미만',10:'10대',20:'20대',30:'30대',
             40:'40대',50:'50대',60:'60이상',70:'60이상',80:'60이상'}

mig_cand = _mig[_mig['raw_code'].isin(cand_codes)].copy()
mig_cand['age_label'] = mig_cand['age'].map(AGE_LABEL)

mig_age_q = (mig_cand.groupby(['raw_code', 'yrq', 'age_label'])['mig_total']
             .sum().reset_index())
mig_age_q['행정동'] = mig_age_q['raw_code'].map(dong_name)

mig_pivot = mig_age_q.pivot_table(
    index=['행정동', 'yrq'], columns='age_label',
    values='mig_total', aggfunc='sum'
).reset_index()
mig_pivot.columns.name = None
mig_pivot = mig_pivot.rename(columns={'yrq': '연도분기'}).sort_values(['행정동', '연도분기'])

# ──────────────────────────────────────────────
# 8. 세대별 매출 분기별 (절대 + 비율)
# ──────────────────────────────────────────────
raw_cand = _raw2[_raw2['raw_code'].isin(cand_codes)].copy()
raw_cand['행정동'] = raw_cand['raw_code'].map(dong_name)

sales_abs = raw_cand[['행정동', 'yrq', 'sales'] + AGE_SALES].copy()
for c in ['sales'] + AGE_SALES:
    sales_abs[c] = (sales_abs[c] / 1e8).round(2)
sales_abs = sales_abs.rename(columns={
    'yrq': '연도분기', 'sales': '총매출(억)',
    **{k: f'{v}(억)' for k, v in AGE_KR.items()}
}).sort_values(['행정동', '연도분기'])

sales_ratio = raw_cand[['행정동', 'yrq', 'sales'] + AGE_SALES].copy()
for c in AGE_SALES:
    sales_ratio[f'{AGE_KR[c]}_비중(%)'] = (
        sales_ratio[c] / sales_ratio['sales'].replace(0, np.nan) * 100
    ).round(2)
ratio_cols = ['행정동', 'yrq'] + [f'{AGE_KR[c]}_비중(%)' for c in AGE_SALES]
sales_ratio = sales_ratio[ratio_cols].rename(columns={'yrq': '연도분기'}).sort_values(['행정동', '연도분기'])

# ──────────────────────────────────────────────
# 9. 20대 이동 YoY vs 매출 YoY 분기별
# ──────────────────────────────────────────────
yoy_comp = (
    mig20[mig20['raw_code'].isin(cand_codes)][['raw_code','qnum','mig20','mig20_yoy']]
    .merge(sales_q[sales_q['raw_code'].isin(cand_codes)][['raw_code','qnum','sales','sales_yoy']],
           on=['raw_code','qnum'], how='inner')
)
yoy_comp['행정동']        = yoy_comp['raw_code'].map(dong_name)
yoy_comp['yr']           = 2020 + (yoy_comp['qnum'] - 1) // 4
yoy_comp['q_n']          = ((yoy_comp['qnum'] - 1) % 4) + 1
yoy_comp['연도분기']      = yoy_comp['yr'].astype(str) + 'Q' + yoy_comp['q_n'].astype(str)
yoy_comp['20대이동(만명)'] = (yoy_comp['mig20'] / 1e4).round(1)
yoy_comp['매출(억)']       = (yoy_comp['sales']  / 1e8).round(1)
yoy_comp['20대이동_YoY(%)'] = (yoy_comp['mig20_yoy'] * 100).round(1)
yoy_comp['매출_YoY(%)']     = (yoy_comp['sales_yoy']  * 100).round(1)
yoy_out = (yoy_comp[['행정동','연도분기','20대이동(만명)','20대이동_YoY(%)','매출(억)','매출_YoY(%)']]
           .dropna(subset=['20대이동_YoY(%)','매출_YoY(%)'])
           .sort_values(['행정동','연도분기']))

# ──────────────────────────────────────────────
# 10. 요약표
# ──────────────────────────────────────────────
mig20_yr = (_mig[_mig['age'] == 20]
            .groupby(['raw_code', 'yr'])['mig_total'].sum().reset_index())

summary_rows = []
for _, row in cand.iterrows():
    code = row['raw_code']
    m20  = mig20_yr[mig20_yr['raw_code'] == code].set_index('yr')['mig_total'].to_dict()
    s_yr = sales_yr[sales_yr['raw_code'] == code].set_index('yr')['sales'].to_dict()

    sub20  = _raw2[(_raw2['raw_code'] == code) & (_raw2['yr'] == 2020)]
    sub24  = _raw2[(_raw2['raw_code'] == code) & (_raw2['yr'] == 2024)]
    tot20  = sub20['sales'].sum()
    tot24  = sub24['sales'].sum()

    def ratio(sub, tot, c):
        return round(sub[c].sum() / tot * 100, 1) if tot > 0 else np.nan

    rec = {
        '행정동': row['dong'],
        'CCF_선행분기': row['peak_lag'],
        'CCF_최고r': row['peak_r'],
        '매출성장률_2020→2024(%)': row['sales_growth'],
        '총매출_2020(억)': round(s_yr.get(2020, np.nan) / 1e8, 0),
        '총매출_2024(억)': round(s_yr.get(2024, np.nan) / 1e8, 0),
        '20대이동_2020(만명)': round(m20.get(2020, np.nan) / 1e4, 1) if 2020 in m20 else np.nan,
        '20대이동_2024(만명)': round(m20.get(2024, np.nan) / 1e4, 1) if 2024 in m20 else np.nan,
        '20대이동_변화(%)': round((m20[2024] - m20[2020]) / m20[2020] * 100, 1)
                            if (2020 in m20 and 2024 in m20 and m20[2020] > 0) else np.nan,
    }
    for c in AGE_SALES:
        kr = AGE_KR[c]
        rec[f'{kr}매출비중_2020(%)'] = ratio(sub20, tot20, c)
        rec[f'{kr}매출비중_2024(%)'] = ratio(sub24, tot24, c)
    summary_rows.append(rec)

summary = pd.DataFrame(summary_rows)

# CCF 전체 결과 컬럼명 정리
ccf_out = ccf_df.rename(columns={
    'raw_code': '행정동코드', 'dong': '행정동',
    'peak_lag': '최고_선행분기', 'peak_r': '최고_r', 'peak_p': '최고_p',
    'sales_growth': '매출성장률_2020→2024(%)'
})

# ──────────────────────────────────────────────
# 11. 저장
# ──────────────────────────────────────────────
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as w:
    summary.to_excel(w,      sheet_name='①요약_가설부합행정동',  index=False)
    ccf_out.to_excel(w,      sheet_name='②CCF_전체결과',         index=False)
    yoy_out.to_excel(w,      sheet_name='③YoY비교_분기별',        index=False)
    mig_pivot.to_excel(w,    sheet_name='④세대별_이동인구_분기',  index=False)
    sales_abs.to_excel(w,    sheet_name='⑤세대별_매출절대_분기',  index=False)
    sales_ratio.to_excel(w,  sheet_name='⑥세대별_매출비율_분기',  index=False)

print(f"\n[저장완료] {OUT_PATH.name}")
