import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────────────
df_idx  = pd.read_csv('data/행정동별 상권발달 개별지수.csv', encoding='cp949')
df_dong = pd.read_csv('data/행정동코드_2014.csv', encoding='utf-8-sig')
df_trend = pd.read_csv('data/naver_trend_행정동_보정_20260510.csv', encoding='utf-8-sig')

# ── 행정동코드 → 읍면동명 매핑 ───────────────────────────────
df_dong['join_key'] = df_dong['행정기관코드'].astype(str)
df_idx['join_key']  = df_idx['행정동코드(ADSTRD_CD)'].astype(str) + '00'
df_idx = df_idx.merge(df_dong[['join_key','시군구명','읍면동명']], on='join_key', how='left')

# ── 읍면동명 → 네이버 트렌드 그룹명 매핑 ────────────────────
dong_to_group = {
    '화곡제6동':'화곡', '화곡제8동':'화곡',
    '당산제2동':'당산', '난향동':'난곡',
    '중계본동':'중계', '연희동':'연희동',
    '교남동':'독립문', '가락2동':'가락', '가락제2동':'가락',
    '방화제2동':'방화', '후암동':'후암동',
    '을지로동':'을지로', '제기동':'청량리',
    '청구동':'청구', '도봉제1동':'도봉',
    '신월3동':'신월', '신길제4동':'신길',
    '혜화동':'대학로', '왕십리도선동':'왕십리',
    '개포2동':'개포', '개포제2동':'개포',
    '회기동':'회기', '명일제1동':'명일', '명일제2동':'명일',
    '노량진제1동':'노량진', '금호1가동':'금호',
    '보라매동':'보라매', '둔촌제1동':'둔촌',
}
df_idx['group'] = df_idx['읍면동명'].map(dong_to_group)

# ── Step 1: 행정동별 상권발달 지수 평균 집계 ─────────────────
sales_agg = (
    df_idx[df_idx['group'].notna()]
    .groupby('group')[['매출지수(SALES)','인프라지수(INFRASTRUCTURE)',
                        '가맹점지수(STORE)','인구지수(POPULATION)','금융지수(DEPOSIT)']]
    .mean().round(2).reset_index()
    .rename(columns={
        '매출지수(SALES)':          'avg_매출지수',
        '인프라지수(INFRASTRUCTURE)':'avg_인프라지수',
        '가맹점지수(STORE)':         'avg_가맹점지수',
        '인구지수(POPULATION)':      'avg_인구지수',
        '금융지수(DEPOSIT)':         'avg_금융지수',
    })
)

# ── Step 1: 네이버 트렌드 평균 집계 ──────────────────────────
trend_agg = (
    df_trend.groupby('group')['adjusted_ratio']
    .mean().round(2).reset_index()
    .rename(columns={'adjusted_ratio': 'avg_검색량'})
)

# ── Step 1: 합치기 ────────────────────────────────────────────
df_merged = sales_agg.merge(trend_agg, on='group')
df_merged = df_merged.sort_values('avg_매출지수', ascending=False).reset_index(drop=True)

output_path = 'data/행정동_통합분석.csv'
df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"[Step 1] 통합 파일 저장 완료: {output_path}")
print(df_merged.to_string(index=False))

# ── Step 2: 시각화 ────────────────────────────────────────────
corr, pval = stats.spearmanr(df_merged['avg_매출지수'], df_merged['avg_검색량'])
print(f"\nSpearman r={corr:.3f}  p={pval:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('서울 행정동 상권발달지수 vs 네이버 검색 트렌드', fontsize=15, fontweight='bold')

# ── 패널1: 산점도 ─────────────────────────────────────────────
ax = axes[0][0]
ax.scatter(df_merged['avg_매출지수'], df_merged['avg_검색량'], s=90, color='steelblue', alpha=0.75, zorder=3)
for _, row in df_merged.iterrows():
    ax.annotate(row['group'], (row['avg_매출지수'], row['avg_검색량']),
                fontsize=8, xytext=(4, 4), textcoords='offset points')
m, b = np.polyfit(df_merged['avg_매출지수'], df_merged['avg_검색량'], 1)
xs = np.linspace(df_merged['avg_매출지수'].min(), df_merged['avg_매출지수'].max(), 100)
ax.plot(xs, m*xs+b, '--', color='tomato', linewidth=1.5)
ax.axhline(100, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('평균 매출지수 (2018~2019)', fontsize=10)
ax.set_ylabel('평균 보정 검색량 (서울=100, 2022~2024)', fontsize=10)
ax.set_title(f'상관관계  Spearman r={corr:.3f}  p={pval:.4f}', fontsize=11)
ax.grid(True, alpha=0.3)

# ── 패널2: 매출지수 순위 막대 ────────────────────────────────
ax = axes[0][1]
df_s = df_merged.sort_values('avg_매출지수')
colors = ['#d62728' if v >= 50 else '#1f77b4' for v in df_s['avg_매출지수']]
ax.barh(df_s['group'], df_s['avg_매출지수'], color=colors, alpha=0.82)
ax.set_xlabel('평균 매출지수', fontsize=10)
ax.set_title('행정동별 평균 매출지수 (빨강≥50)', fontsize=11)
ax.grid(True, axis='x', alpha=0.3)

# ── 패널3: 검색량 순위 막대 ──────────────────────────────────
ax = axes[1][0]
df_t = df_merged.sort_values('avg_검색량')
colors = ['#d62728' if v >= 100 else '#2ca02c' for v in df_t['avg_검색량']]
ax.barh(df_t['group'], df_t['avg_검색량'], color=colors, alpha=0.82)
ax.axvline(100, color='gray', linestyle='--', linewidth=1, label='서울=100')
ax.set_xlabel('평균 보정 검색량', fontsize=10)
ax.set_title('행정동별 평균 보정 검색량 (빨강=서울 초과)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='x', alpha=0.3)

# ── 패널4: 5개 지수 레이더 (상위 5개 행정동) ─────────────────
ax = axes[1][1]
top5 = df_merged.nlargest(5, 'avg_검색량')
metrics = ['avg_매출지수','avg_인프라지수','avg_가맹점지수','avg_인구지수','avg_금융지수']
labels  = ['매출','인프라','가맹점','인구','금융']
x = np.arange(len(metrics))
width = 0.15
for i, (_, row) in enumerate(top5.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=row['group'], alpha=0.8)
ax.set_xticks(x + width*2)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('지수 값', fontsize=10)
ax.set_title('검색량 상위 5개 행정동 — 상권발달 5개 지수 비교', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_img = 'data/행정동_통합분석.png'
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print(f"\n[Step 2] 시각화 저장 완료: {out_img}")
plt.show()
