# -*- coding: utf-8 -*-
"""
트렌드 상권 검증 → 레퍼런스 선정 → 차세대 후보 발굴 파이프라인
복합점수 사용 없음. 4개 독립 신호만 사용.

Signal 1: 소비 업종  — 업종점수 ≥ 0.50
Signal 2: 2030 유입  — 2030비중 ≥ 0.35
Signal 3: 블로그     — 연간 언급량 상위 50% (46개 상권만 커버)
Signal 4: 임대료     — 임대료 YoY 상승 중 (68개 상권만 커버)

확정 트렌드: S1 + S2 충족, S3·S4 중 하나 이상 충족 (데이터 있는 경우)
후보 트렌드: S1 + S2 충족, S3·S4 데이터 없음
레퍼런스:   2020년 미해당 → 2022~2024년 확정/후보로 전환된 곳
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 임계값 ─────────────────────────────────────────────────────
TH_INDUSTRY  = 0.50   # 업종점수
TH_2030      = 0.35   # 2030비중 (소비)
TH_BLOG_PCT  = 50     # 블로그 언급량 퍼센타일 기준 (상위 50%)
MAX_DIV_2024 = 0.60   # 후보: 2024 현재 업종점수 이하 (아직 전환 전)

# ── 1. 데이터 로드 ─────────────────────────────────────────────
tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
blog = pd.read_csv('data/main_data/블로그트렌드_상권.csv', encoding='utf-8-sig')
rental = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
rental_map = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')

ml_tr = pd.read_csv('data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
ml_pr = pd.read_csv('data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
ml = pd.concat([ml_tr, ml_pr], ignore_index=True)
cols = ml.columns.tolist()

# ── 2. 연간 피처 집계 (ML 분기 → 연간 평균) ───────────────────
FEAT_IDX = {
    'FB카페_비중':       6,
    'FB식사_비중':       7,
    '주류유흥_비중':     8,
    '라이프스타일_비중': 9,
    '주말_비중':        18,
    '야간_비중':        25,
    '2030소비_비중':    30,
    '금요효과':         31,
    '평일_비중':        33,
    '트렌드시간대_비중': 34,
    '방문형_강도':      35,
    '2030이동비율':     76,
    '2030이동비율_변화': 83,
}

ml_annual = ml.groupby([cols[1], cols[2]])[
    [cols[i] for i in FEAT_IDX.values()]
].mean().reset_index()
ml_annual.columns = ['행정동', '연도'] + list(FEAT_IDX.keys())

# ── 3. 블로그 신호 매핑 (상권명 → 행정동) ─────────────────────
# 블로그 상권명 ↔ 대표 행정동 매핑 (알려진 것만)
BLOG_TO_DONG = {
    '성수':'성수2가1동', '문래':'문래동', '연남':'연남동', '홍대':'서교동',
    '합정':'합정동',    '이태원':'이태원1동', '서촌':'청운효자동', '북촌':'가회동',
    '대학로':'혜화동',  '한남':'한남동', '망원':'망원동', '뚝섬':'성수1가2동',
    '건대':'화양동',    '잠실':'잠실6동', '을지로':'을지로동', '역삼':'역삼2동',
    '장안':'장안1동',   '연희동':'연희동', '사당':'사당동', '노량진':'노량진2동',
    '방배':'방배2동',   '해방촌':'용산2가동', '청량리':'청량리동', '왕십리':'왕십리2동',
    '흑석':'흑석동',    '상도':'상도1동', '군자':'군자동', '서초':'서초3동',
    '영등포':'영등포동', '코엑스':'삼성1동', '화양':'화양동', '압구정로데오':'신사동',
    '가로수길':'신사동', '청담':'청담동', '익선동':'익선동', '강남역':'역삼1동',
    '경리단길':'이태원2동', '미아':'미아동', '중계':'중계1동', '당산':'당산6동',
    '도화동':'도화동', '면목':'면목본동', '신길':'신길동', '봉천':'봉천동',
    '신정':'신정2동', '화곡':'화곡1동',
}

blog_annual = {}
blog_cols = [str(y) for y in range(2020, 2026) if str(y) in blog.columns]
blog_median = {col: blog[col].median() for col in blog_cols}

for _, row in blog.iterrows():
    dong = BLOG_TO_DONG.get(row['area'])
    if dong is None:
        continue
    for yr_col in blog_cols:
        yr = int(yr_col)
        val = row[yr_col]
        median = blog_median[yr_col]
        blog_annual[(dong, yr)] = 1 if val >= median else 0

# ── 4. 임대료 신호 매핑 ────────────────────────────────────────
rental_signal = {}
rental_map_clean = rental_map.rename(columns={rental_map.columns[0]:'상권명',
                                               rental_map.columns[1]:'대표_행정동'})

for _, mrow in rental_map_clean.iterrows():
    dong = mrow['대표_행정동']
    sang = mrow['상권명']
    r_sub = rental[rental['상권명'] == sang].sort_values('연도')
    if len(r_sub) < 2:
        continue
    r_sub = r_sub.set_index('연도')
    for yr in range(2021, 2025):
        if yr in r_sub.index and (yr-1) in r_sub.index:
            prev = r_sub.loc[yr-1, '임대료']
            curr = r_sub.loc[yr, '임대료']
            if pd.notna(prev) and pd.notna(curr) and prev > 0:
                rental_signal[(dong, yr)] = 1 if curr > prev else 0

# ── 5. 검증 테이블 생성 ────────────────────────────────────────
rows = []
for _, t in tsm.iterrows():
    dong = t['행정동']
    yr   = int(t['연도'])

    s1 = int(t['업종점수'] >= TH_INDUSTRY)
    s2 = int(t['2030비중'] >= TH_2030)
    s3 = blog_annual.get((dong, yr), None)
    s4 = rental_signal.get((dong, yr), None)

    # 확정/후보 분류
    if s1 and s2:
        if (s3 == 1) or (s4 == 1):
            status = '확정'
        elif s3 is None and s4 is None:
            status = '후보'
        else:
            status = '후보'  # 데이터 있지만 신호 약함도 후보 유지
    else:
        status = '미해당'

    rows.append({
        '행정동': dong, '연도': yr,
        'S1_업종': s1, 'S2_2030소비': s2,
        'S3_블로그': s3, 'S4_임대료': s4,
        '상태': status,
        '업종점수': t['업종점수'], '2030비중': t['2030비중'],
    })

val_df = pd.DataFrame(rows)

# ── 6. 레퍼런스 선정 — 전환 행정동 ────────────────────────────
# 조건: 2020년 미해당 → 2022·2023·2024 중 하나 이상 확정/후보
status_2020 = val_df[val_df['연도']==2020].set_index('행정동')['상태']
status_late = val_df[val_df['연도'].isin([2022,2023,2024])].groupby('행정동')['상태'].apply(
    lambda x: '확정' if '확정' in x.values else ('후보' if '후보' in x.values else '미해당')
)

transition = []
for dong in status_2020.index:
    if dong not in status_late.index:
        continue
    was_not = status_2020[dong] == '미해당'
    became  = status_late[dong] in ['확정','후보']
    if was_not and became:
        transition.append({
            '행정동': dong,
            '2020_상태': status_2020[dong],
            '전환_상태': status_late[dong],
        })

ref_df = pd.DataFrame(transition)

# 블로그/임대료 외부 검증 있는 곳 우선 정렬
def has_external(dong):
    return any(blog_annual.get((dong, yr), 0) == 1 for yr in range(2020,2025)) or \
           any(rental_signal.get((dong, yr), 0) == 1 for yr in range(2020,2025))

ref_df['외부검증'] = ref_df['행정동'].apply(has_external)
ref_df_confirmed = ref_df[ref_df['전환_상태']=='확정'].sort_values('외부검증', ascending=False)
ref_df_candidate = ref_df[ref_df['전환_상태']=='후보']

print(f'전환 레퍼런스: 확정 {len(ref_df_confirmed)}개, 후보 {len(ref_df_candidate)}개')

# 외부 검증 있는 "확정" 레퍼런스 우선 선택, 부족하면 후보로 보완
top_refs = ref_df_confirmed['행정동'].tolist()
if len(top_refs) < 5:
    top_refs += ref_df_candidate[~ref_df_candidate['행정동'].isin(top_refs)]['행정동'].tolist()
top_refs = top_refs[:8]  # 최대 8개

print(f'최종 레퍼런스 ({len(top_refs)}개): {top_refs}')

# ── 7. 레퍼런스 전환 시점 파악 ─────────────────────────────────
def get_transition_year(dong):
    sub = val_df[val_df['행정동']==dong].sort_values('연도')
    for _, r in sub.iterrows():
        if r['상태'] in ['확정','후보'] and r['연도'] >= 2021:
            return int(r['연도'])
    return None

ref_transition = {d: get_transition_year(d) for d in top_refs}
print('레퍼런스 전환 연도:', ref_transition)

# ── 8. 레퍼런스 시그니처 추출 (전환 1~2년 전 패턴) ────────────
sig_feats = list(FEAT_IDX.keys())

sig_vectors = []
for dong in top_refs:
    ty = ref_transition.get(dong)
    if ty is None:
        continue
    pre_yr = max(ty - 2, 2020)
    sub = ml_annual[(ml_annual['행정동']==dong) & (ml_annual['연도'].isin([pre_yr, pre_yr+1]))]
    if len(sub) == 0:
        continue
    vec = sub[sig_feats].mean().values
    sig_vectors.append(vec)

if len(sig_vectors) == 0:
    print('레퍼런스 벡터 없음 - 조건 완화 필요')
    exit(1)

sig_matrix   = np.array(sig_vectors)
# z-score 정규화 (전체 2024 기준)
ml_2024 = ml_annual[ml_annual['연도']==2024][sig_feats]
feat_mean = ml_2024.mean()
feat_std  = ml_2024.std().replace(0, 1)
sig_znorm = (sig_matrix - feat_mean.values) / feat_std.values
sig_znorm = np.nan_to_num(sig_znorm, nan=0.0)
ref_signature = sig_znorm.mean(axis=0)

# ── 9. 후보 매칭 (2024년 기준, 아직 전환 안 된 곳) ────────────
not_yet = val_df[(val_df['연도']==2024) & (val_df['상태']=='미해당')]['행정동'].tolist()
not_yet = [d for d in not_yet if d not in top_refs]

cand_vecs  = []
cand_names = []
for dong in not_yet:
    sub = ml_annual[(ml_annual['행정동']==dong) & (ml_annual['연도']==2024)]
    if len(sub) == 0:
        continue
    vec = sub[sig_feats].values[0]
    zvec = (vec - feat_mean.values) / feat_std.values
    zvec = np.nan_to_num(zvec, nan=0.0)
    cand_vecs.append(zvec)
    cand_names.append(dong)

if len(cand_vecs) == 0:
    print('후보 없음')
    exit(1)

cand_matrix = np.array(cand_vecs)
sims = cosine_similarity(cand_matrix, ref_signature.reshape(1,-1)).flatten()

result = pd.DataFrame({'행정동': cand_names, '유사도': sims})
result = result.sort_values('유사도', ascending=False).reset_index(drop=True)
result['순위'] = result.index + 1

# 후보에 2030이동비율, 업종점수, 2030비중 추가
ml_2024_sub = ml_annual[ml_annual['연도']==2024][['행정동'] + sig_feats]
result = result.merge(ml_2024_sub, on='행정동', how='left')
val_2024 = val_df[val_df['연도']==2024][['행정동','업종점수','2030비중']]
result = result.merge(val_2024, on='행정동', how='left')

top25 = result.head(25)
print('\n=== 차세대 후보 TOP 25 ===')
with open('data/main_data/trend_candidates_v2.csv','w',encoding='utf-8-sig') as f:
    top25.to_csv(f, index=False)
print(top25[['순위','행정동','유사도','업종점수','2030비중','2030이동비율']].head(15).to_string(index=False))

# ── 10. 시각화 ────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#FAFAFA')
gs  = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.38,
                        left=0.06, right=0.97, top=0.93, bottom=0.05)
fig.suptitle('트렌드 상권 검증 → 레퍼런스 선정 → 차세대 후보 발굴\n(복합점수 미사용 · 4개 독립 신호 기반)',
             fontsize=14, fontweight='bold')

COLORS = plt.cm.tab10(np.linspace(0,1,10))

# ─ P1: 연도별 검증 상태 흐름 (확정·후보·미해당 누적 스택) ─────
ax1 = fig.add_subplot(gs[0, 0])
yr_list = [2020,2021,2022,2023,2024]
cnf_cnt = [val_df[(val_df['연도']==y)&(val_df['상태']=='확정')].shape[0] for y in yr_list]
can_cnt = [val_df[(val_df['연도']==y)&(val_df['상태']=='후보')].shape[0] for y in yr_list]
not_cnt = [val_df[(val_df['연도']==y)&(val_df['상태']=='미해당')].shape[0] for y in yr_list]

ax1.bar(yr_list, cnf_cnt, label='확정 트렌드', color='#E63946', alpha=0.85)
ax1.bar(yr_list, can_cnt, bottom=cnf_cnt, label='후보 트렌드', color='#457B9D', alpha=0.75)
ax1.set_title('① 연도별 트렌드 상권 확인 수\n(S1업종 ≥0.50 AND S2 2030소비 ≥0.35)', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylabel('행정동 수 (전체 423개)')
ax1.set_xticks(yr_list)
ax1.grid(axis='y', alpha=0.3)
for i, (c,ca) in enumerate(zip(cnf_cnt, can_cnt)):
    ax1.text(yr_list[i], c + ca + 2, f'{c+ca}', ha='center', fontsize=8.5, fontweight='bold')

# ─ P2: 레퍼런스 시그니처 바 ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
feat_labels = [
    '2030이동↑변화','평일비중','라이프스타일',
    '트렌드시간대','방문형_강도','주말비중','야간비중',
    '2030소비비중','금요효과','FB카페','FB식사','주류유흥','2030이동'
]
sorted_idx = np.argsort(ref_signature)[::-1]
sorted_vals = ref_signature[sorted_idx]
sorted_names = [list(FEAT_IDX.keys())[i].replace('_비중','').replace('_','·') for i in sorted_idx]
colors_sig = ['#E63946' if v >= 0 else '#6c757d' for v in sorted_vals]

bars2 = ax2.barh(sorted_names, sorted_vals, color=colors_sig, alpha=0.85, height=0.7)
ax2.axvline(0, color='black', lw=0.8)
ax2.set_title(f'② 레퍼런스 시그니처\n(전환 전 공통 패턴, {len(top_refs)}개 참조)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Z-score')
ax2.grid(axis='x', alpha=0.3)
for bar, val in zip(bars2, sorted_vals):
    ax2.text(val + (0.02 if val>=0 else -0.02),
             bar.get_y()+bar.get_height()/2, f'{val:+.2f}',
             va='center', ha='left' if val>=0 else 'right', fontsize=7.5)

# ─ P3: 레퍼런스 동 업종점수 × 2030비중 궤적 ──────────────────
ax3 = fig.add_subplot(gs[0, 2])
for i, dong in enumerate(top_refs):
    d = tsm[tsm['행정동']==dong].sort_values('연도')
    if len(d) == 0: continue
    sc = ax3.scatter(d['업종점수'], d['2030비중'],
                     c=d['연도'], cmap='YlOrRd', vmin=2020, vmax=2024,
                     s=60, zorder=5, alpha=0.9)
    ax3.plot(d['업종점수'], d['2030비중'], color=COLORS[i%10], lw=1.2, alpha=0.5)
    # 마지막 점에 이름
    ax3.annotate(dong, xy=(d['업종점수'].iloc[-1], d['2030비중'].iloc[-1]),
                 xytext=(4,0), textcoords='offset points', fontsize=7.5, color=COLORS[i%10])

ax3.axvline(TH_INDUSTRY, color='#E63946', lw=1, ls='--', alpha=0.7, label=f'업종 ≥{TH_INDUSTRY}')
ax3.axhline(TH_2030, color='#457B9D', lw=1, ls='--', alpha=0.7, label=f'2030소비 ≥{TH_2030}')
ax3.set_xlabel('업종점수', fontsize=9)
ax3.set_ylabel('2030소비 비중', fontsize=9)
ax3.set_title('③ 레퍼런스 동 전환 궤적\n(업종↗ × 2030비중↗ 동시 상승)', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ─ P4 (bottom-left): 후보 TOP 20 유사도 ─────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
top20 = top25.head(20)
bar_colors = [plt.cm.RdYlGn(s) for s in top20['유사도']]
bars4 = ax4.barh(top20['행정동'][::-1], top20['유사도'][::-1],
                 color=bar_colors[::-1], alpha=0.88, height=0.7)
ax4.set_xlabel('레퍼런스 패턴 코사인 유사도')
ax4.set_title('④ 차세대 후보 TOP 20 (패턴 유사도 순)\n — 복합점수 없이 소비업종·2030유입·이동 신호만으로 도출',
              fontsize=10, fontweight='bold')
ax4.axvline(0.75, color='#888', lw=1, ls='--', alpha=0.6, label='유사도 0.75 기준선')
ax4.legend(fontsize=8)
ax4.grid(axis='x', alpha=0.3)
for bar, (_, row) in zip(bars4, top20.iloc[::-1].iterrows()):
    label = f'{row["유사도"]:.3f}  |  업종:{row["업종점수"]:.2f}  2030:{row["2030비중"]:.2f}'
    ax4.text(row['유사도']+0.003, bar.get_y()+bar.get_height()/2,
             label, va='center', fontsize=7.5)

# ─ P5: 후보 업종점수 × 2030비중 산점도 ────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
top15 = top25.head(15)
sc5 = ax5.scatter(top15['업종점수'], top15['2030비중'],
                  c=top15['유사도'], cmap='YlOrRd', s=100, zorder=5, vmin=0.5, vmax=1.0)
for _, row in top15.iterrows():
    ax5.annotate(row['행정동'], xy=(row['업종점수'], row['2030비중']),
                 xytext=(3,3), textcoords='offset points', fontsize=7.5)

ax5.axvline(TH_INDUSTRY, color='#E63946', lw=1, ls='--', alpha=0.6, label='업종 0.50 기준')
ax5.axhline(TH_2030, color='#457B9D', lw=1, ls='--', alpha=0.6, label='2030소비 0.35 기준')
plt.colorbar(sc5, ax=ax5, label='유사도')
ax5.set_xlabel('업종점수', fontsize=9)
ax5.set_ylabel('2030소비 비중', fontsize=9)
ax5.set_title('⑤ 후보 현재 위치\n(왼쪽 하단 = 아직 전환 전)', fontsize=10, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# ─ P6: 시그널별 히트맵 — 확정 트렌드 상권 목록 ─────────────────
ax6 = fig.add_subplot(gs[2, :])

confirmed_list = val_df[val_df['상태']=='확정'].sort_values(['연도','2030비중'], ascending=[True,False])
confirmed_2024 = confirmed_list[confirmed_list['연도']==2024].head(25)
conf_dongs = confirmed_2024['행정동'].tolist()

# 히트맵 데이터
mat_rows = ['S1·업종점수', 'S2·2030소비비중', 'S3·블로그', 'S4·임대료']
mat = np.full((4, len(conf_dongs)), np.nan)

for ci, dong in enumerate(conf_dongs):
    r2024 = confirmed_2024[confirmed_2024['행정동']==dong].iloc[0]
    mat[0, ci] = min(r2024['업종점수'] / 0.90, 1.0)
    mat[1, ci] = min(r2024['2030비중'] / 0.70, 1.0)
    b = blog_annual.get((dong, 2024))
    mat[2, ci] = float(b) if b is not None else np.nan
    r = rental_signal.get((dong, 2024))
    mat[3, ci] = float(r) if r is not None else np.nan

im = ax6.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax6.set_xticks(range(len(conf_dongs)))
ax6.set_xticklabels(conf_dongs, rotation=40, ha='right', fontsize=8)
ax6.set_yticks(range(4))
ax6.set_yticklabels(mat_rows, fontsize=9)
ax6.set_title('⑥ 2024년 확정 트렌드 상권 — 신호별 강도\n(초록=강한 신호 · 빨강=약함 · 회색=데이터 없음)',
              fontsize=10, fontweight='bold')
for r in range(4):
    for c in range(len(conf_dongs)):
        if not np.isnan(mat[r,c]):
            ax6.text(c, r, f'{mat[r,c]:.2f}', ha='center', va='center',
                     fontsize=7, color='white' if mat[r,c] < 0.35 or mat[r,c] > 0.75 else 'black')
        else:
            ax6.text(c, r, '—', ha='center', va='center', fontsize=8, color='#bbb')

plt.colorbar(im, ax=ax6, orientation='vertical', fraction=0.01, pad=0.01)
plt.savefig('image/trend_validation_pipeline.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('\n저장: image/trend_validation_pipeline.png')

# ── 요약 출력 ─────────────────────────────────────────────────
print('\n=== 확정 트렌드 상권 (2024, 업종+2030+외부신호) ===')
with open('data/temp_confirmed_2024.txt','w',encoding='utf-8') as f:
    out = val_df[(val_df['연도']==2024) & (val_df['상태']=='확정')].sort_values('2030비중',ascending=False)
    f.write(out[['행정동','업종점수','2030비중','S3_블로그','S4_임대료']].to_string())
print(f'  {len(out)}개 확정')

print('\n=== 레퍼런스 동 (2020→2022+ 전환) ===')
for d in top_refs:
    ty = ref_transition.get(d, '?')
    ext = '외부검증O' if has_external(d) else '외부검증X'
    v20 = val_df[(val_df['행정동']==d)&(val_df['연도']==2020)].iloc[0] if len(val_df[(val_df['행정동']==d)&(val_df['연도']==2020)])>0 else None
    v24 = val_df[(val_df['행정동']==d)&(val_df['연도']==2024)].iloc[0] if len(val_df[(val_df['행정동']==d)&(val_df['연도']==2024)])>0 else None
    u20 = f'{v20["업종점수"]:.2f}' if v20 is not None else '-'
    u24 = f'{v24["업종점수"]:.2f}' if v24 is not None else '-'
    b20 = f'{v20["2030비중"]:.2f}' if v20 is not None else '-'
    b24 = f'{v24["2030비중"]:.2f}' if v24 is not None else '-'
    print(f'  {d}: 전환년도={ty} {ext} | 업종:{u20}→{u24} | 2030:{b20}→{b24}')
