"""
구역 유형별 매출 구성 특성 분석
- 오피스형 / 혼합형 / 주거형 / 상권형 + 서울 전체 평균
- 0.99 clipping 적용 (비율 지표 이상치 제거)
- 2020~2024년 전체 데이터 사용 (구역유형 분류는 2024년 생활인구 기준 고정)
- 행정동별 연평균 비율로 집계 (절대 매출 규모 차이 중립화)
"""
import pandas as pd
import numpy as np

# ── 데이터 로드 ──────────────────────────────────────────────
area = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
sales = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
raw = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

# ── 2020~2024 전체 연간 집계 (raw → 행정동×연도별 합산) ─────
agg_cols = [
    'FB카페_매출금액', 'FB식사_매출금액', '주류유흥_매출금액', '라이프스타일_매출금액', '기타_매출금액', '총_매출금액',
    '월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액', '목요일_매출_금액',
    '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액',
    '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
    '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
    '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
    '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
    '남성_매출_금액', '여성_매출_금액',
    '주중_매출_금액', '주말_매출_금액',
]
# 분기 → 행정동×연도 합산
raw_yearly = raw.groupby(['행정동', '연도'])[agg_cols].sum().reset_index()
# 연도 내 비율 산출 후 행정동별 연평균으로 집계 (절대 매출 규모 영향 제거)
raw24_agg = raw_yearly  # 아래에서 비율 계산 후 연평균 집계

# ── 비율 지표 산출 (행정동×연도별) ───────────────────────────
def safe_div(a, b):
    return np.where(b > 0, a / b, np.nan)

df = raw_yearly.copy()
total = df['총_매출금액']

# 업종 비중
df['FB카페_비중']      = safe_div(df['FB카페_매출금액'],      total)
df['FB식사_비중']      = safe_div(df['FB식사_매출금액'],      total)
df['주류유흥_비중']    = safe_div(df['주류유흥_매출금액'],    total)
df['라이프스타일_비중'] = safe_div(df['라이프스타일_매출금액'], total)
df['기타업종_비중']    = safe_div(df['기타_매출금액'],        total)
df['업종점수']         = df['FB카페_비중'] + df['FB식사_비중'] + df['주류유흥_비중'] + df['라이프스타일_비중']

# 요일 비중
for day in ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']:
    df[f'{day}_비중'] = safe_div(df[f'{day}_매출_금액'], total)

df['주말비중']  = safe_div(df['토요일_매출_금액'] + df['일요일_매출_금액'],
                          df[['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
                              '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']].sum(axis=1))
df['금요효과']  = safe_div(df['금요일_매출_금액'], df['목요일_매출_금액'])
df['평일비중']  = safe_div(df['주중_매출_금액'], total)

# 시간대 비중
for t in ['00~06', '06~11', '11~14', '14~17', '17~21', '21~24']:
    df[f'시간대_{t}_비중'] = safe_div(df[f'시간대_{t}_매출_금액'], total)

df['야간비중'] = df['시간대_17~21_비중'] + df['시간대_21~24_비중']
df['주간비중'] = df['시간대_06~11_비중'] + df['시간대_11~14_비중'] + df['시간대_14~17_비중']

# 연령대 비중
for age in ['10', '20', '30', '40', '50', '60_이상']:
    df[f'연령{age}_비중'] = safe_div(df[f'연령대_{age}_매출_금액'], total)

df['2030비중'] = df['연령20_비중'] + df['연령30_비중']
df['20대비중'] = df['연령20_비중']

# 성별
df['남성비중'] = safe_div(df['남성_매출_금액'], total)
df['여성비중'] = safe_div(df['여성_매출_금액'], total)

# ── 0.99 clipping (비율 지표만, 전체 행 기준) ──────────────
ratio_cols = [c for c in df.columns if '비중' in c or c in ['업종점수', '금요효과']]

clip_upper = {}
for col in ratio_cols:
    q99 = df[col].quantile(0.99)
    clip_upper[col] = q99
    df[col] = df[col].clip(upper=q99)

# ── area_type_profile 조인 (구역유형은 2024년 분류 고정) ────
area_slim = area[['행정동', '주말집중도', '2030주말집중도', '구역유형']].copy()
df = df.merge(area_slim, on='행정동', how='left')

# trend_sales_master 조인 (복합점수_분위·트렌드매출_비중 — 연도별)
sales_slim = sales[['행정동', '연도', '복합점수_분위', '트렌드매출_비중']].copy()
df = df.merge(sales_slim, on=['행정동', '연도'], how='left')

# ── 행정동별 연평균으로 최종 집계 ─────────────────────────
# (각 행정동의 5년 평균 → 구역유형별 평균)
# 주말집중도는 고정값(2024 기준)이므로 first로 가져옴
avg_cols = ratio_cols + ['복합점수_분위', '트렌드매출_비중']
df_dong_avg = df.groupby(['행정동', '구역유형'])[avg_cols].mean().reset_index()

# 주말집중도 별도 조인 (연도와 무관하게 고정값)
area_concentr = area[['행정동', '주말집중도', '2030주말집중도']].copy()
df_dong_avg = df_dong_avg.merge(area_concentr, on='행정동', how='left')

# ── 분석 대상 지표 ─────────────────────────────────────────
analysis_metrics = {
    # 업종 구성
    'FB카페_비중':      'FB카페 비중',
    'FB식사_비중':      'FB식사 비중',
    '주류유흥_비중':    '주류·유흥 비중',
    '라이프스타일_비중': '라이프스타일 비중',
    '업종점수':         '업종점수 (트렌드 4개 합)',
    '기타업종_비중':    '기타업종 비중',
    # 요일 패턴
    '금요효과':         '금요효과 (금/목)',
    '주말비중':         '주말 매출 비중 (토+일/7일)',
    '평일비중':         '평일 매출 비중',
    # 시간대 패턴
    '야간비중':         '야간 비중 (17~24시)',
    '주간비중':         '주간 비중 (06~17시)',
    # 연령대
    '20대비중':         '20대 매출 비중',
    '2030비중':         '2030 매출 비중',
    '연령40_비중':      '40대 매출 비중',
    '연령50_비중':      '50대 매출 비중',
    # 성별
    '남성비중':         '남성 매출 비중',
    '여성비중':         '여성 매출 비중',
    # 생활인구
    '주말집중도':       '주말집중도 (주말/평일 생활인구)',
    '2030주말집중도':   '2030 주말집중도',
    # 종합
    '트렌드매출_비중':  '트렌드 대표 매출 비중',
    '복합점수_분위':    '복합점수 분위 (0~1)',
}

# ── 유형별 집계 (행정동 연평균 기준) ────────────────────────
tier_order = ['오피스형', '혼합형', '주거형', '상권형']
result_rows = []

# 서울 전체 (구역유형 있는 행만)
seoul_df = df_dong_avg.dropna(subset=['구역유형'])
row_seoul = {'구역유형': '서울 전체', '행정동 수': len(seoul_df)}
for col in analysis_metrics:
    if col in seoul_df.columns:
        row_seoul[col] = seoul_df[col].mean()
result_rows.append(row_seoul)

# 유형별
for tier in tier_order:
    sub = df_dong_avg[df_dong_avg['구역유형'] == tier]
    row = {'구역유형': tier, '행정동 수': len(sub)}
    for col in analysis_metrics:
        if col in sub.columns:
            row[col] = sub[col].mean()
    result_rows.append(row)

result = pd.DataFrame(result_rows)

# ── 출력 ──────────────────────────────────────────────────
with open('_tmp_area_profile.txt', 'w', encoding='utf-8') as f:

    f.write('# 구역 유형별 매출 구성 특성 분석 (2020~2024 연평균, 0.99 clipping 적용)\n')
    f.write('# 집계 방식: 행정동별 5년 연평균 비율 → 구역유형별 평균 (절대 매출 규모 중립화)\n')
    f.write('# 구역유형 분류 기준: 2024년 생활인구 주말집중도\n\n')
    f.write(f'행정동 수: {df_dong_avg.dropna(subset=["구역유형"]).groupby("구역유형").size().to_dict()}\n\n')

    # 섹션별 출력
    sections = {
        '업종 구성': ['FB카페_비중', 'FB식사_비중', '주류유흥_비중', '라이프스타일_비중', '업종점수', '기타업종_비중'],
        '요일 패턴': ['금요효과', '주말비중', '평일비중'],
        '시간대 패턴': ['야간비중', '주간비중'],
        '연령대': ['20대비중', '2030비중', '연령40_비중', '연령50_비중'],
        '성별': ['남성비중', '여성비중'],
        '생활인구': ['주말집중도', '2030주말집중도'],
        '종합 지표': ['트렌드매출_비중', '복합점수_분위'],
    }

    cols_display = ['구역유형', '행정동 수']

    for section, cols in sections.items():
        f.write(f'\n## {section}\n\n')
        disp = result[cols_display + cols].copy()

        # 퍼센트 포맷 (비율 0~1 → %)
        pct_cols = [c for c in cols if c not in ['금요효과', '주말집중도', '2030주말집중도', '복합점수_분위']]
        for c in pct_cols:
            disp[c] = disp[c].map(lambda x: f'{x*100:.2f}%' if pd.notnull(x) else '-')

        # 나머지 포맷
        for c in cols:
            if c == '금요효과':
                disp[c] = disp[c].map(lambda x: f'{x:.4f}' if pd.notnull(x) else '-')
            elif c in ['주말집중도', '2030주말집중도']:
                disp[c] = disp[c].map(lambda x: f'{x:.4f}' if pd.notnull(x) else '-')
            elif c == '복합점수_분위':
                disp[c] = disp[c].map(lambda x: f'{x:.4f}' if pd.notnull(x) else '-')

        disp.columns = cols_display + [analysis_metrics[c] for c in cols]
        f.write(disp.to_string(index=False) + '\n')

    f.write('\n\n## 0.99 Clipping 상한값\n\n')
    for col, val in clip_upper.items():
        f.write(f'  {col}: {val:.4f}\n')

print('분석 완료 → _tmp_area_profile.txt')
