# -*- coding: utf-8 -*-
"""
레퍼런스 16개 동 정보 표 생성 (v6 §3.2.1 / §3.2.2)
출력:
  report/data/reference_3_2_1.csv  — 기본 정보표 (§3.2.1)
  report/data/reference_3_2_2.csv  — 핵심 수치표 (§3.2.2)
  report/data/reference_signal_detail.csv — 연도별 신호 상세
"""

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import linregress

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs('report/data', exist_ok=True)

TH_INDUSTRY = 0.50
TH_2030     = 0.35

# ────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────
tsm        = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
blog       = pd.read_csv('data/main_data/블로그트렌드_상권.csv', encoding='utf-8-sig')
rental     = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
rental_map = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')
area_type  = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
mig_raw    = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
kt         = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
kt_map     = dict(zip(kt['행정동코드'], kt['행정동명']))

# 자치구 매핑
gu_map = dict(zip(tsm.groupby('행정동').first().reset_index()['행정동'],
                  [None]*len(tsm['행정동'].unique())))
# 행정동코드 → 자치구
# 자치구: KT 시군구 코드 → 이름 변환
GU_CODE = {
    11010:'종로구',11020:'중구',11030:'용산구',11040:'성동구',11050:'광진구',
    11060:'동대문구',11070:'중랑구',11080:'성북구',11090:'강북구',11100:'도봉구',
    11110:'노원구',11120:'은평구',11130:'서대문구',11140:'마포구',11150:'양천구',
    11160:'강서구',11170:'구로구',11180:'금천구',11190:'영등포구',11200:'동작구',
    11210:'관악구',11220:'서초구',11230:'강남구',11240:'송파구',11250:'강동구',
}
kt_seoul = kt[kt['시도'] == 11000]  # 서울만
dong_gu_map = {row['행정동명']: GU_CODE.get(int(row['시군구']), str(row['시군구']))
               for _, row in kt_seoul.iterrows()}

# ────────────────────────────────────────────────────────────────
# 블로그 / 임대료 신호 재구성
# ────────────────────────────────────────────────────────────────
BLOG_TO_DONG = {
    '성수':'성수2가1동','문래':'문래동','연남':'연남동','홍대':'서교동',
    '합정':'합정동','이태원':'이태원1동','서촌':'청운효자동','북촌':'가회동',
    '대학로':'혜화동','한남':'한남동','망원':'망원동','뚝섬':'성수1가2동',
    '건대':'화양동','잠실':'잠실6동','을지로':'을지로동','역삼':'역삼2동',
    '장안':'장안1동','연희동':'연희동','사당':'사당동','노량진':'노량진2동',
    '방배':'방배2동','해방촌':'용산2가동','청량리':'청량리동','왕십리':'왕십리2동',
    '흑석':'흑석동','상도':'상도1동','군자':'군자동','서초':'서초3동',
    '영등포':'영등포동','코엑스':'삼성1동','화양':'화양동','압구정로데오':'신사동',
    '가로수길':'신사동','청담':'청담동','익선동':'익선동','강남역':'역삼1동',
    '경리단길':'이태원2동','미아':'미아동','중계':'중계1동','당산':'당산6동',
}
blog_cols   = [str(y) for y in range(2020, 2025) if str(y) in blog.columns]
blog_median = {c: blog[c].median() for c in blog_cols}
blog_annual = {}
for _, row in blog.iterrows():
    dong = BLOG_TO_DONG.get(row['area'])
    if dong:
        for yc in blog_cols:
            blog_annual[(dong, int(yc))] = 1 if row[yc] >= blog_median[yc] else 0

rental_signal = {}
rental_map_c = rental_map.rename(columns={rental_map.columns[0]:'상권명',
                                           rental_map.columns[1]:'대표_행정동'})
for _, mrow in rental_map_c.iterrows():
    dong, sang = mrow['대표_행정동'], mrow['상권명']
    r_sub = rental[rental['상권명']==sang].sort_values('연도').set_index('연도')
    for yr in range(2021,2025):
        if yr in r_sub.index and (yr-1) in r_sub.index:
            prev, curr = r_sub.loc[yr-1,'임대료'], r_sub.loc[yr,'임대료']
            if pd.notna(prev) and pd.notna(curr) and prev > 0:
                rental_signal[(dong, yr)] = 1 if curr > prev else 0

# ────────────────────────────────────────────────────────────────
# 전환 레퍼런스 탐색 (16개)
# ────────────────────────────────────────────────────────────────
rows_v = []
for _, t in tsm.iterrows():
    dong, yr = t['행정동'], int(t['연도'])
    s1 = int(t['업종점수'] >= TH_INDUSTRY)
    s2 = int(t['2030비중'] >= TH_2030)
    s3 = blog_annual.get((dong, yr), None)
    s4 = rental_signal.get((dong, yr), None)
    status = '미해당'
    if s1 and s2:
        status = '확정' if (s3==1 or s4==1) else '후보'
    rows_v.append({'행정동':dong,'연도':yr,'s1':s1,'s2':s2,'s3':s3,'s4':s4,'상태':status,
                   '업종점수':t['업종점수'],'2030비중':t['2030비중']})

val_df = pd.DataFrame(rows_v)
status_2020 = val_df[val_df['연도']==2020].set_index('행정동')['상태']
status_late = val_df[val_df['연도'].isin([2022,2023,2024])].groupby('행정동')['상태'].apply(
    lambda x: '확정' if '확정' in x.values else ('후보' if '후보' in x.values else '미해당')
)

ref_list = []
for dong in status_2020.index:
    if dong not in status_late.index: continue
    if status_2020[dong]=='미해당' and status_late[dong] in ['확정','후보']:
        sub = val_df[(val_df['행정동']==dong)&(val_df['연도']>=2021)].sort_values('연도')
        t_yr = None
        for _, r in sub.iterrows():
            if r['상태'] in ['확정','후보']:
                t_yr = int(r['연도']); break
        has_s3 = any(blog_annual.get((dong,y),0)==1 for y in range(2020,2025))
        has_s4 = any(rental_signal.get((dong,y),0)==1 for y in range(2020,2025))
        ref_list.append({'행정동':dong, '전환상태':status_late[dong], '전환시점':t_yr,
                         'S3_블로그':has_s3, 'S4_임대료':has_s4})

ref_df = pd.DataFrame(ref_list).sort_values(['전환상태','S3_블로그','S4_임대료'], ascending=[True,False,False])
print(f'레퍼런스 총 {len(ref_df)}개: 확정 {(ref_df["전환상태"]=="확정").sum()}개, 후보 {(ref_df["전환상태"]=="후보").sum()}개')

# ────────────────────────────────────────────────────────────────
# 이동비율 데이터 (20대 YoY 피크 계산)
# ────────────────────────────────────────────────────────────────
mig20  = mig_raw[mig_raw['연령대']==20].groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mall   = mig_raw.groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mq     = mig20.merge(mall, on=['도착_행정동코드','연도','분기'], suffixes=('_20','_전체'))
mq['비율'] = mq['합계_이동인구_20'] / mq['합계_이동인구_전체'].replace(0, np.nan)
mq['행정동'] = mq['도착_행정동코드'].map(kt_map)
mq_yr  = mq.groupby(['행정동','연도'])['비율'].mean().reset_index()

def get_yoy_peak(dong):
    sub = mq_yr[mq_yr['행정동']==dong].sort_values('연도')
    if len(sub) < 2: return np.nan
    best_yr, best_val = None, -999
    for yr in range(2021, 2025):
        cur  = sub[sub['연도']==yr]['비율']
        prev = sub[sub['연도']==yr-1]['비율']
        if len(cur) and len(prev) and prev.values[0]>0:
            yoy = (cur.values[0] - prev.values[0]) / prev.values[0]
            if yoy > best_val:
                best_val, best_yr = yoy, yr
    return round(best_val*100, 1) if best_yr else np.nan

# ────────────────────────────────────────────────────────────────
# 상권 통칭 매핑 (알려진 것)
# ────────────────────────────────────────────────────────────────
NICKNAME = {
    '성수1가2동':'뚝섬/성수','성수2가1동':'성수','성수2가3동':'성수',
    '합정동':'합정/홍대 인근','청운효자동':'서촌','삼성1동':'코엑스/삼성',
    '역삼1동':'강남역','구로3동':'구로디지털단지','군자동':'군자/자양',
    '방배2동':'방배','가양1동':'가양','대학동':'서울대입구/관악',
    '사직동':'사직/경복궁','상계2동':'상계','신촌동':'신촌/연대앞',
    '휘경1동':'휘경/외대앞',
}

# ────────────────────────────────────────────────────────────────
# §3.2.1 기본 정보표 생성
# ────────────────────────────────────────────────────────────────
at_map     = dict(zip(area_type['행정동'], area_type['구역유형']))
conc_map   = dict(zip(area_type['행정동'], area_type['주말집중도']))

table_321 = []
for i, (_, r) in enumerate(ref_df.iterrows(), 1):
    dong = r['행정동']
    # S1/S2 최초 동시 충족 연도
    sub_d = val_df[(val_df['행정동']==dong)&(val_df['연도']>=2021)].sort_values('연도')
    s1_yr = next((int(x['연도']) for _, x in sub_d.iterrows() if x['s1']==1), None)
    s2_yr = next((int(x['연도']) for _, x in sub_d.iterrows() if x['s2']==1), None)
    table_321.append({
        '번호':         i,
        '행정동':       dong,
        '자치구':       dong_gu_map.get(dong, ''),
        '상권통칭':     NICKNAME.get(dong, ''),
        '레퍼런스등급': r['전환상태'],
        '전환시점(연도)': r['전환시점'],
        'S1_업종충족연도': s1_yr,
        'S2_2030소비충족연도': s2_yr,
        'S3_블로그신호': 'O' if r['S3_블로그'] else 'X',
        'S4_임대료신호': 'O' if r['S4_임대료'] else 'X',
    })

df_321 = pd.DataFrame(table_321)
df_321.to_csv('report/data/reference_3_2_1.csv', encoding='utf-8-sig', index=False)
print('\n=== §3.2.1 레퍼런스 기본 정보표 ===')
print(df_321.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# §3.2.2 핵심 수치표 생성
# ────────────────────────────────────────────────────────────────
table_322 = []
for _, r in ref_df.iterrows():
    dong = r['행정동']
    r20 = tsm[(tsm['행정동']==dong)&(tsm['연도']==2020)]
    r24 = tsm[(tsm['행정동']==dong)&(tsm['연도']==2024)]
    u20 = round(r20['업종점수'].values[0], 3) if len(r20) else np.nan
    u24 = round(r24['업종점수'].values[0], 3) if len(r24) else np.nan
    b20 = round(r20['2030비중'].values[0], 3) if len(r20) else np.nan
    b24 = round(r24['2030비중'].values[0], 3) if len(r24) else np.nan
    yoy_pk = get_yoy_peak(dong)
    conc   = round(conc_map.get(dong, np.nan), 3) if dong in conc_map else None
    atype  = at_map.get(dong, '미측정')
    table_322.append({
        '행정동':          dong,
        '레퍼런스등급':    r['전환상태'],
        '업종점수_2020':   u20,
        '업종점수_2024':   u24,
        '업종점수_변화':   round(u24-u20, 3) if (not np.isnan(u20) and not np.isnan(u24)) else None,
        '2030비중_2020':   b20,
        '2030비중_2024':   b24,
        '2030비중_변화':   round(b24-b20, 3) if (not np.isnan(b20) and not np.isnan(b24)) else None,
        '20대이동YoY피크(%)': yoy_pk,
        '주말집중도':       conc,
        '구역유형':         atype,
    })

df_322 = pd.DataFrame(table_322)
df_322.to_csv('report/data/reference_3_2_2.csv', encoding='utf-8-sig', index=False)
print('\n=== §3.2.2 레퍼런스 핵심 수치표 ===')
print(df_322.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# 연도별 신호 상세 (참고용)
# ────────────────────────────────────────────────────────────────
detail_rows = []
for dong in ref_df['행정동']:
    for yr in range(2020, 2025):
        sub = val_df[(val_df['행정동']==dong)&(val_df['연도']==yr)]
        if len(sub)==0: continue
        s = sub.iloc[0]
        detail_rows.append({
            '행정동':dong, '연도':yr,
            '업종점수':round(s['업종점수'],3), '2030비중':round(s['2030비중'],3),
            'S1_업종':s['s1'], 'S2_2030':s['s2'],
            'S3_블로그':s['s3'], 'S4_임대료':s['s4'],
            '상태':s['상태'],
        })

df_detail = pd.DataFrame(detail_rows)
df_detail.to_csv('report/data/reference_signal_detail.csv', encoding='utf-8-sig', index=False)

print('\n=== 완료 ===')
print('  report/data/reference_3_2_1.csv')
print('  report/data/reference_3_2_2.csv')
print('  report/data/reference_signal_detail.csv')
