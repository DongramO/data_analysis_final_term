# 데이터 컬럼 정의서

## subway_stations.csv
- rows: 8, cols: 2

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | line | str | 미분류 |
| 2 | name | str | 경제 |

## 그리드별 상권발달 개별지수.csv
- rows: 500, cols: 13

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 일자(DATE) | int64 | 201803 |
| 2 | 그리드ID(GRID50_ID) | str | GS00030952 |
| 3 | 블록코드(BLCK_CD) | str | 3*9*4* |
| 4 | 시군구코드(SIGNGU_CD) | int64 | 11110 |
| 5 | 행정동코드(ADSTRD_CD) | int64 | 11560535 |
| 6 | 지목구분(JIMOK) | str | L |
| 7 | 상권구분코드(TRDAR_SE_CD) | str | nan |
| 8 | 상권번호(TRDAR_NO) | float64 | 1000429.0 |
| 9 | 매출지수(SALES) | float64 | 20.48 |
| 10 | 인프라지수(INFRASTRUCTURE) | float64 | 62.56 |
| 11 | 가맹점지수(STORE) | float64 | 5.38 |
| 12 | 인구지수(POPULATION) | float64 | nan |
| 13 | 금융지수(DEPOSIT) | float64 | 48.29 |

## 무인점포 현황_20251231.csv
- rows: 4251, cols: 4

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 연번 | int64 | 1 |
| 2 | 자치구 | str | 종로 |
| 3 | 업소명 | str | 영카이브 서촌점 |
| 4 | 업종 | str | 무인사진관 |

## 상권별 상권발달 개별지수.csv
- rows: 500, cols: 8

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 일자(DATE) | int64 | 201903 |
| 2 | 상권구분코드(TRDAR_SE_CD) | str | A |
| 3 | 상권번호(TRDAR_NO) | int64 | 1000433 |
| 4 | 매출지수(SALES) | float64 | 14.19 |
| 5 | 인프라지수(INFRASTRUCTURE) | float64 | 30.07 |
| 6 | 가맹점지수(STORE) | float64 | 41.69 |
| 7 | 인구지수(POPULATION) | float64 | 36.05 |
| 8 | 금융지수(DEPOSIT) | float64 | 44.34 |

## 서울 관광지_SNS_채널_언급량_수.csv
- rows: 3, cols: 13

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 서울관광지_코드(SLTA_CD) | str | SLTA053 |
| 2 | 서울관광지_명(SLTA_NM) | str | 노량진수산물도매시장 |
| 3 | 연도_월(YYYYMM) | int64 | 202409 |
| 4 | 서울관광지_총 언급량(TOT_BUZZ_CNT) | int64 | 1948 |
| 5 | 서울관광지_매스미디어 언급량(MASS_BUZZ_CNT) | int64 | 294 |
| 6 | 서울관광지_인스타그램 언급량(INST_BUZZ_CNT) | int64 | 258 |
| 7 | 서울관광지_블로그 언급량(BLOG_BUZZ_CNT) | int64 | 1115 |
| 8 | 서울관광지_커뮤니티 언급량(COMM_BUZZ_CNT) | int64 | 150 |
| 9 | 서울관광지_유튜브 언급량(YT_BUZZ_CNT) | int64 | 29 |
| 10 | 서울관광지_X 언급량(X_BUZZ_CNT) | int64 | 102 |
| 11 | 서울관광지_긍정언급량(POS_SENT_CNT) | int64 | 1472 |
| 12 | 서울관광지_중립언급량(NEU_SENT_CNT) | int64 | 249 |
| 13 | 서울관광지_부정언급량(NEG_SENT_CNT) | int64 | 227 |

## 서울 관광지_감정_단어_수.csv
- rows: 964, cols: 6

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 연도_월(YYYYMM) | int64 | 202401 |
| 2 | 서울관광지_코드(SLTA_CD) | str | SLTA048 |
| 3 | 서울관광지_명(SLTA_NM) | str | 가락시장 |
| 4 | 서울관광지_긍부정(SENT_TYPE) | str | 부정 |
| 5 | 서울관광지_긍부정(단어)(SENT_WORD) | str | 못한 |
| 6 | 서울관광지_긍부정(단어) 언급량(SENT_WORD_BUZZ_CNT) | int64 | 1651 |

## 서울 관광지_마스터_테이블.csv
- rows: 3, cols: 16

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 서울관광지_코드(SLTA_CD) | str | SLTA039 |
| 2 | 서울관광지_명(SLTA_NM) | str | 해방촌신흥시장 |
| 3 | 서울관광지_유형(SLTA_TYPE) | str | 상업지 |
| 4 | 서울관광지_방문자_등급(SLTA_VSTR_GRDE) | str | S |
| 5 | 서울관광지_매출_등급(SLTA_SALE_GRDE) | str | S |
| 6 | 서울관광지_규모_등급(SLTA_SIZE_GRDE) | str | S |
| 7 | 서울관광지_주소(SLTA_ADDR) | str |  서울특별시 용산구 신흥로 95-9  |
| 8 | 서울관광지_중심_X_좌표(SLTA_COR_X_AXIS) | float64 | 126.98521 |
| 9 | 서울관광지_중심_Y_좌표(SLTA_COR_Y_AXIS) | float64 | 37.545286 |
| 10 | 서울관광지_형상정보(SLTA_SHAPE_WKT) | str | POLYGON ((954600 1949700 |
| 11 | 서울관광지_영역_셀(50X50M)_수(SLTA_AREA_CELL_CNT) | int64 | 10 |
| 12 | 서울관광지_상권_영역(SLTA_COML_DIST) | str | CELL_범위 |
| 13 | 방문_정보_존재여부(VIST_INFO_YN) | str | Y |
| 14 | 매출_정보_존재여부(SALE_INFO_YN) | str | Y |
| 15 | 관심_정보_존재여부(SNS_INFO_YN) | str | Y |
| 16 | 서울관광지_방문시간_그룹(VIST_TIME_GP) | str | 전체 |

## 서울 관광지_영역_매출.csv
- rows: 82, cols: 8

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 연월(YYYYMM) | int64 | 202412 |
| 2 | 서울관광지_코드(SLTA_CD) | str | SLTA090 |
| 3 | 서울관광지_명(SLTA_NM) | str | 남산골한옥마을 |
| 4 | 요일_구분(DAY_TYPE) | str | 주말 |
| 5 | 매출_고객_수(SALE_CUST_CNT) | float64 | nan |
| 6 | 매출_건수(SALE_CNT) | int64 | 128935 |
| 7 | 매출_액(SALE_AMT) | float64 | 3777782221.0 |
| 8 | 매출_순위(SALE_AMT_RNK) | int64 | 59 |

## 서울 관광지_영역_상권_정보.csv
- rows: 3, cols: 17

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 연월(YYYYMM) | int64 | 202412 |
| 2 | 서울관광지_코드(SLTA_CD) | str | SLTA069 |
| 3 | 서울관광지_명(SLTA_NM) | str | 청와대 |
| 4 | 매출_등급(SALE_GRADE) | int64 | 10 |
| 5 | 상업_지역_비중(COML_AREA_PCT) | float64 | 22.7 |
| 6 | 오피스_지역_비중(OFFC_AREA_PCT) | float64 | 11.4 |
| 7 | 주거_지역_비중(RESD_AREA_PCT) | float64 | 29.5 |
| 8 | 공업_지역_비중(MNFT_AREA_PCT) | float64 | 0.0 |
| 9 | 기타_지역_비중(ETC_AREA_PCT) | float64 | 36.4 |
| 10 | 역세권_지역_여부(SBWY_AREA_YN) | float64 | 0.0 |
| 11 | 대학가_지역_여부(UNIV_AREA_YN) | float64 | 1.0 |
| 12 | 음식_업종_점포_비중(FOOD_IDKT_STRE_PCT) | float64 | 48.1 |
| 13 | 소매/유통_업종_점포_비중(RTL_IDKT_STRE_PCT) | float64 | 27.3 |
| 14 | 생활서비스_업종_점포_비중(SVC_IDKT_STRE_PCT) | float64 | 8.2 |
| 15 | 의료서비스_업종_점포_비중(MEDC_IDKT_STRE_PCT) | float64 | 5.5 |
| 16 | 교육서비스_업종_점포_비중(EDU_IDKT_STRE_PCT) | float64 | 6.4 |
| 17 | 여가서비스_업종_점포_비중(LESR_IDKT_STRE_PCT) | float64 | 4.5 |

## 서울 관광지_영역_업종_매출.csv
- rows: 1000, cols: 12

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 연월(YYYYMM) | int64 | 202409 |
| 2 | 서울관광지_코드(SLTA_CD) | str | SLTA090 |
| 3 | 서울관광지_명(SLTA_NM) | str | 남산골한옥마을 |
| 4 | 요일_구분(DAY_TYPE) | str | 전체 |
| 5 | 업종_순위(IDKD_RNK) | int64 | 19 |
| 6 | 업종_분류(IDKD_CLSS) | str | 음식 |
| 7 | 업종_명(IDKD_NM) | str | 낙지/오징어 |
| 8 | 업종_코드(IDKD_CD) | str | F019 |
| 9 | 매출_고객_수(SALE_CUST_CNT) | float64 | 2796.0 |
| 10 | 매출_건수(SALE_CNT) | int64 | 2861 |
| 11 | 매출_액(SALE_AMT) | int64 | 217030386 |
| 12 | 점포_수(STORE_CNT) | int64 | 4 |

## 행정동별 상권발달 개별지수.csv
- rows: 424, cols: 7

| # | 컬럼명 | 타입 | 샘플값 |
|---|--------|------|--------|
| 1 | 일자(DATE) | int64 | 201907 |
| 2 | 행정동코드(ADSTRD_CD) | int64 | 11260575 |
| 3 | 매출지수(SALES) | float64 | 9.11 |
| 4 | 인프라지수(INFRASTRUCTURE) | float64 | 11.47 |
| 5 | 가맹점지수(STORE) | float64 | 23.19 |
| 6 | 인구지수(POPULATION) | float64 | 24.49 |
| 7 | 금융지수(DEPOSIT) | float64 | 60.92 |

