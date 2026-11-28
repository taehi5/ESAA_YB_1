"""04·05 노트북 공통 파생변수 함수.

make_features(train, test, opts) -> (X, X_test)
opts 예: {'debut': 1, 'distributor': 1, 'date': 1, 'rating': 1}
그룹 설명은 04_tabpfn_insights.ipynb 1번 섹션 참고.
06에서 추가: 'debut_mode' ('zero_flag' | 'zero' | 'nan'), 'comp_all', 'comp_big' (경쟁작 수)
"""
import re

import numpy as np
import pandas as pd

RATINGS = ['전체 관람가', '12세 관람가', '15세 관람가', '청소년 관람불가']
BIG4 = {'CJ': 'CJ', '롯데': '롯데', 'NEW': 'NEW', '쇼박스': '쇼박스'}


def norm_distributor(name):
    n = re.sub(r'\(주\)|㈜|주식회사|\s', '', name)
    for key, brand in BIG4.items():
        if key.lower() in n.lower():
            return brand
    return n


def make_features(train, test, opts):
    tr, te = train.copy(), test.copy()
    feats = []

    # 공통: 기본 수치형 + 개봉 연·월 (02 베이스라인과 동일)
    for df in (tr, te):
        d = pd.to_datetime(df['release_time'])
        df['년'] = d.dt.year
        df['월'] = d.dt.month
    feats += ['time', 'dir_prev_num', 'num_staff', 'num_actor', '년', '월']

    # debut: dir_prev_bfnum 결측(=데뷔작) 처리
    #   debut_mode = 'zero_flag' (기본): 0으로 채우고 is_debut 플래그 추가
    #                'zero'            : 0으로만 일괄 채우기
    #                'nan'             : 결측 그대로 두기 (TabPFN이 직접 처리)
    if opts.get('debut'):
        mode = opts.get('debut_mode', 'zero_flag')
        for df in (tr, te):
            if mode == 'zero_flag':
                df['is_debut'] = df['dir_prev_bfnum'].isna().astype(int)
            if mode in ('zero_flag', 'zero'):
                df['dir_prev_bfnum'] = df['dir_prev_bfnum'].fillna(0)
        feats += ['dir_prev_bfnum'] + (['is_debut'] if mode == 'zero_flag' else [])

    # log_num: 치우친 수치형 로그 변환
    if opts.get('log_num'):
        for col in ['num_staff', 'dir_prev_bfnum']:
            if col in feats:
                for df in (tr, te):
                    df[f'log_{col}'] = np.log1p(df[col])
                feats = [f'log_{col}' if f == col else f for f in feats]

    # 배급사: 이름 정규화 → 4대 배급사 / 편수 기반 묶기 / 배급사 편수
    if opts.get('distributor'):
        for df in (tr, te):
            df['dist_norm'] = df['distributor'].map(norm_distributor)
        cnt = tr['dist_norm'].value_counts()
        for df in (tr, te):
            df['dist_count'] = df['dist_norm'].map(cnt).fillna(0)
            df['is_big4'] = df['dist_norm'].isin(BIG4.values()).astype(int)
            df['dist_group'] = np.select(
                [df['dist_norm'].isin(BIG4.values()), df['dist_count'] >= 10, df['dist_count'] >= 2],
                [df['dist_norm'], '중견(10편+)', '소형(2~9편)'], default='1편·신규')
        feats += ['dist_count', 'is_big4', 'dist_group']
    else:  # 02 베이스라인: 상위 5개 + 기타
        top5 = tr['distributor'].value_counts().index[:5]
        for df in (tr, te):
            df['dist_group'] = df['distributor'].where(df['distributor'].isin(top5), '기타')
        feats += ['dist_group']

    # 개봉 시기: 요일 · 분기 · 성수기
    if opts.get('date'):
        for df in (tr, te):
            d = pd.to_datetime(df['release_time'])
            df['weekday'] = d.dt.day_of_week
            df['quarter'] = d.dt.quarter
            df['is_peak'] = d.dt.month.isin([7, 8, 12, 1]).astype(int)
        feats += ['weekday', 'quarter', 'is_peak']

    # 관람 등급: 15세 · 전체 관람가 플래그 + 등급별 log(스태프 수)
    if opts.get('rating'):
        for df in (tr, te):
            df['is_15'] = (df['screening_rat'] == '15세 관람가').astype(int)
            df['is_all'] = (df['screening_rat'] == '전체 관람가').astype(int)
            log_staff = np.log1p(df['num_staff'])
            for i, r in enumerate(RATINGS):
                df[f'staff_x_rat{i}'] = log_staff * (df['screening_rat'] == r)
        feats += ['is_15', 'is_all'] + [f'staff_x_rat{i}' for i in range(len(RATINGS))]

    # 경쟁작 수: 개봉일 ±7일 안에 개봉한 다른 영화 수 (train + test 개봉일 모두 사용, 관객수는 안 씀)
    #   comp_all: 전체 경쟁작 수 / comp_big: 그중 큰 영화(스태프 수 상위 25%) 수
    if opts.get('comp_all') or opts.get('comp_big'):
        both = pd.concat([tr[['release_time', 'num_staff']], te[['release_time', 'num_staff']]], ignore_index=True)
        days = pd.to_datetime(both['release_time']).values.astype('datetime64[D]').astype(np.int64)
        near = (np.abs(days[:, None] - days[None, :]) <= 7) & ~np.eye(len(both), dtype=bool)
        is_big = (both['num_staff'] >= both['num_staff'].quantile(0.75)).values
        comp = {'comp_all': near.sum(axis=1), 'comp_big': (near & is_big[None, :]).sum(axis=1)}
        for name in ['comp_all', 'comp_big']:
            if opts.get(name):
                tr[name] = comp[name][:len(tr)]
                te[name] = comp[name][len(tr):]
                feats.append(name)

    # 제목: 숫자 포함(속편 가능성) · 부제 포함
    if opts.get('title'):
        for df in (tr, te):
            df['title_has_digit'] = df['title'].str.contains(r'\d').astype(int)
            df['title_has_sub'] = df['title'].str.contains(r'[:：\-]').astype(int)
        feats += ['title_has_digit', 'title_has_sub']

    # 범주형 원핫 (장르 · 등급 · 배급사 그룹)
    cat_cols = ['genre', 'screening_rat', 'dist_group']
    X = pd.get_dummies(tr[feats + cat_cols[:2]], columns=cat_cols, dtype=float)
    X_test = pd.get_dummies(te[feats + cat_cols[:2]], columns=cat_cols, dtype=float).reindex(columns=X.columns, fill_value=0)
    return X.astype(float), X_test.astype(float)
