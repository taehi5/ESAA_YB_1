"""KOBIS 일별 박스오피스에서 스크린 수 · 상영 횟수를 받아 영화별 외부 변수로 정리.

사용법:  python kobis_fetch.py
- API 키는 환경변수 KOBIS_KEY 에서 읽음 (코드에 키를 쓰지 말 것)
- 날짜별 결과는 kobis_cache/YYYYMMDD.json 에 저장 → 다시 실행하면 받은 날짜는 건너뜀
- 정답에 가까운 관객수(audiCnt, audiAcc) · 매출액은 저장하지 않음 (누수 방지)
- 결과: kobis_movies.csv (KOBIS 영화별 집계)
"""
import json
import os
import sys
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

URL = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
KEEP = ['rank', 'movieCd', 'movieNm', 'openDt', 'scrnCnt', 'showCnt']
CACHE = 'kobis_cache'
START, END = date(2010, 1, 1), date(2016, 2, 29)   # 2015년 말 개봉작의 상영 기간까지


def fetch_day(key, d):
    path = os.path.join(CACHE, f'{d:%Y%m%d}.json')
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8'))
    with urllib.request.urlopen(f'{URL}?key={key}&targetDt={d:%Y%m%d}', timeout=20) as r:
        data = json.loads(r.read().decode('utf-8'))
    if 'faultInfo' in data:   # 키 오류 · 호출 한도 초과 등
        raise RuntimeError(data['faultInfo'].get('message', 'KOBIS API error'))
    rows = [{k: m.get(k) for k in KEEP} for m in data['boxOfficeResult']['dailyBoxOfficeList']]
    json.dump(rows, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    time.sleep(0.05)
    return rows


def main():
    key = os.environ.get('KOBIS_KEY')
    if not key:
        sys.exit('환경변수 KOBIS_KEY가 없습니다.')
    os.makedirs(CACHE, exist_ok=True)

    days = [START + timedelta(n) for n in range((END - START).days + 1)]
    records = []
    for i, d in enumerate(days):
        for attempt in range(3):
            try:
                rows = fetch_day(key, d)
                break
            except RuntimeError as e:
                sys.exit(f'{d} 에서 중단: {e} (받은 날짜는 저장됨 → 나중에 다시 실행하면 이어서 받음)')
            except Exception as e:
                if attempt == 2:
                    sys.exit(f'{d} 네트워크 오류: {type(e).__name__}')
                time.sleep(2)
        for r in rows:
            records.append({**r, 'targetDt': d.isoformat()})
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(days)}일 완료', flush=True)

    df = pd.DataFrame(records)
    for c in ['rank', 'scrnCnt', 'showCnt']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['openDt'] = df['openDt'].str.strip().replace('', None)

    # 영화별 집계: 개봉일(또는 개봉 후 첫 진입일) 스크린 수, 최대 스크린/상영 횟수, 상위 10위 일수, 최고 순위
    after_open = df[df['openDt'].isna() | (df['targetDt'] >= df['openDt'].fillna(''))]
    first = after_open.sort_values('targetDt').groupby('movieCd').first()
    agg = df.groupby('movieCd').agg(movieNm=('movieNm', 'first'), openDt=('openDt', 'first'),
                                    max_scrn=('scrnCnt', 'max'), max_show=('showCnt', 'max'),
                                    days_top10=('targetDt', 'nunique'), best_rank=('rank', 'min'))
    agg['open_scrn'] = first['scrnCnt']
    agg['open_show'] = first['showCnt']
    agg.reset_index().to_csv('kobis_movies.csv', index=False, encoding='utf-8-sig')
    print(f'완료: 영화 {len(agg)}편 → kobis_movies.csv')


if __name__ == '__main__':
    main()
