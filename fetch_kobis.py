import os
import time
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# 발급받은 KOBIS 오픈API 키를 입력하세요
KOBIS_API_KEY = "2d62aaf73d7e3deec789de820b1630a9"


def fetch_kobis_full_data(df, api_key):
    search_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json"
    info_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
    box_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"

    ext_rows = []
    boxoffice_cache = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="KOBIS 데이터 수집 중"):
        title = str(row["title"]).strip()
        clean_title = title.replace(" ", "")
        release_dt = str(row["release_time"]).strip()
        release_year = release_dt[:4]
        target_dt_str = release_dt.replace("-", "")

        feat = {
            "kobis_nation": "Unknown",
            "kobis_watch_grade": "Unknown",
            "kobis_company_cnt": 0,
            "kobis_actor_cnt": 0,
            "kobis_staff_cnt": 0,
            "kobis_show_tm": np.nan,
            "kobis_open_scrn_cnt": np.nan,
            "kobis_open_show_cnt": np.nan,
            "kobis_open_rank": np.nan,
        }

        try:
            params = {
                "key": api_key,
                "movieNm": title,
                "openStartDt": str(int(release_year) - 1),
                "openEndDt": str(int(release_year) + 1),
            }
            res = requests.get(search_url, params=params, timeout=5).json()
            movie_list = res.get("movieListResult", {}).get("movieList", [])

            target_cd = None
            for m in movie_list:
                if m.get("movieNm", "").replace(" ", "") == clean_title:
                    target_cd = m.get("movieCd")
                    break
            if not target_cd and movie_list:
                target_cd = movie_list[0].get("movieCd")

            if target_cd:
                info_res = requests.get(
                    info_url, params={"key": api_key, "movieCd": target_cd}, timeout=5
                ).json()
                info = info_res.get("movieInfoResult", {}).get("movieInfo", {})

                nations = info.get("nations", [])
                if nations:
                    feat["kobis_nation"] = nations[0].get("nationNm", "Unknown")

                audits = info.get("audits", [])
                if audits:
                    feat["kobis_watch_grade"] = audits[0].get("watchGradeNm", "Unknown")

                feat["kobis_company_cnt"] = len(info.get("companys", []))
                feat["kobis_actor_cnt"] = len(info.get("actors", []))
                feat["kobis_staff_cnt"] = len(info.get("staffs", []))
                if info.get("showTm"):
                    feat["kobis_show_tm"] = float(info.get("showTm"))

            if len(target_dt_str) == 8:
                if target_dt_str not in boxoffice_cache:
                    b_res = requests.get(
                        box_url,
                        params={
                            "key": api_key,
                            "targetDt": target_dt_str,
                            "repNationCd": "K",
                        },
                        timeout=5,
                    ).json()
                    daily_list = b_res.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
                    boxoffice_cache[target_dt_str] = daily_list
                    time.sleep(0.05)

                for item in boxoffice_cache[target_dt_str]:
                    if (target_cd and item.get("movieCd") == target_cd) or (
                        item.get("movieNm", "").replace(" ", "") == clean_title
                    ):
                        feat["kobis_open_scrn_cnt"] = float(item.get("scrnCnt", np.nan))
                        feat["kobis_open_show_cnt"] = float(item.get("showCnt", np.nan))
                        feat["kobis_open_rank"] = float(item.get("rank", np.nan))
                        break

        except Exception:
            pass

        ext_rows.append(feat)
        time.sleep(0.05)

    ext_df = pd.DataFrame(ext_rows)
    return pd.concat([df.reset_index(drop=True), ext_df], axis=1)


if __name__ == "__main__":
    if KOBIS_API_KEY == "여기에_KOBIS_API_KEY_입력":
        print("경고: KOBIS_API_KEY를 입력한 뒤 실행해주세요!")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train = pd.read_csv(os.path.join(base_dir, "movies_train.csv"))
        test = pd.read_csv(os.path.join(base_dir, "movies_test.csv"))

        train_ext = fetch_kobis_full_data(train, KOBIS_API_KEY)
        test_ext = fetch_kobis_full_data(test, KOBIS_API_KEY)

        train_ext.to_csv(os.path.join(base_dir, "movies_train_ext.csv"), index=False)
        test_ext.to_csv(os.path.join(base_dir, "movies_test_ext.csv"), index=False)
        print("외부 데이터 병합 파일(movies_train_ext.csv, movies_test_ext.csv) 저장 완료!")
