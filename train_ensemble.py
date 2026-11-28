import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_distributor(name):
    if pd.isna(name):
        return "기타"
    name = re.sub(r"\(주\)|㈜|\s+", "", str(name))

    major_map = {
        "CJ": ["CJ", "씨제이"],
        "롯데": ["롯데"],
        "쇼박스": ["쇼박스"],
        "NEW": ["NEW", "넥스트엔터테인먼트월드"],
        "폭스": ["폭스", "이십세기폭스"],
        "워너": ["워너브러더스"],
        "소니": ["소니픽쳐스", "콜럼비아"],
        "메가박스": ["메가박스"],
        "리틀빅": ["리틀빅픽처스"],
        "CGV": ["CGV", "아트하우스"],
        "인벤트": ["인벤트"],
        "시네마서비스": ["시네마서비스"],
    }
    for key, keywords in major_map.items():
        for kw in keywords:
            if kw in name:
                return key
    return name


def feature_engineering(train_df, test_df):
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # 1. 배급사 정규화 및 대형 배급사 여부
    df["dist_clean"] = df["distributor"].apply(clean_distributor)
    top_dists = ["CJ", "롯데", "쇼박스", "NEW", "폭스", "워너", "시네마서비스"]
    df["is_major_dist"] = df["dist_clean"].isin(top_dists).astype(int)
    df["dist_movie_cnt"] = df["dist_clean"].map(df["dist_clean"].value_counts())

    # 2. 개봉일 시계열 및 성수기 파생변수
    df["release_time"] = pd.to_datetime(df["release_time"])
    df["rel_year"] = df["release_time"].dt.year
    df["rel_month"] = df["release_time"].dt.month
    df["rel_day"] = df["release_time"].dt.day
    df["rel_dayofweek"] = df["release_time"].dt.dayofweek
    df["rel_quarter"] = df["release_time"].dt.quarter
    df["is_peak_season"] = df["rel_month"].isin([1, 2, 7, 8, 9, 12]).astype(int)

    # 3. 상영등급 순서형 인코딩
    rat_map = {"전체 관람가": 0, "12세 관람가": 1, "15세 관람가": 2, "청소년 관람불가": 3}
    df["screening_rat_num"] = df["screening_rat"].map(rat_map).fillna(2)

    # 4. 감독 파급력 피처
    df["has_dir_prev"] = (~df["dir_prev_bfnum"].isna()).astype(int)
    df["dir_prev_bfnum_fill"] = df["dir_prev_bfnum"].fillna(0)
    df["log_dir_prev_bfnum"] = np.log1p(df["dir_prev_bfnum_fill"])
    df["dir_total_bfnum"] = df["dir_prev_bfnum_fill"] * df["dir_prev_num"]
    df["log_dir_total_bfnum"] = np.log1p(df["dir_total_bfnum"])
    df["director_movie_cnt"] = df["director"].map(df["director"].value_counts())

    # 5. 제작 규모 및 밀도 피처
    df["log_num_staff"] = np.log1p(df["num_staff"])
    df["staff_per_min"] = df["num_staff"] / (df["time"] + 1)
    df["staff_per_actor"] = df["num_staff"] / (df["num_actor"] + 1)
    df["scale_score"] = df["num_staff"] * df["is_major_dist"]

    genre_staff_mean = df.groupby("genre")["num_staff"].transform("mean")
    df["staff_ratio_by_genre"] = df["num_staff"] / (genre_staff_mean + 1)

    # 6. 외부 데이터(KOBIS) 파생변수 (존재할 경우 자동 생성)
    if "kobis_open_scrn_cnt" in df.columns:
        df["has_boxoffice_rank"] = (~df["kobis_open_scrn_cnt"].isna()).astype(int)
        df["kobis_open_scrn_fill"] = df["kobis_open_scrn_cnt"].fillna(0)
        df["kobis_open_show_fill"] = df["kobis_open_show_cnt"].fillna(0)
        df["log_open_scrn"] = np.log1p(df["kobis_open_scrn_fill"])
        df["log_open_show"] = np.log1p(df["kobis_open_show_fill"])
        df["show_per_scrn"] = df["kobis_open_show_fill"] / (df["kobis_open_scrn_fill"] + 1)

    # 7. 범주형 변수 처리
    cat_cols = ["dist_clean", "genre", "screening_rat", "director"]
    for ext_col in ["kobis_nation", "kobis_watch_grade"]:
        if ext_col in df.columns:
            cat_cols.append(ext_col)

    for col in cat_cols:
        df[col] = df[col].astype(str).astype("category")

    drop_cols = ["title", "distributor", "release_time", "box_off_num"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_fe = df.iloc[: len(train_df)][feature_cols].copy()
    test_fe = df.iloc[len(train_df) :][feature_cols].copy()

    return train_fe, test_fe, cat_cols


def main():
    ext_train_path = os.path.join(BASE_DIR, "movies_train_ext.csv")
    ext_test_path = os.path.join(BASE_DIR, "movies_test_ext.csv")

    if os.path.exists(ext_train_path) and os.path.exists(ext_test_path):
        train = pd.read_csv(ext_train_path)
        test = pd.read_csv(ext_test_path)
        print("[INFO] KOBIS 외부 데이터 병합 파일 로드 완료")
    else:
        train = pd.read_csv(os.path.join(BASE_DIR, "movies_train.csv"))
        test = pd.read_csv(os.path.join(BASE_DIR, "movies_test.csv"))
        print("[INFO] 기본 데이터 파일 로드 완료")

    submission = pd.read_csv(os.path.join(BASE_DIR, "submission.csv"))

    X_train, X_test, cat_cols = feature_engineering(train, test)
    y_train_log = np.log1p(train["box_off_num"].values)
    y_train_orig = train["box_off_num"].values

    # Out-of-Fold 타겟 인코딩
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for col in ["dist_clean", "genre"]:
        X_train[f"{col}_te"] = np.nan
        for tr_idx, val_idx in kf.split(X_train):
            tr_df = pd.DataFrame({col: X_train.iloc[tr_idx][col], "target": y_train_log[tr_idx]})
            means = tr_df.groupby(col, observed=False)["target"].mean()
            global_mean = y_train_log[tr_idx].mean()
            X_train.iloc[val_idx, X_train.columns.get_loc(f"{col}_te")] = (
                X_train.iloc[val_idx][col].map(means).astype(float).fillna(global_mean)
            )

        full_df = pd.DataFrame({col: X_train[col], "target": y_train_log})
        full_means = full_df.groupby(col, observed=False)["target"].mean()
        X_test[f"{col}_te"] = X_test[col].map(full_means).astype(float).fillna(y_train_log.mean())

    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))

    preds_lgb = np.zeros(len(X_test))
    preds_xgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train_log[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train_log[val_idx]

        # 1) LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
        )
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = np.expm1(np.clip(model_lgb.predict(X_va), 0, None))
        preds_lgb += np.expm1(np.clip(model_lgb.predict(X_test), 0, None)) / kf.n_splits

        # 2) XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=100,
            random_state=42 + fold,
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = np.expm1(np.clip(model_xgb.predict(X_va), 0, None))
        preds_xgb += np.expm1(np.clip(model_xgb.predict(X_test), 0, None)) / kf.n_splits

        # 3) CatBoost
        X_tr_cat, X_va_cat, X_te_cat = X_tr.copy(), X_va.copy(), X_test.copy()
        for c in cat_cols:
            X_tr_cat[c] = X_tr_cat[c].astype(str)
            X_va_cat[c] = X_va_cat[c].astype(str)
            X_te_cat[c] = X_te_cat[c].astype(str)

        model_cat = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.025,
            depth=5,
            l2_leaf_reg=5,
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0,
        )
        model_cat.fit(
            X_tr_cat,
            y_tr,
            eval_set=(X_va_cat, y_va),
            early_stopping_rounds=100,
            verbose=False,
        )
        oof_cat[val_idx] = np.expm1(np.clip(model_cat.predict(X_va_cat), 0, None))
        preds_cat += np.expm1(np.clip(model_cat.predict(X_te_cat), 0, None)) / kf.n_splits

    print(f"LGBM OOF RMSE: {root_mean_squared_error(y_train_orig, oof_lgb):,.2f}")
    print(f"XGB  OOF RMSE: {root_mean_squared_error(y_train_orig, oof_xgb):,.2f}")
    print(f"CAT  OOF RMSE: {root_mean_squared_error(y_train_orig, oof_cat):,.2f}")

    # 최적 블렌딩 가중치 탐색
    best_rmse = float("inf")
    best_weights = (0.33, 0.33, 0.34)
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1 - w1 - w2
            if w3 < -1e-5:
                continue
            blend_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            score = root_mean_squared_error(y_train_orig, blend_oof)
            if score < best_rmse:
                best_rmse = score
                best_weights = (w1, w2, w3)

    w1, w2, w3 = best_weights
    print(f"\n최적 앙상블 가중치 (LGBM, XGB, CAT): ({w1:.2f}, {w2:.2f}, {w3:.2f})")
    print(f"최종 앙상블 OOF RMSE: {best_rmse:,.2f}")

    final_preds = np.clip(w1 * preds_lgb + w2 * preds_xgb + w3 * preds_cat, 0, None)
    submission["box_off_num"] = final_preds
    out_path = os.path.join(BASE_DIR, "submission_ensemble.csv")
    submission.to_csv(out_path, index=False)
    print(f"제출 파일 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
