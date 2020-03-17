import datetime
import gc
import os
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset
PROJECT_NAME = "TalkingData AdTracking Fraud Detection"
PROJECT_FOLDER_PATH = os.path.join(
    os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME
)
VANILLA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_FOLDER_PATH, "train.csv")
TEST_FILE_PATH = os.path.join(VANILLA_FOLDER_PATH, "test.csv")
SAMPLE_NUM = None

# Submission
TEAM_NAME = "Aurora"
SUBMISSION_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "submission")
os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

# Training and Testing procedure
NUM_BOOST_ROUND = 1000000
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 10

# Generate a zip archive for a file
create_zip_archive = lambda file_path: shutil.make_archive(
    file_path[: file_path.rindex(".")],
    "zip",
    os.path.abspath(os.path.join(file_path, "..")),
    os.path.basename(file_path),
)


def release_resources():
    unreachable_objects_num = gc.collect()
    print("Collected {} unreachable objects ...".format(unreachable_objects_num))


def load_data(nrows=SAMPLE_NUM):
    parse_dates = ["click_time"]
    dtype = {
        "ip": "uint32",
        "app": "uint16",
        "device": "uint16",
        "os": "uint16",
        "channel": "uint16",
        "is_attributed": "bool",
        "click_id": "uint32",
    }

    print("Loading training data ...")
    train_df = pd.read_csv(
        TRAIN_FILE_PATH, parse_dates=parse_dates, dtype=dtype, nrows=nrows
    )
    train_df.drop("attributed_time", axis=1, inplace=True)
    train_num = len(train_df)

    print("Loading testing data ...")
    test_df = pd.read_csv(
        TEST_FILE_PATH, parse_dates=parse_dates, dtype=dtype, nrows=nrows
    )
    submission_df = pd.DataFrame(test_df["click_id"])
    test_df.drop("click_id", axis=1, inplace=True)

    print("Merging training data and testing data ...")
    merged_df = pd.concat([train_df, test_df], copy=False)
    train_df, test_df = None, None
    release_resources()

    print("Extracting date time info ...")
    merged_df["day"] = merged_df["click_time"].dt.day.astype("uint8")
    merged_df["hour"] = merged_df["click_time"].dt.hour.astype("uint8")
    merged_df.drop("click_time", axis=1, inplace=True)

    print("Grouping by ip-day-hour ...")
    temp_df = (
        merged_df[["ip", "day", "hour", "channel"]]
        .groupby(by=["ip", "day", "hour"])[["channel"]]
        .count()
        .reset_index()
        .rename(index=str, columns={"channel": "ip_tcount"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "day", "hour"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip-app ...")
    temp_df = (
        merged_df[["ip", "app", "channel"]]
        .groupby(by=["ip", "app"])[["channel"]]
        .count()
        .reset_index()
        .rename(index=str, columns={"channel": "ip_app_count"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "app"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip-app-os ...")
    temp_df = (
        merged_df[["ip", "app", "os", "channel"]]
        .groupby(by=["ip", "app", "os"])[["channel"]]
        .count()
        .reset_index()
        .rename(index=str, columns={"channel": "ip_app_os_count"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "app", "os"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip_day_chl_var_hour ...")
    temp_df = (
        merged_df[["ip", "day", "hour", "channel"]]
        .groupby(by=["ip", "day", "channel"])[["hour"]]
        .var()
        .reset_index()
        .rename(index=str, columns={"hour": "ip_tchan_count"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "day", "channel"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip_app_os_var_hour ...")
    temp_df = (
        merged_df[["ip", "app", "os", "hour"]]
        .groupby(by=["ip", "app", "os"])[["hour"]]
        .var()
        .reset_index()
        .rename(index=str, columns={"hour": "ip_app_os_var"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "app", "os"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip_app_channel_var_day ...")
    temp_df = (
        merged_df[["ip", "app", "channel", "day"]]
        .groupby(by=["ip", "app", "channel"])[["day"]]
        .var()
        .reset_index()
        .rename(index=str, columns={"day": "ip_app_channel_var_day"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "app", "channel"], how="left")
    temp_df = None
    release_resources()

    print("Grouping by ip_app_chl_mean_hour ...")
    temp_df = (
        merged_df[["ip", "app", "channel", "hour"]]
        .groupby(by=["ip", "app", "channel"])[["hour"]]
        .mean()
        .reset_index()
        .rename(index=str, columns={"hour": "ip_app_channel_mean_hour"})
    )
    merged_df = merged_df.merge(temp_df, on=["ip", "app", "channel"], how="left")
    temp_df = None
    release_resources()

    print("Splitting data ...")
    train_indexes, valid_indexes = train_test_split(
        np.arange(train_num), test_size=0.1, random_state=0
    )
    train_df, valid_df = merged_df.iloc[train_indexes], merged_df.iloc[valid_indexes]
    test_df = merged_df.iloc[train_num:].drop("is_attributed", axis=1)
    merged_df = None
    release_resources()

    print("Getting summary of train_df, valid_df and test_df ...")
    for current_df in [train_df, valid_df, test_df]:
        current_df.info(verbose=False, memory_usage=True)

    return train_df, valid_df, test_df, submission_df


def run():
    print("Loading data ...")
    train_df, valid_df, test_df, submission_df = load_data()

    print("Generating LightGBM datasets ...")
    target_name = "is_attributed"
    categorical_feature = ["ip", "app", "device", "os", "channel"]
    train_dataset = lgb.Dataset(
        train_df.drop(target_name, axis=1),
        train_df[target_name],
        categorical_feature=categorical_feature,
    )
    train_df = None
    release_resources()
    valid_dataset = lgb.Dataset(
        valid_df.drop(target_name, axis=1),
        valid_df[target_name],
        categorical_feature=categorical_feature,
        reference=train_dataset,
    )
    valid_df = None
    release_resources()

    print("Performing the training procedure ...")
    best_params = {
        "learning_rate": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
    }  # Use empirical parameters
    model = lgb.train(
        params=best_params,
        train_set=train_dataset,
        valid_sets=[valid_dataset],
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )

    print("Performing the testing procedure ...")
    prediction_array = model.predict(test_df, num_iteration=model.best_iteration)
    submission_df["is_attributed"] = prediction_array
    submission_file_path = os.path.join(
        SUBMISSION_FOLDER_PATH,
        "{} {}.csv".format(
            TEAM_NAME, str(datetime.datetime.now()).split(".")[0]
        ).replace(" ", "_"),
    )
    print("Saving submission to {} ...".format(submission_file_path))
    submission_df.to_csv(submission_file_path, float_format="%.6f", index=False)
    compressed_submission_file_path = create_zip_archive(submission_file_path)
    print(
        "Saving compressed submission to {} ...".format(compressed_submission_file_path)
    )

    print("All done!")


if __name__ == "__main__":
    run()
