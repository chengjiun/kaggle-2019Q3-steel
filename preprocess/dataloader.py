from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
import pandas as pd

image_df = pd.read_pickle("./image_df.pickle")
MEAN1 = image_df["mean"].median() / 255
STD1 = image_df["std"].median() / 255
MEAN2 = image_df["mean_val"].median() / 255
STD2 = image_df["std_val"].median() / 255
IMAGE_STATS_GLOBAL1 = tuple([[MEAN1] * 3, [STD1] * 3])
IMAGE_STATS_GLOBAL2 = tuple([[MEAN1] * 3, [STD1] * 3])


def upsample_defect(df):
    df_normal = df.loc[df["Detected"] is False]
    df_detect = df.loc[df["Detected"] is True]
    df_detect_upsampled = resample(df_detect, replace=False, n_samples=len(df_normal))
    return pd.concat([df_normal, df_detect_upsampled])


def prepare_train_valid_split(image_df, valid_ratio=0.1, upsample=False):
    print(f"run split sample with {valid_ratio}, upsample: {upsample}")
    image_df["is_valid"] = False
    sss = StratifiedShuffleSplit(test_size=valid_ratio)
    train_index, test_index = next(sss.split(image_df["ImageId"], image_df["Detected"]))
    image_df.loc[test_index, "is_valid"] = True
    print(f'train / valid ratio: {image_df["is_valid"].mean()}')
    tr_df = image_df[image_df["is_valid"] == False]
    val_df = image_df[image_df["is_valid"] == True]

    if upsample:
        tr_df = upsample_defect(tr_df)
        image_df = pd.concat([tr_df, val_df])
    print(f'train defect ratio: {tr_df["Detected"].mean()}')
    print(f'valid defect ratio: {val_df["Detected"].mean()}')

    return image_df


def prepare_class_df(train_df_path, defect_types=None, valid_ratio=0.1, upsample=None):
    if defect_types is None:
        defect_types = ["1", "2", "3", "4"]
    print(f"select defect types {defect_types}")
    df = pd.read_csv(train_df_path)
    df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
    df["ClassId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[1])
    df["Detected"] = df["EncodedPixels"].isnull() == False
    df = df[df["ClassId"].isin(defect_types)]
    image_df = pd.DataFrame(
        (df.groupby("ImageId")["Detected"].sum() > 0).astype("int")
    ).reset_index()
    print(f'images included in the df: {len(image_df)} from {df["ImageId"].nunique()}')
    print(
        f'images with defect {image_df["Detected"].sum()}, ratio {image_df["Detected"].mean()}'
    )
    if valid_ratio > 0:
        image_df = prepare_train_valid_split(image_df, valid_ratio, upsample)
    return image_df

