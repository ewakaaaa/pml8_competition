from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
from transliterate import translit
import pandas as pd


class DFTransform(TransformerMixin, BaseEstimator):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def fit(self, *_):
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)

    def get_feature_names(self):
        return self.X.column_name.tolist()


def one_hot_encoding(df, column, classes):
    df[column] = [i if str(i) != 'nan' else [] for i in df[column]]
    mlb = MultiLabelBinarizer()
    mlb.fit([[i] for i in classes])
    res = mlb.transform(df[column])
    res = pd.DataFrame(mlb.transform(df[column]), columns=[i + "_ohe" for i in classes])
    df = df.join(res)
    return df.drop(column, axis=1)


def rename_security(value):
    if (
        value == "yes"
        or value == "is"
        or value == "security"
        or value == "provided to help"
        or value == "protected area"
        or value == "protected"
        or value == "guarded area"
    ):
        return "provided"
    if value == "not allowed" or value == "no" or value == "cat t":
        return "nan"
    if (
        value == "barrier"
        or value == "ogorojennaja territory"
        or value == "closed territory"
        or value == "perimeter fencing"
        or value == "private protected area"
    ):
        return "closed area"
    if value == "access system":
        return "access control system"
    if (
        value
        == "security alarm of all premises with life support systems of the building"
        or value == "warning system and evacuation management"
        or value == "alarms"
        or value == "burglar alarm"
    ):
        return "alarm system"
    if "intercom" in value:
        return "intercom"
    if (
        "video" in value
        or value == "well guarded by security cameras around the perimeter"
        or "cctv" in value
    ):
        return "video surveillance"
    if "checkpoint" in value:
        return "checkpoint"
    if "concierge" in value or "chop" in value:
        return "concierge"
    if "fenced" in value or value == "enclosed courtyard":
        return "fenced area"
    if "security" in value or value == 'armed guards' or value == '24-hour guarded territory':
        return "round the clock security"
    if "access" in value:
        return "access control system"
    if "fire" in value:
        return "fire system"
    if "parking" in value:
        return "parking"
    else:
        return value


def security(df, column_name="Security:"):
    df["security_split"] = [
        [s.lower().strip() for s in elem.split(",")] if str(elem) != "nan" else "nan"
        for elem in df[column_name]
    ]
    df["security_clean"] = [
        list(map(rename_security, elem)) if str(elem) != "nan" else "nan"
        for elem in df["security_split"]
    ]
    df["security_ohe"] = [0 if str(elem) == "nan" else 1 for elem in df["security_clean"]]

    df = df.drop("Security:", axis=1)
    df = df.drop("security_split", axis=1)

    return df


def date(df, column_name="date"):
    def trans(value):
        return translit(value, "ru", reversed=True)

    def array_to_str(value):
        return [" ".join([str(elem).lower().strip() for elem in value])][0]

    df["date"] = [array_to_str(elem).split(" ")[1] for elem in df[column_name]]

    return df


def metro(df, column_name="breadcrumbs"):
    def get_metro(array):
        return [",".join([str(elem) for elem in array if "МЦК" in elem])][0]

    df["metro"] = [get_metro(elem) for elem in df[column_name]]
    df = df.drop("breadcrumbs", axis=1)
    return df
