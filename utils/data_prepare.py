from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer

# from transliterate import translit
import pandas as pd
from utils.my_dict import metro_dict, gminy_dict
from utils.my_list import security_unique_values, drogie


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
    df[column] = [i if str(i) != "nan" else [] for i in df[column]]
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
        or value == "secure area"
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
    if (
        "security" in value
        or value == "armed guards"
        or value == "24-hour guarded territory"
        or value == "round the clock protected area"
    ):
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
    df["security"] = [0 if str(elem) == "nan" else 1 for elem in df["security_clean"]]
    # df["security"] = [len(elem) if elem != "nan" else 0 for elem in df.security_clean]

    # df = df.drop("security_clean", axis=1)
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


def array_to_str(array):
    result = [",".join([str(elem) for elem in array])][0]
    if result != "":
        return result.strip()
    else:
        return None


def split_elements_by_prefix(array):
    elem_with_prefix = []
    elem_without_prefix = []
    for elem in array:
        split_elem = elem.split(".")
        if len(split_elem) > 1:
            elem_with_prefix.append(split_elem[1].strip())
        else:
            elem_without_prefix.append(split_elem[0])
    return elem_with_prefix, elem_without_prefix


def get_object(array, obj):
    result = [str(elem) for elem in array if obj in elem]
    if len(result) > 0:
        array = [elem for elem in array if elem not in result]
        return array_to_str(result), array
    else:
        return None, array


def get_elems(array):
    if "Москва" in array:
        array.remove("Москва")
    if "г. Москва" in array:
        array.remove("г. Москва")
    nova_mockba, array = get_object(array, "Новая Москва")
    mck, array = get_object(array, "МЦК ")
    m, array = get_object(array, "м. ")

    elem_with_prefix, elem_without_prefix = split_elements_by_prefix(array)
    return nova_mockba, mck, m, elem_with_prefix, elem_without_prefix


def check_elem_on_list(array, set_):
    if len(array) > 0:
        return array_to_str([elem for elem in array if elem in set_])


def lat_lon(elem_g, elem_m):
    if elem_g is not None:
        return gminy_dict[elem_g]
    elif elem_m is not None:
        return metro_dict[elem_m]
    else:
        return (-10, -10)


def breadcrumbs(df_origin, column_name="breadcrumbs"):
    data = [get_elems(array) for array in df_origin[column_name]]
    df = pd.DataFrame(
        data=data,
        columns=["nowa_moskwa", "metro", "m", "with_prefix", "without_prefix"],
    )
    df["gminy"] = [
        check_elem_on_list(elem, list(gminy_dict.keys()))
        for elem in df["without_prefix"]
    ]
    df["lat_lon"] = [
        lat_lon(elem_g, elem_m) for elem_g, elem_m in zip(df.gminy, df.metro)
    ]
    df["lat"], df["lon"] = df.lat_lon.str
    df["drogie_ohe"] = [1 if elem in drogie else 0 for elem in df.gminy]

    df = df.drop("lat_lon", axis=1)
    df = df.drop("with_prefix", axis=1)
    df = df.drop("without_prefix", axis=1)

    result = pd.concat([df_origin, df], axis=1)
    return result


def metro(df, column_name="breadcrumbs"):
    def get_object(array, object):
        return [",".join([str(elem) for elem in array if object in elem])]

    df["metro"] = [get_metro(elem) for elem in df[column_name]]
    df["lat_lon"] = [metro_dict[elem] for elem in df["metro"]]
    df["lat"], df["lon"] = df.lat_lon.str
    df = df.drop("breadcrumbs", axis=1)
    df = df.drop("metro", axis=1)
    df = df.drop("lat_lon", axis=1)
    return df
