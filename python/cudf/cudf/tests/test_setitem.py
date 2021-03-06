# Copyright (c) 2018, NVIDIA CORPORATION.

import pandas as pd
import pytest

from cudf.core.dataframe import DataFrame, Series
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [[True, False, True], [True, True, True]])
@pytest.mark.parametrize("value", [0, -1])
def test_dataframe_setitem_bool_mask_scaler(df, arg, value):
    gdf = DataFrame.from_pandas(df)

    df[arg] = value
    gdf[arg] = value
    assert_eq(df, gdf)


# pandas incorrectly adds nulls with dataframes
# but works fine with scalers
@pytest.mark.xfail()
def test_dataframe_setitem_scaler_bool_inconsistency():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df[[True, False, True]] = pd.DataFrame({"a": [-1, -2]})

    gdf = DataFrame({"a": [1, 2, 3]})
    gdf[[True, False, True]] = DataFrame({"a": [-1, -2]})
    assert_eq(df, gdf)


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [["a"], "a", "b"])
@pytest.mark.parametrize("value", [-10, pd.DataFrame({"a": [-1, -2, -3]})])
def test_dataframe_setitem_columns(df, arg, value):
    gdf = DataFrame.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = DataFrame.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [["b", "c"]])
@pytest.mark.parametrize(
    "value", [pd.DataFrame({"0": [-1, -2, -3], "1": [-0, -10, -1]})]
)
def test_dataframe_setitem_new_columns(df, arg, value):
    gdf = DataFrame.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = DataFrame.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=True)


# we don't raise keyerror
# pandas raises it
# inconsistent with dataframe again
@pytest.mark.xfail()
def test_dataframe_setitem_scaler_keyerror():
    df = DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        df[["x"]] = 0


# set_item_series inconsistency
def test_series_setitem_index():
    df = pd.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )

    df["b"] = pd.Series(data=[12, 11, 10], index=[3, 2, 1])
    gdf = DataFrame(data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3])
    gdf["b"] = Series(data=[12, 11, 10], index=[3, 2, 1])
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_set_item(psr, arg):
    gsr = Series.from_pandas(psr)

    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.MultiIndex.from_frame(
                pd.DataFrame({"b": [3, 2, 1], "c": ["a", "b", "c"]})
            ),
        ),
        pd.DataFrame({"a": [1, 2, 3]}, index=["a", "b", "c"]),
    ],
)
def test_setitem_dataframe_series_inplace(df):
    pdf = df
    gdf = DataFrame.from_pandas(pdf)

    pdf["a"].replace(1, 500, inplace=True)
    gdf["a"].replace(1, 500, inplace=True)

    assert_eq(pdf, gdf)

    psr_a = pdf["a"]
    gsr_a = gdf["a"]

    psr_a.replace(500, 501, inplace=True)
    gsr_a.replace(500, 501, inplace=True)

    assert_eq(pdf, gdf)
