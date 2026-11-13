from src.config import FEATURES
from src.data import add_binary_target, load_wine_data


def test_load_wine_data_merges_both_datasets():
    df = load_wine_data()
    assert len(df) == 1599 + 4898
    assert set(df["type"]) == {"red", "white"}
    assert all(col in df.columns for col in FEATURES + ["quality"])


def test_binary_target_threshold():
    df = add_binary_target(load_wine_data())
    assert set(df["good"]) == {0, 1}
    assert (df.loc[df["quality"] >= 7, "good"] == 1).all()
    assert (df.loc[df["quality"] < 7, "good"] == 0).all()
