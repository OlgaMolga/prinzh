import pytest
import pandas as pd
from app import analyze_data

@pytest.fixture
def sample_df():
    #Тестовые данные
    return pd.DataFrame({
        "Sex": ["female", "female", "female", "male", "male", "male"],
        "Survived": [1, 0, 1, 0, 1, 0],
        "Age": [22, 85, 31, 87, 27, 70]
    })


def test_female_survived(sample_df): #Выжившие женщины
    result=analyze_data(sample_df, gender="Женский", status="Спасен")
    assert not result.empty
    assert result.iloc[0]["Sex"] == "Женский"
    assert result.iloc[0]["Survived"] == "Спасен"
    assert result.iloc[0]["min_age"] == 22
    assert result.iloc[0]["max_age"] == 31


def test_male_dead(sample_df): #Погибшие мужчины
    result = analyze_data(sample_df, gender="Мужской", status="Погиб")
    assert result.iloc[0]["Sex"] == "Мужской"
    assert result.iloc[0]["Survived"] == "Погиб"
    assert result.iloc[0]["min_age"] == 70
    assert result.iloc[0]["max_age"] == 87


def test_female_any_status(sample_df): #Фильтр только по полу
    result = analyze_data(sample_df, gender="Женский", status="Любой")
    assert set(result["Sex"]) == {"Женский"}
    assert result["min_age"].min() == 22
    assert result["max_age"].max() == 85