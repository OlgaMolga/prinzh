import streamlit as st
import pandas as pd

st.set_page_config(page_title="Анализ пассажиров Титаника", page_icon="🚢", layout="centered")

titanic = pd.read_csv('https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv', index_col='PassengerId')

st.image("https://i.pinimg.com/originals/45/de/d3/45ded390044f2f5944b18097378bd176.jpg?nii=t", width="stretch")
st.title("Найти диапазон возрастов (min и max) пассажиров, указав пол и спасен/погиб.")


gender = st.selectbox("Пол:", ("Любой", "Мужской", "Женский"))
status = st.selectbox("Статус (Survived):", ("Любой", "Спасен", "Погиб"))

def analyze_data(df: pd.DataFrame, gender: str = "Любой", status: str = "Любой") -> pd.DataFrame:
    df_filtered = df.copy()

    # фильтр по полу
    if gender == "Мужской":
        df_filtered = df_filtered[df_filtered["Sex"] == "male"]
    elif gender == "Женский":
        df_filtered = df_filtered[df_filtered["Sex"] == "female"]

    # фильтр по статусу
    if status == "Спасен":
        df_filtered = df_filtered[df_filtered["Survived"] == 1]
    elif status == "Погиб":
        df_filtered = df_filtered[df_filtered["Survived"] == 0]

    if df_filtered.empty:
        return pd.DataFrame()

    result = df_filtered.groupby(['Sex', 'Survived']).agg(
        min_age=('Age', 'min'),
        max_age=('Age', 'max')
    ).reset_index()

    result['Survived'] = result['Survived'].map({0: 'Погиб', 1: 'Спасен'})
    result['Sex'] = result['Sex'].map({'male': 'Мужской', 'female': 'Женский'})

    return result


result = analyze_data(titanic, gender, status)

if not result.empty:
    st.subheader("Результаты анализа:")
    st.table(result)
else:
    st.warning("По выбранным параметрам нет данных.")


