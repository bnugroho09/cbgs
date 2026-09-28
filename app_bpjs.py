import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.title("Multi-Output Classifier Demo")

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file:

    sampel = pd.read_excel(uploaded_file)

    sampel = (
        sampel[['INACBG', 'UMUR_TAHUN', 'PTD', 'SEX', 'DPJP', 'DIAGLIST', 'PROCLIST']].copy()
        .assign(
            diag = lambda x: x['DIAGLIST'].str.split(';').str[0],
            cmg = lambda x: x['INACBG'].str[0],
            grp = lambda x: x['INACBG'].str[2],
            typ = lambda x: x['INACBG'].str[4:6],
            svr = lambda x: x['INACBG'].str[7:]
        )
    )

    encoders = {}

    for col in ['DPJP','diag','cmg','grp','typ','svr']:
        le = LabelEncoder()
        sampel[col] = le.fit_transform(sampel[col])
        encoders[col] = le   # save the fitted encoder

        # Fitur dan target
    X2 = sampel[['UMUR_TAHUN','PTD','SEX','DPJP','diag']]
    y2 = sampel[['cmg','grp','typ','svr']]

    # Split data
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Train model
    model2 = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    model2.fit(X_train2, y_train2)

    # Evaluasi
    y_pred2 = model2.predict(X_test2)
    results = {}
    for i, col in enumerate(y_test2.columns):
        results[col] = accuracy_score(y_test2[col], y_pred2[:, i])

    st.subheader("Accuracy per Label")
    acc_df = pd.DataFrame(list(results.items()), columns=['Label','Accuracy'])
    st.dataframe(acc_df)
    st.bar_chart(acc_df.set_index('Label'))

    # -------------------------------
    # Form input manual
    st.subheader("Predict for a Single Patient")

    umur = st.number_input("Umur (tahun)", min_value=0, max_value=120, value=30)
    ptd = st.selectbox("PTD", options=sorted(sampel['PTD'].unique()))
    sex = st.selectbox("Jenis Kelamin", options=sorted(sampel['SEX'].unique()))
    dpjp = st.selectbox("DPJP", options=sorted(sampel['DPJP'].unique()))
    diag = st.text_input("Diagnosis (kode ICD)", value="E11.9")

    if st.button("Predict"):
        input_df = pd.DataFrame([[umur, ptd, sex, dpjp, diag]], columns=X2.columns)
        pred = model2.predict(input_df)
        pred_df = pd.DataFrame(pred, columns=y2.columns)
        st.write("Prediksi hasil:")
        st.dataframe(pred_df)
