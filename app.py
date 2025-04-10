import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import zipfile
import os
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("⚽ Modelo de Predicción de Resultados de Partidos")

# Subir archivo
archivo = st.file_uploader("Sube tu archivo partidos.csv", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    # Verifica columnas necesarias
    if 'goles_local' not in df.columns or 'goles_visitante' not in df.columns:
        st.error("❌ El archivo debe tener las columnas 'goles_local' y 'goles_visitante'.")
        st.stop()

    st.subheader("📋 Vista previa del archivo")
    st.dataframe(df.head())

    # Crear columna 'resultado'
    df['resultado'] = df.apply(lambda row: 1 if row['goles_local'] > row['goles_visitante']
                               else (-1 if row['goles_local'] < row['goles_visitante'] else 0), axis=1)

    # Features y target
    features = ['posesión_local', 'tiros_local', 'tiros_visitante', 'tarjetas_local', 'tarjetas_visitante']
    if not all(col in df.columns for col in features):
        st.error("❌ Faltan algunas columnas necesarias para el modelo.")
        st.stop()

    X = df[features]
    y = df['resultado']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Reporte
    report = classification_report(y_test, y_pred, output_dict=False)
    st.subheader("📊 Reporte de Clasificación")
    st.text(report)

    # Importancia
    importancias = model.feature_importances_
    importancia_dict = dict(zip(features, importancias))
    st.subheader("🔥 Importancia de Características")
    st.bar_chart(pd.Series(importancia_dict))

    # Gráfico de importancia
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(importancia_dict.values()), y=list(importancia_dict.keys()), palette="viridis")
    plt.title("Importancia de las características")
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.tight_layout()
    img_buf1 = io.BytesIO()
    plt.savefig(img_buf1, format="png")
    st.image(img_buf1)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Local", "Empate", "Visitante"],
                yticklabels=["Local", "Empate", "Visitante"])
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    img_buf2 = io.BytesIO()
    plt.savefig(img_buf2, format="png")
    st.image(img_buf2)

    # Comparación Reales vs Predichos
    valores_clases = [1, 0, -1]
    etiquetas = ["Local", "Empate", "Visitante"]
    reales = [sum(y_test == c) for c in valores_clases]
    predichas = [sum(y_pred == c) for c in valores_clases]
    x = np.arange(len(etiquetas))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, reales, width, label='Reales', color='steelblue')
    plt.bar(x + width/2, predichas, width, label='Predichas', color='orange')
    plt.xticks(x, etiquetas)
    plt.legend()
    plt.title("Reales vs. Predichos")
    plt.tight_layout()
    img_buf3 = io.BytesIO()
    plt.savefig(img_buf3, format="png")
    st.image(img_buf3)

    # Crear CSV con predicciones
    df_test = X_test.copy()
    df_test['real'] = y_test.values
    df_test['prediccion'] = y_pred
    df_test = df_test.join(df[['goles_local', 'goles_visitante']].iloc[df_test.index])
    csv_buf = io.StringIO()
    df_test.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')

    # Crear TXT de reporte
    txt_buf = io.StringIO()
    txt_buf.write("=== Reporte de Clasificación ===\n")
    txt_buf.write(report)
    txt_buf.write("\n\n=== Importancia de Características ===\n")
    for f, imp in importancia_dict.items():
        txt_buf.write(f"{f}: {imp:.4f}\n")
    txt_bytes = txt_buf.getvalue().encode('utf-8')

    # Crear ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zipf:
        zipf.writestr("partidos_con_predicciones.csv", csv_bytes)
        zipf.writestr("resultados_modelo.txt", txt_bytes)
        zipf.writestr("importancia_caracteristicas.png", img_buf1.getvalue())
        zipf.writestr("matriz_confusion.png", img_buf2.getvalue())
        zipf.writestr("comparacion_reales_vs_predichos.png", img_buf3.getvalue())

    # Botón para descargar
    st.subheader("📦 Descargar resultados")
    st.download_button("Descargar ZIP", zip_buf.getvalue(), file_name="resultados_modelo.zip", mime="application/zip")
