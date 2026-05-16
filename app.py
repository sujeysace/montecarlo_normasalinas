import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Simulación Monte Carlo Actuarial",
    page_icon="📊",
    layout="wide"
)

st.title("Simulación Monte Carlo Actuarial")
st.write(
    "Aplicación basada en el notebook de simulación Monte Carlo para estimar pérdidas agregadas, "
    "prima pura, VaR, TVaR, insuficiencia de prima y un modelo predictivo simple."
)

with st.sidebar:
    st.header("Parámetros del modelo")
    n_simulaciones = st.slider("Número de simulaciones", 1000, 50000, 10000, step=1000)
    lambda_frecuencia = st.slider("Frecuencia promedio esperada", 10, 200, 80, step=5)
    media_log_severidad = st.slider("Media logarítmica de severidad", 5.0, 12.0, 8.5, step=0.1)
    sigma_log_severidad = st.slider("Sigma logarítmica de severidad", 0.1, 2.0, 0.9, step=0.1)
    nivel_confianza = st.slider("Nivel de confianza", 0.80, 0.99, 0.95, step=0.01)
    margen_seguridad = st.slider("Margen de seguridad", 0.00, 0.50, 0.15, step=0.01)
    semilla = st.number_input("Semilla aleatoria", value=2026, step=1)

@st.cache_data(show_spinner=False)
def simular_monte_carlo(n_simulaciones, lambda_frecuencia, media_log_severidad, sigma_log_severidad, nivel_confianza, margen_seguridad, semilla):
    np.random.seed(int(semilla))

    frecuencia = np.random.poisson(lam=lambda_frecuencia, size=n_simulaciones)
    frecuencia_promedio = np.mean(frecuencia)
    frecuencia_desviacion = np.std(frecuencia)

    total_siniestros = int(np.sum(frecuencia))
    severidades = np.random.lognormal(
        mean=media_log_severidad,
        sigma=sigma_log_severidad,
        size=total_siniestros
    )

    severidad_promedio = np.mean(severidades)
    severidad_mediana = np.median(severidades)
    severidad_p95 = np.percentile(severidades, 95)

    perdidas_agregadas = []
    posicion = 0
    for n in frecuencia:
        severidades_escenario = severidades[posicion:posicion + n]
        perdidas_agregadas.append(np.sum(severidades_escenario))
        posicion += n

    perdidas_agregadas = np.array(perdidas_agregadas)
    perdida_promedio = np.mean(perdidas_agregadas)
    perdida_mediana = np.median(perdidas_agregadas)
    prima_pura = perdida_promedio

    var = np.percentile(perdidas_agregadas, nivel_confianza * 100)
    perdidas_extremas = perdidas_agregadas[perdidas_agregadas >= var]
    tvar = np.mean(perdidas_extremas)
    margen_riesgo = tvar - prima_pura

    prima_con_margen = prima_pura * (1 + margen_seguridad)
    probabilidad_insuficiencia = np.mean(perdidas_agregadas > prima_con_margen)

    datos = pd.DataFrame({
        "frecuencia": frecuencia,
        "perdida_agregada": perdidas_agregadas
    })

    X = datos[["frecuencia"]]
    y = datos["perdida_agregada"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2026)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    resumen = pd.DataFrame({
        "Indicador": [
            "Frecuencia promedio",
            "Desviación estándar frecuencia",
            "Severidad promedio",
            "Severidad mediana",
            "Percentil 95 severidad",
            "Prima pura",
            f"VaR {int(nivel_confianza * 100)}%",
            f"TVaR {int(nivel_confianza * 100)}%",
            "Margen de riesgo TVaR - Prima pura",
            f"Prima con margen {int(margen_seguridad * 100)}%",
            "Probabilidad de insuficiencia",
            "MAE modelo ML",
            "R2 modelo ML"
        ],
        "Valor": [
            frecuencia_promedio,
            frecuencia_desviacion,
            severidad_promedio,
            severidad_mediana,
            severidad_p95,
            prima_pura,
            var,
            tvar,
            margen_riesgo,
            prima_con_margen,
            probabilidad_insuficiencia,
            mae,
            r2
        ]
    })

    return datos, severidades, resumen, {
        "frecuencia": frecuencia,
        "perdidas_agregadas": perdidas_agregadas,
        "prima_pura": prima_pura,
        "perdida_mediana": perdida_mediana,
        "var": var,
        "tvar": tvar,
        "prima_con_margen": prima_con_margen,
        "nivel_confianza": nivel_confianza,
        "y_test": y_test,
        "predicciones": predicciones
    }

with st.spinner("Ejecutando simulación Monte Carlo..."):
    datos, severidades, resumen, resultados = simular_monte_carlo(
        n_simulaciones,
        lambda_frecuencia,
        media_log_severidad,
        sigma_log_severidad,
        nivel_confianza,
        margen_seguridad,
        semilla
    )

st.subheader("Indicadores principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prima pura", f"{resultados['prima_pura']:,.2f}")
col2.metric(f"VaR {int(nivel_confianza * 100)}%", f"{resultados['var']:,.2f}")
col3.metric(f"TVaR {int(nivel_confianza * 100)}%", f"{resultados['tvar']:,.2f}")
col4.metric("Prima con margen", f"{resultados['prima_con_margen']:,.2f}")

st.subheader("Tabla resumen")
st.dataframe(resumen.style.format({"Valor": "{:,.4f}"}), use_container_width=True)

st.subheader("Dataset simulado")
st.dataframe(datos.head(100), use_container_width=True)
st.download_button(
    "Descargar dataset simulado en CSV",
    data=datos.to_csv(index=False).encode("utf-8"),
    file_name="dataset_monte_carlo_actuarial.csv",
    mime="text/csv"
)

st.subheader("Visualizaciones")
tab1, tab2, tab3, tab4 = st.tabs([
    "Frecuencia", "Severidad", "Pérdida agregada", "VaR y TVaR"
])

with tab1:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(resultados["frecuencia"], bins=30, edgecolor="black")
    ax.axvline(np.mean(resultados["frecuencia"]), linestyle="--", label="Promedio simulado")
    ax.set_title("Distribución simulada de la frecuencia anual de siniestros")
    ax.set_xlabel("Número de siniestros en el año")
    ax.set_ylabel("Cantidad de escenarios simulados")
    ax.legend()
    st.pyplot(fig)

with tab2:
    limite = np.percentile(severidades, 99)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(severidades[severidades <= limite], bins=40, edgecolor="black")
    ax.axvline(np.median(severidades), linestyle="--", label="Mediana")
    ax.axvline(np.mean(severidades), linestyle="-.", label="Promedio")
    ax.set_title("Distribución simulada de severidades individuales")
    ax.set_xlabel("Monto individual del siniestro")
    ax.set_ylabel("Cantidad de siniestros simulados")
    ax.legend()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(resultados["perdidas_agregadas"], bins=40, edgecolor="black")
    ax.axvline(resultados["prima_pura"], linestyle="--", label="Pérdida promedio / prima pura")
    ax.axvline(resultados["perdida_mediana"], linestyle="-.", label="Mediana")
    ax.set_title("Distribución simulada de la pérdida agregada anual")
    ax.set_xlabel("Pérdida agregada anual")
    ax.set_ylabel("Cantidad de escenarios simulados")
    ax.ticklabel_format(style="plain", axis="x")
    ax.legend()
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(resultados["perdidas_agregadas"], bins=40, edgecolor="black")
    ax.axvline(resultados["prima_pura"], linestyle="--", label="Prima pura")
    ax.axvline(resultados["var"], linestyle="-.", label="VaR")
    ax.axvline(resultados["tvar"], linestyle=":", label="TVaR")
    ax.axvline(resultados["prima_con_margen"], linestyle="--", label="Prima con margen")
    ax.set_title("Distribución de pérdida agregada con VaR, TVaR y prima con margen")
    ax.set_xlabel("Pérdida agregada anual")
    ax.set_ylabel("Cantidad de escenarios simulados")
    ax.ticklabel_format(style="plain", axis="x")
    ax.legend()
    st.pyplot(fig)

st.subheader("Modelo predictivo simple")
col_a, col_b = st.columns(2)
mae = resumen.loc[resumen["Indicador"] == "MAE modelo ML", "Valor"].iloc[0]
r2 = resumen.loc[resumen["Indicador"] == "R2 modelo ML", "Valor"].iloc[0]
col_a.metric("MAE", f"{mae:,.2f}")
col_b.metric("R²", f"{r2:,.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
y_test = resultados["y_test"]
predicciones = resultados["predicciones"]
ax.scatter(y_test, predicciones, alpha=0.35)
minimo = min(y_test.min(), predicciones.min())
maximo = max(y_test.max(), predicciones.max())
ax.plot([minimo, maximo], [minimo, maximo], linestyle="--", label="Predicción perfecta")
ax.set_title("Pérdida real versus pérdida predicha")
ax.set_xlabel("Pérdida real")
ax.set_ylabel("Pérdida predicha")
ax.ticklabel_format(style="plain", axis="both")
ax.legend()
st.pyplot(fig)

st.info(
    "Interpretación: la prima pura representa la pérdida promedio esperada. El VaR muestra el umbral de pérdida "
    "para el nivel de confianza seleccionado, mientras que el TVaR resume el promedio de los escenarios más severos."
)
