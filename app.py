import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# Cargar scaler y modelo
scaler = joblib.load('models/scaler.pkl')

modelo = load_model('models/modelo_completo.keras')  # Cambia por modelo_5_importantes.keras si quieres usar ese
modeloPequeno = load_model('models/modelo_5_importantes.keras')  # Cambia por modelo_5_importantes.keras si quieres usar ese


query_params = st.query_params
pagina = query_params.get("page", "inicio")

if pagina == "inicio":

    st.set_page_config(page_title="Predicción de Fallo Cardíaco",page_icon="🫀", layout="centered")
    
    st.title("🫀 Predicción de Fallo Cardíaco")
    st.markdown("Esta aplicación predice la probabilidad de fallo cardíaco a partir de datos clínicos.")
    # Opciones para ingresar los datos
    #opcion = st.radio("¿Cómo deseas ingresar los datos?", ("Formulario", "Archivo CSV"))
    opcion = st.radio("¿Qué modelo deseas usar?", ("Modelo grande", "Modelo pequeño (6 variables más importantes)"))
    
    if opcion == "Modelo grande":
        st.subheader("📋 ¡Usemos el modelo completo!")
        
        # Variables ingresadas por el usuario
        age = st.number_input("Edad", min_value=1, max_value=120)
    
        respuesta = st.selectbox("¿Tiene anemia?", ["No", "Sí"])
        anaemia = 1 if respuesta == "Sí" else 0
    
        creatinine_phosphokinase = st.number_input("Número de CPK (creatina fosfoquinasa) en la sangre por mcg/L", min_value=0)
    
        respuesta1 = st.selectbox("¿Tiene diabetes?", ["No", "Sí"])
        diabetes = 1 if respuesta1 == "Sí" else 0
    
        ejection_fraction = st.number_input("Porcentaje de sangre dejando el corazón en cada contracción", min_value=0, max_value=100)
    
        respuesta4 = st.selectbox("¿Tiene presión alta?", ["No", "Sí"])
        high_blood_pressure = 1 if respuesta4 == "Sí" else 0
    
        platelets = st.number_input("Número de plaquetas en la sangre (kiloplaquetas/mL)", min_value=0)
    
        serum_creatinine = st.number_input("Cantidad de creatinina sérica en la sangre (mg/dL)", min_value=0)
    
        serum_sodium = st.number_input("Cantidad de sodio sérico en la sangre (mEq/L)", min_value=0)
    
        respuesta2 = st.selectbox("Seleccione su sexo biológico", ["Femenino", "Masculino"])
        sex = 1 if respuesta2 == "Masculino" else 0
    
        respuesta3 = st.selectbox("¿Fuma?", ["No", "Sí"])
        smoking = 1 if respuesta3 == "Sí" else 0
    
        # Organizar variables
        
            # Variables numéricas
        num_vars = pd.DataFrame([{
            'age': age,
            'creatinine_phosphokinase': creatinine_phosphokinase,
            'ejection_fraction': ejection_fraction,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium
        }])
        
        # Variables binarias
        binary_vars = pd.DataFrame([{
            "anaemia": anaemia,
            "diabetes": diabetes,
            "high_blood_pressure": high_blood_pressure,
            "sex": sex,
            "smoking": smoking
        }])
        
        # Escalar las variables numéricas
        num_scaled = scaler.transform(num_vars)
        num_scaled_df = pd.DataFrame(num_scaled, columns=num_vars.columns)
        
        # Combinar todas las variables en un único DataFrame
        combined_df = pd.concat([num_scaled_df, binary_vars], axis=1)
        
        # Reordenar las columnas en el orden esperado por el modelo
        ordered_columns = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking'
        ]
        
        # Variable final de entrada
        final_input = combined_df[ordered_columns].values
        
        
        # Predicción
        if st.button("Predecir"):
            pred = modelo.predict(final_input)
            st.success(f"🧠 Probabilidad de fallo cardíaco: {pred[0][0] * 100:.2f}%")
    
        st.markdown("Recuerda que no es una aplicación médica, si sientes alguna complicación acude a tu médico.")
    
    
    
    
    # Opción para cargar archivo CSV
    #elif opcion == "Archivo CSV":
    #    st.subheader("📁 Cargar archivo CSV")
    #    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        
    #    if archivo:
    #        df = pd.read_csv(archivo)
    
            # Identificar columnas binarias y numéricas
    #        columnas_binarias = ["anaemia", "high_blood_pressure"]
    #        columnas_numericas = [col for col in df.columns if col not in columnas_binarias]
    
            # Escalar numéricas
    #        datos_numericos = scaler.transform(df[columnas_numericas])
    
            # Unir con las binarias (sin escalar)
    #        datos_finales = np.concatenate([datos_numericos, df[columnas_binarias].values], axis=1)
    
            # Predecir
    #        predicciones = modelo.predict(datos_finales)
    
    #        df_resultado = df.copy()
    #        df_resultado["Probabilidad_Fallo_Cardiaco"] = (predicciones * 100).round(2)
    
    #        st.write("🧾 Resultados:")
    #        st.dataframe(df_resultado)
    
    if opcion == "Modelo pequeño (6 variables más importantes)":
        st.subheader("📋 ¡Usemos las variables más importantes!")
        
        # Variables ingresadas por el usuario
        age = st.number_input("Edad", min_value=1, max_value=120)
    
        creatinine_phosphokinase = st.number_input("Número de CPK (creatina fosfoquinasa) en la sangre por mcg/L", min_value=0)
    
        ejection_fraction = st.number_input("Porcentaje de sangre dejando el corazón en cada contracción", min_value=0, max_value=100)
    
        platelets = st.number_input("Número de plaquetas en la sangre (kiloplaquetas/mL)", min_value=0)
    
        serum_creatinine = st.number_input("Cantidad de creatinina sérica en la sangre (mg/dL)", min_value=0)
    
        serum_sodium = st.number_input("Cantidad de sodio sérico en la sangre (mEq/L)", min_value=0)
    
        # Organizar variables
        
            # Variables numéricas
        num_vars = pd.DataFrame([{
            'age': age,
            'creatinine_phosphokinase': creatinine_phosphokinase,
            'ejection_fraction': ejection_fraction,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium
        }])
        
        # Escalar las variables numéricas
        num_scaled = scaler.transform(num_vars)
        num_scaled_df = pd.DataFrame(num_scaled, columns=num_vars.columns)
        
        # Combinar todas las variables en un único DataFrame
        combined_df = pd.concat([num_scaled_df], axis=1)
        
        # Reordenar las columnas en el orden esperado por el modelo
        ordered_columns = [
            'age', 'creatinine_phosphokinase','ejection_fraction','platelets', 'serum_creatinine', 'serum_sodium'
        ]
        
        # Variable final de entrada
        final_input = combined_df[ordered_columns].values
        
        
        # Predicción
        if st.button("Predecir"):
            pred = modeloPequeno.predict(final_input)
            st.success(f"🧠 Probabilidad de fallo cardíaco: {pred[0][0] * 100:.2f}%")
    
        st.markdown("Recuerda que no es una aplicación médica, si sientes alguna complicación acude a tu médico.")

    if st.button("Haz clic aquí para saber cómo hicimos la predicción"):
       st.query_params["page"] = "comoLoHicimos" 
       st.rerun()



elif pagina == "comoLoHicimos":
    st.title("Así lo hicimos")
    st.write(
    "Al escoger el problema y el [conjunto de datos](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data), "
    "nos propusimos predecir el riesgo de falla cardíaca en pacientes.")

    st.subheader("📊 Preprocesamiento de los datos")

    st.write(
    "Comenzamos analizando las distribuciones de las variables numéricas. Como ninguna seguía una distribución específica, "
    "optamos por utilizar la **mediana** para imputar los valores faltantes. Esto nos permitió usar un valor *central* que no se ve afectado por valores atípicos.")

    st.subheader("📋 Clustering no supervisado - K-Means")
    st.write(
    "Para explorar posibles patrones ocultos en los datos, aplicamos un método de clustering no supervisado utilizando **K-Means**. "
    "Sin embargo, al utilizar el método del codo, no encontramos un número óptimo de grupos claro, por lo que decidimos no continuar por esta vía.")

    image = Image.open("images/kmeans.png")
    st.image(image, caption="Método del codo para determinar el numero de clústers optimo")

    st.subheader("🌳 Clasificación supervisada - Random Forest")
    st.write(
    "Probamos varios algoritmos de clasificación, entre ellos **Random Forest**. Aunque los resultados no fueron muy prometedores "
    "(con una exactitud inferior al 60%), este modelo nos permitió identificar la **importancia de las características**, lo cual fue clave para "
    "desarrollar versiones más eficientes del modelo.") 

    image = Image.open("images/importanciaDeCaracteristicasRM.png")
    st.image(image, caption="Importancia de caracteristicas usando random forest")

    st.subheader("💻 Clasificación supervisada - Redes neuronales")
    st.write(
    "Con una red neuronal con dos capas ocultas de 64 y 32 neuronas respectivamente, logramos mejorar el rendimiento, alcanzando una **exactitud del 67%**")

    image = Image.open("images/matrizCompleto.png")
    st.image(image, caption="Matriz de confusión, modelo completo")

    image = Image.open("images/perdidaEntrenamientoCompleto.png")
    st.image(image, caption="Perdida por épocas, modelo completo")

    st.subheader("🔮 Clasificación supervisada - Redes neuronales (6 características más importantes)")
    st.write(
    "Finalmente, entrenamos una red neuronal utilizando únicamente las **seis características más relevantes**. Este enfoque resultó ser el más efectivo: "
    "alcanzamos una **exactitud del 71%**, disminuyendo el tiempo de inferencia en un **20%** respecto al original, "
    "sin sacrificar precisión.")

    image = Image.open("images/matriz6masimportantes.png")
    st.image(image, caption="Matriz de confusión, modelo con 6 variables más importantes")

    image = Image.open("images/perdida6Importantes.png")
    st.image(image, caption="Perdida por épocas, modelo con 6 variables más importantes")


    if st.button("Haz clic aquí para volver a la página inicial"):
       st.query_params["page"] = "inicio" 
       st.rerun()
