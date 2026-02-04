import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Proyecto Final")
st.sidebar.title("Módulos")

modulo = st.sidebar.selectbox("Seleccione un módulo",["Módulo 1: Home","Módulo 2: Carga del dataset"])

if modulo == "Módulo 1: Home":
    
    st.title("Caso de Estudio Telco Customer Churn")
    
    st.divider()
    
    st.header("Breve descripción del objetivo del análisis")
    st.write("""
    Este proyecto busca llevar a cabo un Análisis Exploratorio de Datos (EDA) exhaustivo del dataset TelcoCustomerChurn.csv. 
    Se procederá a examinar, depurar, procesar y visualizar los datos para descubrir patrones asociados a la fuga de clientes, 
    determinar los factores que impactan en este fenómeno y aplicar métodos analíticos avanzados para optimizar la toma de 
    decisiones estratégicas ante el elevado churn rate.         
    """)
    
    st.divider()
    
    st.subheader("Datos del Autor")
    st.info("**Nombre completo:** Fiorella del Pilar, Sánchez Villacorta")
    st.info("**Curso:** Especializacion Python Analytics")
    st.info("**Año:** 2026")

    st.header("Breve explicación del dataset")
    st.write("""
    El dataset TelcoCustomerChurn.csv almacena información detallada sobre la base de clientes, incluyendo servicios contratados, facturación mensual, antigüedad y estatus actual en la empresa. 
    En el contexto de la pandemia de COVID-19, la compañía experimentó un incremento del 0.5% en su tasa de abandono, pasando del 2% al 2.5% en el último mes. Dado que el costo de adquisición de un nuevo cliente es entre 6 y 7 veces superior al de retener a uno existente, resulta fundamental analizar los datos históricos para identificar patrones de comportamiento y mejorar la retención. 
    El objetivo de este proyecto es analizar los datos para entender las causas subyacentes de la fuga de clientes, utilizando un enfoque exploratorio y visual para informar estrategias efectivas de retención.
    """)

    st.header("Tecnologías utilizadas")
    st.write("Python, Pandas, Streamlit, etc.")

elif modulo == "Módulo 2: Carga del dataset":

    # Cargar dataset
    uploaded_file = st.file_uploader("Sube un archivo .csv o .xlsx", type=["csv"])

    # Verificar si el archivo fue cargado
    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
    st.success("Archivo cargado correctamente")

    #Vista previa
    st.write("### Vista previa del dataset")
    st.write(df.head())
    
    #Dimensiones
    st.write(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas")

    st.title("Análisis Exploratorio de Datos (EDA)")

    ############################################################ Ítem 1 ############################################################
    st.subheader("Ítem 1: Información general del dataset")

    df_info = pd.DataFrame({
        'Tipo': df.dtypes,
        'No Nulos': df.count(),
        'Nulos': df.isnull().sum()
    })

    st.dataframe(df_info)

    ############################################################ Ítem 2 ############################################################
    st.subheader("Ítem 2: Clasificación de variables")

    # Definición de función personalizada
    def clasificar_columnas(dataframe):
        num_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
        cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        return num_cols, cat_cols

    # Uso de la función
    numericas, categoricas = clasificar_columnas(df)

    # Mostrar resultados con conteo
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Numéricas ({len(numericas)})")
        st.write(numericas)

    with col2:
        st.subheader(f"Categóricas ({len(categoricas)})")
        st.write(categoricas)

    ############################################################ Ítem 3 ############################################################
    st.subheader("Ítem 3: Estadísticas descriptivas")

    st.dataframe(df.describe())
    st.info("Interpretación: La media indica el promedio, la mediana (50%) el punto central, y la desviación estándar (std) la dispersión de los datos.")

    ############################################################ Ítem 4 ############################################################
    st.subheader("Ítem 4: Análisis de valores faltantes")

    # Contamos los valores nulos por columna
    missing_counts = df.isnull().sum()

    # Calculamos el porcentaje de valores nulos por columna
    missing_percent = df.isnull().mean() * 100

    # Creamos un dataframe resumen
    missing_summary = pd.DataFrame({
        "Valores Nulos": missing_counts,
        "% de Nulos": missing_percent.round(2)
    })

    # Mostramos el resumen
    missing_summary

    st.subheader("Visualización de Valores Nulos")

    # Creamos el gráfico
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title("Mapa de calor de valores nulos")

    # Mostramos el gráfico
    st.pyplot(fig)

    st.info("""
    **Discusión:** El mapa de calor nos permite identificar los Valores Nulos con lineas representativas.
    El gráfico mostrado no se visualizan tales lineas representativas por lo cual podemos afirmar que no existen Valores Nulos.
    """)

    ############################################################ Ítem 5 ############################################################
    st.subheader("Ítem 5: Distribución de variables numéricas")

    # Usamos la lista 'numericas' que definimos arriba
    if numericas:
        # Selector
        var_seleccionada = st.selectbox("Selecciona una variable para ver su distribución:", numericas)
        
        # Creamos la figura de Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Histograma con curva de densidad (KDE)
        sns.histplot(df[var_seleccionada], kde=True, color='skyblue', ax=ax)
        ax.set_title(f"Distribución de {var_seleccionada}")
        ax.set_xlabel(var_seleccionada)
        ax.set_ylabel("Frecuencia")
        
        # Mostramos el gráfico
        st.pyplot(fig)
        
        # Interpretación Visual
        st.subheader("Interpretación Visual")
        
        # Lógica sencilla para ayudar a la interpretación
        if df[var_seleccionada].skew() > 1:
            st.write(f"La variable **{var_seleccionada}** presenta una cola larga a la derecha (sesgo positivo).")
        elif df[var_seleccionada].skew() < -1:
            st.write(f"La variable **{var_seleccionada}** presenta una cola larga a la izquierda (sesgo negativo).")
        else:
            st.write(f"La variable **{var_seleccionada}** sigue una distribución aproximadamente simétrica.")

    ############################################################ Ítem 6 ############################################################
    st.subheader("Ítem 6: Análisis de variables categóricas")

    # Usamos la lista de 'categóricas' que definimos arriba
    if categoricas:
        # Selector
        var_cat = st.selectbox("Selecciona una variable categórica:", categoricas)
        
        # Cálculo de Conteos y Proporciones
        conteo = df[var_cat].value_counts()
        proporcion = df[var_cat].value_counts(normalize=True) * 100
        
        df_resumen = pd.DataFrame({
            'Conteo': conteo,
            'Proporción (%)': proporcion.map('{:.2f}%'.format)
        })
        
        st.write("Tabla de Frecuencias y Proporciones")
        st.dataframe(df_resumen)

        # Mostramos los gráficos (Barras y Proporciones)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Gráfico de Barras")
            fig_bar, ax_bar = plt.subplots()
            sns.countplot(data=df, x=var_cat, palette='viridis', ax=ax_bar)
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

        with col2:
            st.write("Gráfico de Proporciones")
            fig_pie, ax_pie = plt.subplots()
            conteo.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax_pie, cmap='Pastel1')
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)
    ############################################################ Ítem 7 ############################################################
    st.subheader("Ítem 7: Análisis bivariado - MonthlyCharges(numérico) vs Churn(categórico)")

    # 1. Crear la figura para el Boxplot
    st.write("### Distribución de Cargos Mensuales por Churn")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot para comparar distribuciones
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='magma', ax=ax)
    ax.set_title("Comparativa de MonthlyCharges según estado de Churn")
    ax.set_xlabel("Churn (Fuga de Clientes)")
    ax.set_ylabel("Cargos Mensuales (MonthlyCharges)")

    # Mostrar en Streamlit
    st.pyplot(fig)

    # 2. Resumen Estadístico Agrupado
    st.write("### Comparativa de Promedios")
    resumen = df.groupby('Churn')['MonthlyCharges'].agg(['mean', 'median', 'std']).reset_index()

    # Renombrar columnas para que se vea pro
    resumen.columns = ['Estado de Churn', 'Media', 'Mediana', 'Desv. Estándar']
    st.dataframe(resumen.style.highlight_max(axis=0, subset=['Media'], color='#9efc9e'))

    # 3. Interpretación Senior
    st.subheader("Interpretación Técnica")
    media_no = resumen.loc[resumen['Estado de Churn'] == 'No', 'Media'].values[0]
    media_yes = resumen.loc[resumen['Estado de Churn'] == 'Yes', 'Media'].values[0]

    if media_yes > media_no:
        st.warning(f"Se observa que los clientes que abandonan (Yes) tienen un cargo promedio mayor (${media_yes:.2f}) en comparación a los que se quedan (${media_no:.2f}).")
    else:
        st.success("Los cargos mensuales no parecen ser el factor principal de fuga en este segmento.")

    ############################################################ Ítem 8 ############################################################
    st.subheader("Ítem 8: Análisis bivariado - InternetService(categórico) vs Churn(categórico)")

    # 1. Crear Tabla de Contingencia (Frecuencias)
    st.write("### Tabla de Contingencia: Frecuencia de Abandono por Tipo de Internet")
    tabla_frec = pd.crosstab(df['InternetService'], df['Churn'])
    st.dataframe(tabla_frec)

    # 2. Crear Tabla de Proporciones (Porcentajes)
    st.write("### Proporciones (%) por Servicio de Internet")
    # normalize='index' permite ver el % de Churn dentro de cada tipo de servicio
    tabla_prop = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    st.dataframe(tabla_prop.style.format("{:.2f}%").highlight_max(axis=1, color='#ffcccb'))

    # 3. Visualización: Gráfico de Barras Apiladas
    st.write("### Distribución Porcentual de Churn")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficamos la tabla de proporciones
    tabla_prop.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF5722'], ax=ax)

    plt.title("Proporción de Churn según Servicio de Internet")
    plt.xlabel("Servicio de Internet")
    plt.ylabel("Porcentaje (%)")
    plt.legend(title='Churn', loc='upper right')
    plt.xticks(rotation=0)

    # Renderizar en Streamlit
    st.pyplot(fig)

    # 4. Interpretación Técnica
    st.subheader("Interpretación")
    st.info("""
    Este gráfico identifica qué servicios de internet son más críticos para la retención. 
    Si una barra tiene un segmento rojo (Churn=Yes) mucho más grande que las otras, 
    ese servicio específico está experimentando mayores problemas de lealtad de clientes.
    """)

    ############################################################ Ítem 9 ############################################################
    st.subheader("Ítem 9: Análisis dinámico basado en parámetros")

    # 1. Selección de múltiples columnas numéricas (Multiselect)
    st.subheader("Visualización Comparativa de Variables")
    columnas_analisis = st.multiselect(
        "Selecciona las variables numéricas para comparar:",
        options=numericas,
        default=numericas[:2] if len(numericas) >= 2 else numericas
    )

    if columnas_analisis:
        # Mostramos estadísticas dinámicas
        st.write("### Resumen Estadístico Dinámico")
        st.dataframe(df[columnas_analisis].describe())

        # Gráfico de Boxplots comparativos
        fig9_1, ax9_1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[columnas_analisis], orient="h", palette="Set2", ax=ax9_1)

        ax9_1.set_title("Distribución Comparativa")
        st.pyplot(fig9_1)
    else:
        st.warning("Por favor, selecciona al menos una columna numérica.")

    st.divider()

    # 2. Análisis Dinámico de Categorías (Selectbox)
    st.subheader("Análisis Dinámico de Segmentación")
    col_segmento = st.selectbox("Elige una variable categórica para segmentar:", categoricas)
    col_valor = st.selectbox("Elige una variable numérica para promediar:", numericas)

    if col_segmento and col_valor:
        # Cálculo dinámico
        analisis_segmento = df.groupby(col_segmento)[col_valor].mean().sort_values(ascending=False).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### Promedio de {col_valor} por {col_segmento}")
            st.dataframe(analisis_segmento.style.background_gradient(cmap='Blues'))
            
        with col2:
            fig9_2, ax9_2 = plt.subplots()
            sns.barplot(data=analisis_segmento, x=col_segmento, y=col_valor, ax=ax9_2, palette='viridis')
            plt.xticks(rotation=45)
            st.pyplot(fig9_2)

    ############################################################ Ítem 10 ############################################################
    st.subheader("Ítem 10: Hallazgos clave")

    # 1. Dashboard de Métricas Principales
    st.subheader("📊 Resumen General del Dataset")

    col1, col2, col3, col4 = st.columns(4)

    # Cálculos para los KPIs
    total_clientes = len(df)
    tasa_churn = (df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    cargo_promedio = df['MonthlyCharges'].mean()
    servicio_top = df['InternetService'].mode()[0] # El servicio más común

    col1.metric("Total Clientes", f"{total_clientes}")
    col2.metric("Tasa de Churn", f"{tasa_churn:.1f}%")
    col3.metric("Cargo Mensual Prom.", f"${cargo_promedio:.2f}")
    col4.metric("Servicio Dominante", servicio_top)

    st.divider()

    # 2. Visualización Resumen: Relación Crítica
    st.write("### Comparativa Final: Cargos y Servicio vs. Abandono")

    fig10, ax10 = plt.subplots(figsize=(10, 5))
    # Graficamos el cargo promedio por Servicio de Internet, segmentado por Churn
    sns.barplot(data=df, x='InternetService', y='MonthlyCharges', hue='Churn', palette='viridis', ax=ax10)
    ax10.set_title("Impacto de MonthlyCharges y InternetService en el Churn")
    st.pyplot(fig10)

    # 3. Conclusiones Resumidas (Variables específicas)
    st.subheader("💡 Insights y Conclusiones")

    # Usamos una caja de texto profesional para resumir
    st.info(f"""
    Basado en el análisis exploratorio, se presentan los siguientes hallazgos sobre las variables clave:

    * **MonthlyCharges (Cargos Mensuales):** Se observa una relación directa entre los cargos elevados y la fuga de clientes. Los clientes con Churn presentan, en promedio, facturas más altas que los que permanecen en la empresa.
    * **Churn (Tasa de Abandono):** La tasa actual del {tasa_churn:.1f}% representa un punto crítico de atención, especialmente en segmentos con cargos superiores al promedio de ${cargo_promedio:.2f}.
    * **InternetService (Tipo de Servicio):** El servicio de **{servicio_top}** es el de mayor volumen, pero el análisis bivariado muestra que la distribución del Churn varía significativamente según el tipo de tecnología contratada.
    """)