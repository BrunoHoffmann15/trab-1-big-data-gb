import streamlit as st
import pandas as pd
import pickle

# Método para obter o valor numérico de GeneticRisk
def get_formated_genetic_risk(risk):
    map = {
        'Baixo': 0, 
        'Médio': 1, 
        'Alto': 2
    }

    return map[risk]

# Importação do modelo e dos encoders usados no treinamento
model = pickle.load(open('./src/resources/trained_random_forest.pkl', 'rb'))   
one_hot_encoder = pickle.load(open('./src/resources/one_hot_encoder.pkl', 'rb'))
scaler_encoder = pickle.load(open('./src/resources/scaler_encoder.pkl', 'rb'))

st.title('App Predição de Câncer')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    st.header('Dados de Pacientes com Câncer')

    # TODO: Mostrar os gráficos;


st.header('Predição de Diagnóstico')

# Variáveis a serem consideradas:
# - Age
# - Gender
# - BMI
# - Smoking
# - GeneticRisk
# - PhysicalActivity
# - AlcoholIntake
# - CancerHistory

# Separação dos inputs em 2 colunas e 4 linhas.
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

# Configuração de input para obter idade.
with col1:
    age = st.number_input('Idade em Anos', step=1, min_value=20, max_value=80)

# Configuração de input para obter IMC.
with col2:
    bmi = st.number_input('IMC', min_value=15.0, max_value=40.0)

# Configuração de input para obter tempo de atividade física.    
with col3:
    physical_activity = st.number_input('Tempo de Atividade Física Semanal (em Horas)', min_value=0.0, max_value=10.0)

# Configuração de input para obter álcool consumido na semana.
with col4:
    alcohol_intake = st.number_input('Quantidade de Álcool Consumido na Semana', min_value=0.0, max_value=5.0)

# Configuração de input para obter risco genético.
with col5:
    genetic_risk = st.radio(label= 'Possuí Risco Genético', options = ['Baixo', 'Médio', 'Alto'])

# Configuração de input para obter gênero.
with col6:
    gender = st.radio(label= 'Gênero', options = ['Feminino', 'Masculino'])

# Configuração de input para obter se é fumante.
with col7:
    smoking = st.checkbox('É fumante?')

# Configuração de input para obter se possí histórico de câncer.
with col8:
    cancer_history = st.checkbox('Possuí Histórico de Câncer?')

submit = st.button('Predizer Diagnóstico')

if submit:
    # Monta registro do paciente
    patient = {
        'Age': age,
        'Gender': int(gender == "Feminino"),
        'BMI': bmi,
        'Smoking': int(smoking),
        'GeneticRisk': get_formated_genetic_risk(genetic_risk),
        'PhysicalActivity': physical_activity,
        'AlcoholIntake': alcohol_intake,
        'CancerHistory': int(cancer_history)
    }

    # Coloca o registro em dataframe
    df = pd.DataFrame([patient])
    values = df.values

    # Aplica enconders em cima do registro
    hot_encoded_values = one_hot_encoder.transform(values)
    scaler_encoded_values = scaler_encoder.transform(hot_encoded_values)

    # Realiza a predição em cima dos dados.
    results = model.predict(scaler_encoded_values)

    # Retorna para usuário o resultado.
    result_to_show = ":red[Diagnosticado]" if results[0] else ":green[Não Diagnosticado]"
    st.subheader("Resultado da Predição:  " + result_to_show)