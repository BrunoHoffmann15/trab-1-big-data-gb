import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# TODO: Mostrar os gráficos;
# TODO: Solicitar que usuário adicione inputs;
# TODO: Fazer importação de encoders;
# TODO: Fazer a predição;

model = pickle.load(open('./src/resources/trained_random_forest.pkl', 'rb'))   
one_hot_encoder = pickle.load(open('./src/resources/one_hot_encoder.pkl', 'rb'))
scaler_encoder = pickle.load(open('./src/resources/scaler_encoder.pkl', 'rb'))

st.title('App Predição de Câncer')


st.header('Predição de Diagnóstico')

## Age,Gender,BMI,Smoking,GeneticRisk,PhysicalActivity,AlcoholIntake,CancerHistory

# define a linha 1 de inputs com 3 colunas
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

# captura a idade da pessoa, como o step é 1, ele considera a idade como inteira
with col1:
    age = st.number_input('Idade em Anos', step=1, min_value=20, max_value=80)

# captura os anos investidos em educação da pessoa
with col2:
    bmi = st.number_input('IMC', min_value=15.0, max_value=40.0)
    
# captura a horas trabalhadas por semana
with col3:
    physical_activity = st.number_input('Tempo de Atividade Física Semanal (em Horas)', min_value=0.0, max_value=10.0)

with col4:
    alcohol_intake = st.number_input('Quantidade de Alcool Consumido na Semana', min_value=0.0, max_value=5.0)

with col5:
    genetic_risk = st.radio(label= 'Possuí Risco Genético', options = ['Baixo', 'Médio', 'Alto'])

with col6:
    gender = st.radio(label= 'Gênero', options = ['Feminino', 'Masculino'])

with col7:
    smoking = st.checkbox('É fumante?')

with col8:
    cancer_history = st.checkbox('Possuí Histórico de Câncer?')


submit = st.button('Predizer Diagnóstico')
