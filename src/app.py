import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import util

# Validação de login
if not util.check_password():
    # se a senha estiver errada, para o processamento do app
    print("Usuario nao logado")
    st.stop()


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

    # Carrega o dataset e manipula dados quantitativos em qualitativos para melhor visualização dos gráficos
    dataset = pd.read_csv('./src/resources/The_Cancer_data_1500_V2.csv')
    dataset_for_plots = dataset.copy()
    dataset_for_plots['GeneticRisk'] = dataset_for_plots['GeneticRisk'].map({0: 'Baixo', 1: 'Médio', 2: 'Alto'})
    dataset_for_plots['Gender'] =  dataset_for_plots['Gender'].map({0: 'Masculino', 1: 'Feminino'})
    dataset_for_plots['Smoking'] =  dataset_for_plots['Smoking'].map({0: 'Não', 1: 'Sim'})
    dataset_for_plots['CancerHistory']  = dataset_for_plots['CancerHistory'].map({0: 'Sem histórico', 1: 'Com histórico'})

    # Criação de gráficos
    st.subheader('Análise dos Dados')
    st.write('Para a construção dos gráficos foram utilizadas as libs matplotlib, seaborn e plotly em conjunto com o streamlit para renderização.')
    st.subheader('Gráficos')
    st.subheader('Porcentagens de distribuição de pacientes por gênero')
    gender_distribution = np.unique(dataset_for_plots['Gender'], return_counts=True)
    st.write(f"Porcentagem Masculino: {gender_distribution[1][1] / len(dataset_for_plots) * 100}%")
    st.write(f"Porcentagem Feminino: {gender_distribution[1][0] / len(dataset_for_plots) * 100}%")
    plot = sns.countplot(x=dataset_for_plots['Gender'])

    plt.title('Distribuição por Gênero')
    plt.ylabel('Contagem')
    plt.xlabel('Gênero')

    st.pyplot(plot.get_figure())

    # Limpa o plot para não interferir nos próximos
    plt.clf()
    st.subheader('Porcentagens de distribuição de pacientes por gênero e diagnóstico')
    count_cancer_female = np.sum((dataset_for_plots['Gender'] == 'Feminino') & (dataset_for_plots['Diagnosis']  == 1))
    count_cancer_male = np.sum((dataset_for_plots['Gender'] == 'Masculino') & (dataset_for_plots['Diagnosis'] == 1))

    st.write(f"Total de pessoas do gênero feminino com câncer: {count_cancer_female} ({count_cancer_female/gender_distribution[1][1] * 100}%)")
    st.write(f"Total de pessoas do gênero masculino com câncer: {count_cancer_male} ({count_cancer_male/gender_distribution[1][0] * 100}%)")
    plot = sns.countplot(data=dataset_for_plots, x='Gender', hue='Diagnosis')

    plt.title('Relação entre Gênero e Câncer')
    plt.xlabel('Gênero')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    smoking_distribution = np.unique(dataset_for_plots['Smoking'], return_counts=True)
    st.subheader('Porcentagens de distribuição de pacientes por fumantes')
    st.write(f"Porcentagem de Fumantes: {smoking_distribution[1][1] / len(dataset_for_plots) * 100}%")
    st.write(f"Porcentagem de Não fumantes: {smoking_distribution[1][0] / len(dataset_for_plots) * 100}%")

    plot = sns.countplot(x=dataset_for_plots['Smoking'])

    plt.title('Relação entre Fumantes e Câncer')
    plt.xlabel('Fumante')
    plt.ylabel('Contagem')

    st.pyplot(plot.get_figure())

    plt.clf()

    patients_smoking_with_cancer = dataset_for_plots[(dataset_for_plots["Smoking"] == 'Sim') & (dataset_for_plots["Diagnosis"] == 1)];
    patients_not_smoking_with_cancer = dataset_for_plots[(dataset_for_plots["Smoking"] == 'Não') & (dataset_for_plots["Diagnosis"] == 1)];
    st.subheader('Porcentagens de distribuição de pacientes por fumantes e diagnóstico')
    st.write(f"Percentual de Fumantes com Câncer: {len(patients_smoking_with_cancer) / smoking_distribution[1][1] * 100}%")
    st.write(f"Percentual de Não Fumantes com Câncer: {len(patients_not_smoking_with_cancer) / smoking_distribution[1][0] * 100}%")

    plot = sns.countplot(data=dataset_for_plots, x='Smoking', hue='Diagnosis')

    plt.title('Relação entre Fumantes e Câncer')
    plt.xlabel('Fumante')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    genetic_risk_distribution = np.unique(dataset_for_plots['GeneticRisk'], return_counts=True)

    high_risk_perc = genetic_risk_distribution[1][0] / len(dataset_for_plots) * 100
    low_risk_perc = genetic_risk_distribution[1][1] / len(dataset_for_plots) * 100
    medium_risk_perc = genetic_risk_distribution[1][2] / len(dataset_for_plots) * 100

    sizes = [low_risk_perc, medium_risk_perc, high_risk_perc]
    labels = ["Baixo Risco", "Médio Risco", "Alto Risco"]

    st.header('Porcentagens de distribuição de pacientes por risco genético')

    fig = plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Distribuição de Pacientes por Risco Genético')
    plt.show()

    st.pyplot(fig)

    plt.clf()

    patients_low_risk_with_cancer = dataset_for_plots[(dataset_for_plots["GeneticRisk"] == 'Baixo') & (dataset_for_plots["Diagnosis"] == 1)];
    patients_medium_risk_with_cancer = dataset_for_plots[(dataset_for_plots["GeneticRisk"] == 'Médio') & (dataset_for_plots["Diagnosis"] == 1)];
    patients_high_risk_with_cancer = dataset_for_plots[(dataset_for_plots["GeneticRisk"] == 'Alto') & (dataset_for_plots["Diagnosis"] == 1)];

    st.header('Porcentagens de distribuição de pacientes por risco genético e diagnóstico')
    st.write(f"Percentual de Baixo Risco com Câncer: {len(patients_high_risk_with_cancer) / genetic_risk_distribution[1][0] * 100}%")
    st.write(f"Percentual de Médio Risco com Câncer: {len(patients_low_risk_with_cancer) / genetic_risk_distribution[1][1] * 100}%")
    st.write(f"Percentual de Alto Risco com Câncer: {len(patients_medium_risk_with_cancer) / genetic_risk_distribution[1][2] * 100}%")

    plot = sns.countplot(data=dataset_for_plots, x='GeneticRisk', hue='Diagnosis')

    plt.title('Relação entre Risco Genêtico e Câncer')
    plt.xlabel('Risco Genêtico')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    st.header('Porcentagens de distribuição de pacientes por consumo de álcool e diagnóstico')

    plot = sns.violinplot(x='Diagnosis', y='AlcoholIntake', data=dataset_for_plots)

    plt.title('Relação entre Ingestão de Álcool e Câncer')
    plt.xlabel('Diagnosticado com Câncer')
    plt.ylabel('Ingestão de Álcool (Unidades por Semana)')

    st.pyplot(plot.get_figure())

    plt.clf()

    st.header('Porcentagens de distribuição de pacientes por idade e diagnóstico')

    plot = sns.swarmplot(x='Age', hue='Diagnosis', data=dataset_for_plots)

    plt.title('Relação entre Idade e Câncer')
    plt.xlabel('Idade')
    plt.ylabel('Diagnosticado com Câncer')
    plt.legend(title='Diagnosticado com Câncer', labels=['Sim', 'Não'])

    st.pyplot(plot.get_figure())

    plt.clf()

    st.header('Mapa de divisão de dados por Risco Genético, Histórico de Câncer, Fumante e Gênero')
    st.write('Nesse mapa é possível visualizar de forma mais clara a divisão dos dados e quais são os diagnósticos de cada grupo.')

    patients_data_for_tree_map = dataset_for_plots.drop(columns=['BMI', 'Age', 'AlcoholIntake', 'PhysicalActivity'])

    graph = px.treemap(patients_data_for_tree_map, path=['GeneticRisk', 'CancerHistory','Smoking' , 'Gender'], color="Diagnosis", color_continuous_scale=['#32a852', '#3261a8', '#a83259'])
    st.write('Treemap renderizado em nova aba.')
    graph.show()


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
    cancer_history = st.checkbox('Possui Histórico de Câncer?')

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