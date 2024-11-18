import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

    dataset = pd.read_csv('./src/resources/The_Cancer_data_1500_V2.csv')
    dataset_for_plots = dataset.copy()
    dataset_for_plots['GeneticRisk'] = dataset_for_plots['GeneticRisk'].map({0: 'Baixo', 1: 'Médio', 2: 'Alto'})
    dataset_for_plots['Gender'] =  dataset_for_plots['Gender'].map({0: 'Masculino', 1: 'Feminino'})
    dataset_for_plots['Smoking'] =  dataset_for_plots['Smoking'].map({0: 'Não', 1: 'Sim'})
    dataset_for_plots['CancerHistory']  = dataset_for_plots['CancerHistory'].map({0: 'Sem histórico', 1: 'Com histórico'})

    plot = sns.countplot(x=dataset_for_plots['Gender'])

    plt.title('Distribuição por Gênero')
    plt.ylabel('Contagem')
    plt.xlabel('Gênero')

    st.pyplot(plot.get_figure())

    plt.clf()

    plot = sns.countplot(data=dataset_for_plots, x='Gender', hue='Diagnosis')

    plt.title('Relação entre Gênero e Câncer')
    plt.xlabel('Gênero')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    plot = sns.countplot(x=dataset_for_plots['Smoking'])

    plt.title('Relação entre Fumantes e Câncer')
    plt.xlabel('Fumante')
    plt.ylabel('Contagem')

    st.pyplot(plot.get_figure())

    plt.clf()

    plot = sns.countplot(data=dataset_for_plots, x='Smoking', hue='Diagnosis')

    plt.title('Relação entre Fumantes e Câncer')
    plt.xlabel('Fumante')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    genetic_risk_distribution = np.unique(dataset_for_plots['GeneticRisk'], return_counts=True)
    divisor = len(dataset_for_plots) * 100

    high_risk_perc = genetic_risk_distribution[1][0] / divisor
    low_risk_perc = genetic_risk_distribution[1][1] / divisor
    medium_risk_perc = genetic_risk_distribution[1][2] / divisor

    sizes = [low_risk_perc, medium_risk_perc, high_risk_perc]
    labels = ["Baixo Risco", "Médio Risco", "Alto Risco"]

    fig = plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Distribuição de Pacientes por Risco Genético')
    plt.show()

    st.pyplot(fig)

    plt.clf()

    plot = sns.countplot(data=dataset_for_plots, x='GeneticRisk', hue='Diagnosis')

    plt.title('Relação entre Risco Genêtico e Câncer')
    plt.xlabel('Risco Genêtico')
    plt.ylabel('Contagem')
    plt.legend(title='Diagnosticado com Câncer', labels=['Não', 'Sim'])

    st.pyplot(plot.get_figure())

    plt.clf()

    plot = sns.violinplot(x='Diagnosis', y='AlcoholIntake', data=dataset_for_plots)

    plt.title('Relação entre Ingestão de Álcool e Câncer')
    plt.xlabel('Diagnosticado com Câncer')
    plt.ylabel('Ingestão de Álcool (Unidades por Semana)')

    st.pyplot(plot.get_figure())

    plt.clf()

    plot = sns.swarmplot(x='Age', hue='Diagnosis', data=dataset_for_plots)

    plt.title('Relação entre Idade e Câncer')
    plt.xlabel('Idade')
    plt.ylabel('Diagnosticado com Câncer')
    plt.legend(title='Diagnosticado com Câncer', labels=['Sim', 'Não'])

    st.pyplot(plot.get_figure())

    plt.clf()

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