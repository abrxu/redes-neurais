import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.model_loader import load_encoders, load_trained_model, preprocess_input
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from src.ai_helper import AIHelper
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import json

ai_helper = AIHelper()

def load_data(file_path='data/data.csv'):
    if not os.path.exists(file_path):
        st.error(f"Arquivo de dados '{file_path}' não encontrado.")
        st.stop()
    data = pd.read_csv(file_path)
    return data

def main():
    st.set_page_config(page_title="Analisador de Compatibilidade Estudantil", layout="centered")
    st.title("🎓 Analisador de Compatibilidade Estudantil")

    try:
        encoders = load_encoders()
        model = load_trained_model()
    except FileNotFoundError as e:
        st.error(e)
        st.stop()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0

    st.sidebar.header("📝 Insira Seus Dados")

    def user_input_features():
        area = st.sidebar.selectbox("🏫 Área Desejada", ("Saúde", "Tecnologia", "Gestão/Negócios"))
        age = st.sidebar.slider("🎂 Idade", 18, 70, 25)
        gender = st.sidebar.selectbox("🚻 Gênero", ("M", "F"))
        distance = st.sidebar.selectbox("📍 Distância da Faculdade", ("<15km", ">15km"))
        return area, age, gender, distance

    area, age, gender, distance = user_input_features()

    if gender not in encoders['gender'].classes_:
        st.error(f"Gênero inválido: {gender}. Valores esperados: {list(encoders['gender'].classes_)}")
        st.stop()
    if distance not in encoders['distance'].classes_:
        st.error(f"Distância inválida: {distance}. Valores esperados: {list(encoders['distance'].classes_)}")
        st.stop()
    if area not in encoders['area'].classes_:
        st.error(f"Área inválida: {area}. Valores esperados: {list(encoders['area'].classes_)}")
        st.stop()

    try:
        input_data = preprocess_input(age, gender, distance, area, encoders)
        compatibility_score = model.predict(input_data)[0][0] * 100 
        st.markdown(f"### 🎯 **Sua compatibilidade para a vaga é: {compatibility_score:.2f}%**")
    except ValueError as e:
        st.error(f"Erro ao processar entrada: {e}")
        st.stop()

    st.markdown("### ❓ O que você gostaria de fazer a seguir?")

    option = st.selectbox("Escolha uma opção:", ("", "📊 Como você determinou isso?", "📈 Mostrar estatísticas", "💬 Falar com AI"))

    if option == "📊 Como você determinou isso?":
        st.write("""
            A compatibilidade foi calculada utilizando um modelo de rede neural treinado para prever as chances de uma pessoa ser aceita em uma determinada área de estudo com base em diversas características pessoais. O modelo foi desenvolvido e treinado usando dados sintéticos cuidadosamente construídos para refletir padrões de aceitação em áreas específicas, como Saúde, Tecnologia e Gestão/Negócios.

            ### Como o Modelo Funciona

            Este projeto utiliza uma rede neural do tipo feedforward com várias camadas densamente conectadas. As camadas incluem unidades de regularização, como Dropout e L2 regularization, para evitar overfitting, e Batch Normalization para estabilizar o treinamento. A arquitetura da rede foi projetada para processar as seguintes características de entrada:

            - **Idade**: Certas faixas etárias foram configuradas para ter maior probabilidade de aceitação dependendo da área. Por exemplo, em Tecnologia, uma faixa etária mais jovem foi priorizada, enquanto em Gestão/Negócios, faixas etárias intermediárias e mais maduras foram consideradas favoravelmente.
            - **Gênero**: A aceitação pode variar de acordo com a distribuição de gênero, pois as áreas apresentam padrões diferentes. Por exemplo, a área de Saúde foi configurada com uma maior aceitação de pessoas do gênero feminino, enquanto a área de Tecnologia priorizou pessoas do gênero masculino, com base em dados observacionais para construir o perfil esperado de aceitação.
            - **Distância da Faculdade**: A proximidade da faculdade é um fator importante, especialmente em áreas onde a presença física é um diferencial. Distâncias menores (<15km) tendem a aumentar a probabilidade de aceitação, especialmente em Gestão/Negócios e Saúde.
            - **Área de Interesse**: Diferentes áreas apresentam suas próprias taxas de aceitação. O modelo considera como as características de um candidato se alinham com os perfis que historicamente têm maior aceitação em cada área.

            ### Processo de Treinamento do Modelo

            O modelo foi treinado usando o método de K-Fold Cross Validation, o que permitiu avaliar seu desempenho em várias partições dos dados, melhorando a robustez da avaliação. Durante o treinamento, utilizamos métricas de acurácia e perda para monitorar o desempenho da rede neural em cada fold, ajustando automaticamente os pesos da rede para minimizar o erro nas previsões.

            - **Early Stopping**: Este mecanismo foi utilizado para interromper o treinamento assim que o modelo parasse de melhorar na métrica de validação, evitando o treinamento excessivo (overfitting).
            - **Otimizador Adam**: O otimizador Adam foi usado para ajustar os pesos do modelo, oferecendo uma convergência rápida e eficiente.

            ### Avaliação do Desempenho

            Após o treinamento, o modelo foi avaliado com métricas avançadas para garantir sua precisão:

            - **Curva ROC e AUC**: A curva ROC foi utilizada para medir a capacidade do modelo em distinguir entre os candidatos aceitos e rejeitados. A área sob a curva (AUC) indica a eficácia da rede em fazer essa distinção, com valores mais altos indicando uma maior precisão.
            - **Matriz de Confusão**: Essa matriz permitiu verificar onde o modelo acerta ou erra nas suas previsões, ajudando a identificar padrões de classificação incorreta.
            - **Curvas de Acurácia e Perda**: Durante o treinamento, as curvas de acurácia e perda foram geradas para monitorar o aprendizado do modelo, tanto nos dados de treino quanto nos dados de validação.

            ### Como a Predição é Realizada

            Para determinar a compatibilidade de um candidato, o modelo recebe as características pessoais (idade, gênero, distância e área de interesse) e processa essas informações através de sua estrutura de camadas densas, produzindo uma pontuação de compatibilidade em percentual. Essa pontuação representa a probabilidade de aceitação do candidato na área escolhida, de acordo com o perfil treinado.

            Esse processo utiliza os pesos ajustados durante o treinamento, que capturam as relações entre as características de entrada e a aceitação nas diferentes áreas. A predição final reflete como o perfil do candidato se alinha com os perfis históricos de aceitação.

            ### Construção dos Dados de Treinamento

            Os dados de treinamento foram gerados com base em distribuições configuradas para refletir um cenário realista de aceitação:

            - Para cada área, a probabilidade de aceitação foi ajustada para refletir padrões observados. Por exemplo, em Tecnologia, os candidatos mais jovens e do gênero masculino receberam uma maior probabilidade de aceitação.
            - As características de idade, gênero e distância foram distribuídas com base em probabilidades específicas para cada área, garantindo que os dados de treinamento representassem adequadamente os padrões de aceitação observados.

            Esses dados foram então utilizados para treinar a rede neural, permitindo que o modelo aprenda a identificar os perfis mais compatíveis em cada área de interesse.
        """)

    elif option == "📈 Mostrar estatísticas":
        statistics_type = st.selectbox("Escolha o tipo de estatísticas que deseja visualizar:", ("📊 Dados Utilizados", "🤖 Desempenho da IA"))

        if statistics_type == "📊 Dados Utilizados":
            data = load_data()

            try:
                # Mapeia valores numéricos para rótulos
                data['Gênero'] = data['Gênero'].replace({0: 'Feminino', 1: 'Masculino'})
                data['Distância'] = data['Distância'].replace({0: '<15km', 1: '>15km'})
                data['Área'] = data['Área'].replace({0: 'Gestão/Negócios', 1: 'Saúde', 2: 'Tecnologia'})

            except Exception as e:
                st.error(f"Erro ao codificar dados: {e}")
                st.stop()

            # Gráfico interativo de Matrículas por Área
            st.subheader("📊 Distribuição de Matrículas por Área")
            enrollment_counts = data.groupby(['Área', 'Matriculado']).size().unstack(fill_value=0)
            enrollment_counts = enrollment_counts.rename(columns={0: 'Não Matriculado', 1: 'Matriculado'})

            fig = px.bar(
                enrollment_counts, 
                x=enrollment_counts.index, 
                y=['Matriculado', 'Não Matriculado'], 
                title="Distribuição de Matrículas por Área",
                labels={'value': 'Número de Estudantes', 'Área': 'Área', 'variable': 'Status'}
            )
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig)

            # Gráfico interativo de Gênero
            st.subheader("📊 Distribuição de Gênero")
            gender_counts = data['Gênero'].value_counts()
            fig2 = px.pie(
                names=gender_counts.index, 
                values=gender_counts.values, 
                title="Distribuição de Gênero dos Estudantes"
            )
            st.plotly_chart(fig2)

            # Gráfico interativo de Faixa Etária
            st.subheader("🎂 Faixa Etária dos Estudantes")
            fig3 = px.histogram(
                data, x="Idade", nbins=10, title="Distribuição de Idades dos Estudantes",
                labels={'Idade': 'Idade', 'count': 'Número de Estudantes'}
            )
            st.plotly_chart(fig3)

            # Gráficos e estatísticas específicos para cada área
            st.subheader("📊 Análise Específica por Área")

            for area in data['Área'].unique():
                st.markdown(f"### Área: {area}")

                # Filtro para a área específica
                area_data = data[data['Área'] == area]

                # Gênero por área
                gender_counts_area = area_data['Gênero'].value_counts(normalize=True) * 100
                fig_gender_area = px.pie(
                    names=gender_counts_area.index, 
                    values=gender_counts_area.values, 
                    title=f"Distribuição de Gênero na Área {area}",
                    labels={'label': 'Gênero', 'value': 'Percentual'}
                )
                st.plotly_chart(fig_gender_area)

                # Taxa de Matrícula por Gênero na área
                enrollment_gender_area = area_data.groupby(['Gênero', 'Matriculado']).size().unstack(fill_value=0)
                enrollment_gender_area = enrollment_gender_area.rename(columns={0: 'Não Matriculado', 1: 'Matriculado'})
                fig_enrollment_gender_area = px.bar(
                    enrollment_gender_area, 
                    x=enrollment_gender_area.index, 
                    y=['Matriculado', 'Não Matriculado'], 
                    title=f"Taxa de Matrícula por Gênero na Área {area}",
                    labels={'value': 'Número de Estudantes', 'Gênero': 'Gênero', 'variable': 'Status'}
                )
                fig_enrollment_gender_area.update_layout(barmode='stack')
                st.plotly_chart(fig_enrollment_gender_area)

                # Distribuição de Idade para Matriculados e Não Matriculados na área
                fig_age_enrollment_area = px.histogram(
                    area_data, x="Idade", color="Matriculado",
                    title=f"Distribuição de Idade para Matriculados e Não Matriculados na Área {area}",
                    labels={'Idade': 'Idade', 'count': 'Número de Estudantes', 'Matriculado': 'Status'},
                    barmode='overlay'
                )
                fig_age_enrollment_area.update_traces(opacity=0.6)
                st.plotly_chart(fig_age_enrollment_area)

                # Distribuição de Matrícula por Distância na área
                distance_enrollment_area = area_data.groupby(['Distância', 'Matriculado']).size().unstack(fill_value=0)
                distance_enrollment_area = distance_enrollment_area.rename(columns={0: 'Não Matriculado', 1: 'Matriculado'})
                fig_distance_enrollment_area = px.bar(
                    distance_enrollment_area, 
                    x=distance_enrollment_area.index, 
                    y=['Matriculado', 'Não Matriculado'], 
                    title=f"Distribuição de Matrícula por Distância na Área {area}",
                    labels={'value': 'Número de Estudantes', 'Distância': 'Distância', 'variable': 'Status'}
                )
                fig_distance_enrollment_area.update_layout(barmode='stack')
                st.plotly_chart(fig_distance_enrollment_area)

        elif statistics_type == "🤖 Desempenho da IA":
            # Carrega os dados de histórico de métricas e predições
            with open('fold_histories.json', 'r') as f:
                fold_histories = json.load(f)
            
            test_data = pd.read_csv('test_predictions.csv')
            y_true = test_data['labels']
            y_pred = (test_data['predictions'] > 0.5).astype(int)  # Ajuste o threshold conforme necessário

            # Curva de Acurácia
            st.subheader("📈 Curva de Acurácia")
            fig_accuracy = go.Figure()
            for fold_accuracy in fold_histories['accuracy']:
                fig_accuracy.add_trace(go.Scatter(
                    x=list(range(1, len(fold_accuracy) + 1)),
                    y=fold_accuracy,
                    mode='lines',
                    name="Acurácia Treinamento - Fold"
                ))
            for fold_val_accuracy in fold_histories['val_accuracy']:
                fig_accuracy.add_trace(go.Scatter(
                    x=list(range(1, len(fold_val_accuracy) + 1)),
                    y=fold_val_accuracy,
                    mode='lines',
                    name="Acurácia Validação - Fold",
                    line=dict(dash='dash')
                ))
            fig_accuracy.update_layout(
                title="Curva de Acurácia ao Longo das Épocas",
                xaxis_title="Épocas",
                yaxis_title="Acurácia"
            )
            st.plotly_chart(fig_accuracy)

            # Curva de Perda
            st.subheader("📉 Curva de Perda")
            fig_loss = go.Figure()
            for fold_loss in fold_histories['loss']:
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(fold_loss) + 1)),
                    y=fold_loss,
                    mode='lines',
                    name="Perda Treinamento - Fold"
                ))
            for fold_val_loss in fold_histories['val_loss']:
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(fold_val_loss) + 1)),
                    y=fold_val_loss,
                    mode='lines',
                    name="Perda Validação - Fold",
                    line=dict(dash='dash')
                ))
            fig_loss.update_layout(
                title="Curva de Perda ao Longo das Épocas",
                xaxis_title="Épocas",
                yaxis_title="Perda"
            )
            st.plotly_chart(fig_loss)

            # Matriz de Confusão
            st.subheader("📊 Matriz de Confusão")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Não Matriculado', 'Matriculado'],
                y=['Não Matriculado', 'Matriculado'],
                hoverongaps=False,
                colorscale='Blues'
            ))
            fig_cm.update_layout(
                title="Matriz de Confusão",
                xaxis_title="Previsão",
                yaxis_title="Real"
            )
            st.plotly_chart(fig_cm)

            # Curva ROC e AUC
            st.subheader("📈 Curva ROC e AUC")
            fpr, tpr, _ = roc_curve(y_true, test_data['predictions'])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name="Aleatório", line=dict(dash="dash")
            ))
            fig_roc.update_layout(
                title="Curva ROC",
                xaxis_title="Taxa de Falso Positivo (FPR)",
                yaxis_title="Taxa de Verdadeiro Positivo (TPR)"
            )
            st.plotly_chart(fig_roc)


    elif option == "💬 Falar com AI":
        st.write("### 💬 Chat com o Assistente AI")

        # Exibe o histórico de chat
        for speaker, message in st.session_state.chat_history:
            if speaker == "Você":
                st.markdown(f"**Você:** {message}")
            else:
                st.markdown(f"**AI:** {message}")

        # Campo de entrada para nova mensagem com chave dinâmica
        user_message = st.text_input("Digite sua mensagem:", key=f"input_{st.session_state.message_counter}")

        # Processa a mensagem quando o usuário envia
        if user_message:
            # Adiciona a mensagem do usuário ao histórico
            st.session_state.chat_history.append(("Você", user_message))
            
            # Obtém a resposta da IA
            with st.spinner("Pensando..."):
                response = ai_helper.get_response(user_message)
            
            # Adiciona a resposta ao histórico
            st.session_state.chat_history.append(("AI", response))

            # Incrementa o contador para atualizar o campo de entrada automaticamente
            st.session_state.message_counter += 1

if __name__ == "__main__":
    main()
