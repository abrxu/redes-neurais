# app.py

import streamlit as st
import matplotlib.pyplot as plt
from src.model_loader import load_encoders, load_trained_model, preprocess_input
import os
import pandas as pd
import numpy as np
from src.ai_helper import AIHelper
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        ai_helper = AIHelper(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de AI: {e}")
        ai_helper = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

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
            A compatibilidade foi calculada utilizando um modelo de rede neural treinado com base em seus dados de idade, gênero, distância e área de interesse.
            O modelo analisa como essas características influenciam as chances de ser aceito na área escolhida.
        """)
        st.write("**Detalhamento dos fatores considerados:**")
        st.write("""
            - **Idade**: Certas faixas etárias têm maior probabilidade de aceitação.
            - **Gênero**: A aceitação pode variar de acordo com a distribuição de gênero.
            - **Distância**: A proximidade da faculdade pode ser um fator relevante.
            - **Área**: Diferentes áreas têm taxas de aceitação distintas baseadas nos dados analisados.
        """)

    elif option == "📈 Mostrar estatísticas":
        data = load_data()

        try:
            data['Gênero'] = encoders['gender'].transform(data['Gênero'])
            data['Distância'] = encoders['distance'].transform(data['Distância'])
            data['Área'] = encoders['area'].transform(data['Área'])
        except Exception as e:
            st.error(f"Erro ao codificar dados: {e}")
            st.stop()

        area_labels = encoders['area'].classes_
        data['Área'] = data['Área'].astype(int).apply(lambda x: area_labels[x])

        st.subheader("📊 Distribuição de Matrículas por Área")
        enrollment_counts = data.groupby(['Área', 'Matriculado']).size().unstack(fill_value=0)
        enrollment_counts = enrollment_counts.rename(columns={0: 'Não Matriculado', 1: 'Matriculado'})

        fig, ax = plt.subplots(figsize=(10,6))
        enrollment_counts.plot(kind='bar', stacked=True, ax=ax, color=['salmon', 'skyblue'])
        plt.title('Distribuição de Matrículas por Área')
        plt.xlabel('Área')
        plt.ylabel('Número de Estudantes')
        plt.xticks(rotation=0)
        plt.legend(title='Status')
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📊 Distribuição de Gênero")
        gender_counts = data['Gênero'].value_counts().rename(index={0: 'Masculino', 1: 'Feminino'})
        fig2, ax2 = plt.subplots(figsize=(6,6))
        gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'], ax=ax2)
        plt.title('Distribuição de Gênero dos Estudantes')
        plt.ylabel('')
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("🎂 Faixa Etária dos Estudantes")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        plt.hist(data['Idade'], bins=range(18, 71, 5), color='skyblue', edgecolor='black')
        plt.title('Distribuição de Idades dos Estudantes')
        plt.xlabel('Idade')
        plt.ylabel('Número de Estudantes')
        st.pyplot(fig3)
        plt.close(fig3)

    elif option == "💬 Falar com AI":
        st.write("### 💬 Chat com o Assistente AI")
        user_message = st.text_input("Você:", key='ai_input')
        if user_message:
            if ai_helper:
                st.session_state.chat_history.append(("Você", user_message))
                with st.spinner("Pensando..."):
                    response = ai_helper.get_response(user_message)
                st.session_state.chat_history.append(("AI", response))

                for speaker, message in st.session_state.chat_history:
                    if speaker == "Você":
                        st.markdown(f"**Você:** {message}")
                    else:
                        st.markdown(f"**AI:** {message}")
            else:
                st.write("**AI:** Desculpe, a funcionalidade AI não está disponível no momento.")

if __name__ == "__main__":
    main()