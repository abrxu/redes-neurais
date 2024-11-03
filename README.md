
# 🧠 Projeto de Redes Neurais  
### Amostra Científica CESUCA 2024

Bem-vindo ao projeto de redes neurais para a Amostra Científica CESUCA 2024! Este guia ajudará você a configurar e executar o projeto de maneira eficiente. Certifique-se de seguir as etapas cuidadosamente para evitar problemas de configuração.

---

## 📋 Pré-requisitos

Antes de começar, você precisará:

1. **Ollama**: Instale o **Ollama** em sua máquina.
2. **Modelo Llama3**: Baixe o modelo **Llama3** pelo Ollama.

Verifique que o Ollama e o Llama3 estão funcionando corretamente acessando `localhost:11434` no navegador. Você deve ver a mensagem: **"Ollama is running"**.

---

## ⚙️ Preparando o Ambiente

Para configurar o ambiente, siga as instruções abaixo enquanto estiver no diretório do projeto:

```bash
# Instalar as dependências do Python
pip install -r requirements.txt

# Inicializar e instalar pacotes Node.js
npm init -y
npm install express axios
```

---

## 🚀 Executando o Projeto

1. **Iniciar a aplicação Streamlit**: Execute o seguinte comando no terminal:

    ```bash
    streamlit run app.py
    ```

    Caso encontre algum problema ao executar o Streamlit, use o comando alternativo:

    ```bash
    python -m streamlit run app.py
    ```

2. **Executar o servidor para a IA**: Para que a IA funcione corretamente, inicie o `server.js` com o comando:

    ```bash
    node server.js
    ```

---

## 💻 Estrutura do Projeto

Este projeto é organizado da seguinte forma:

- **src/**: Contém os scripts de processamento e modelos da IA.
- **data/**: Diretório para armazenar dados e datasets necessários.
- **models/**: Armazena os modelos treinados.
- **app.py**: Interface do usuário com Streamlit.
- **server.js**: Servidor para comunicação com o modelo de IA via API.

---

## 🛠️ Em Construção

Este projeto está em constante desenvolvimento. Fique atento para novas atualizações e melhorias!
