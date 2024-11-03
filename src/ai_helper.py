import requests

class AIHelper:
    def __init__(self, api_url="http://localhost:3000/ia"):
        self.api_url = api_url

    def get_response(self, prompt, max_tokens=50, temperature=0.7):
        payload = {"text": prompt}
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.text.strip()
        except requests.exceptions.RequestException as e:
            return f"Erro ao obter resposta da IA: {e}"

if __name__ == "__main__":
    ai_helper = AIHelper()
    prompt = "Explique como você funciona."
    response = ai_helper.get_response(prompt)
    print("Resposta da IA:", response)
