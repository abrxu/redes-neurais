from openai import OpenAI

class AIHelper:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("A chave da API da OpenAI não foi fornecida.")
        self.api_key = api_key
        
        self.client = OpenAI(api_key=self.api_key)

    def get_response(self, prompt, max_tokens=50):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Desculpe, ocorreu um erro ao tentar gerar uma resposta: {str(e)}"