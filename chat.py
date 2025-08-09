import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.chat_models import ChatOpenAI

load_dotenv()

class SQLChatBot:
    def __init__(self, db_path):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.db = SQLDatabase(self.engine)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.chain = create_sql_query_chain(self.llm, self.db)

    def ask(self, question):
        try:
            sql_query = self.chain.invoke({"question": question})
            if not sql_query or not isinstance(sql_query, str) or sql_query.strip() == "":
                return "Não foi possível gerar uma consulta SQL para sua pergunta."
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                if not rows:
                    return "Nenhum resultado encontrado."
                resposta_natural = self.llm.invoke(
                    f"Coloque este resultado em palavras humanizadas, resumido e direto: {sql_query}\nResultado: {rows}, não mencione a base de dados ou SQL. Apenas responda com o que foi perguntado. Caso a resposta não condiza com a pergunta, informe que essa informação não foi encontrada."
                )
                if hasattr(resposta_natural, "content"):
                    return resposta_natural.content
                return resposta_natural
        except Exception as e:
            return f"Erro ao gerar resposta: {e}"

def main():
    db_path = "airlines_flights_data.db"  # Altere para o caminho do seu banco
    bot = SQLChatBot(db_path)
    print("ChatBot SQL pronto. Pergunte sobre a base de dados!")
    while True:
        question = input("Pergunta: ")
        answer = bot.ask(question)
        print("Resposta:", answer)

if __name__ == "__main__":
    main()