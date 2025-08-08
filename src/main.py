import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Pai: Leitor de perguntas do usuário
class QuestionReader:
    def __init__(self):
        pass

    def read_question(self):
        return input("Digite sua pergunta: ")

# Filho: Responde com base em base vetorial ou gera SQL
class DynamicResponder(QuestionReader):
    def __init__(self, db_path=None, engine=None):
        super().__init__()
        self.db_path = db_path
        self.engine = engine  # SQLAlchemy engine, se fornecido
        self.llm = OpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

        if db_path:
            self.load_vector_database(db_path)

    def load_vector_database(self, db_path):
        if os.path.exists(db_path):
            try:
                self.vectorstore = FAISS.load_local(db_path, self.embeddings)
            except Exception as e:
                print(f"Erro ao carregar vetorstore: {e}")
        else:
            print(f"Caminho do banco de dados '{db_path}' não encontrado.")

    def interpret_and_respond(self, question):
        context = ""
        if self.vectorstore:
            try:
                docs = self.vectorstore.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Erro na busca semântica: {e}")

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Contexto: {context}\nPergunta: {question}\nResposta:"
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            response = chain.run({"context": context, "question": question})
            return response
        except Exception as e:
            return f"Ocorreu um erro ao gerar a resposta: {e}"

    def natural_language_to_sql(self, question: str) -> str:
        """Gera uma consulta SQL com base em linguagem natural"""
        if not self.engine:
            raise ValueError("Nenhum engine SQL fornecido. Configure o parâmetro `engine`.")

        from langchain.chains import create_sql_query_chain
        from langchain_community.utilities import SQLDatabase
        from langchain.chat_models import ChatOpenAI

        db = SQLDatabase(self.engine)
        chain = create_sql_query_chain(ChatOpenAI(temperature=0), db)
        return chain.invoke({"question": question})

def main():
    db_path = "faiss_db"  # Altere conforme necessário
    # engine = create_engine("sqlite:///seu_banco.db")  # Exemplo se for usar SQL
    responder = DynamicResponder(db_path=db_path)  # , engine=engine)
    
    question = responder.read_question()
    answer = responder.interpret_and_respond(question)
    
    print("Resposta:", answer)

if __name__ == "__main__":
    main()
