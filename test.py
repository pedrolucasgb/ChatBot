import pandas as pd
from sqlalchemy import create_engine

# Nome do arquivo CSV e do banco SQLite
csv_file = r"C:\Users\Pedro\projetos\ChatBot\airlines_flights_data.csv"  # Substitua pelo nome do seu arquivo
db_file = "airlines_flights_data.db"
table_name = "airlines_flights_data"  # Nome da tabela no banco

# Carrega o CSV
df = pd.read_csv(csv_file)

# Cria o banco SQLite e salva o DataFrame como tabela
engine = create_engine(f"sqlite:///{db_file}")
df.to_sql(table_name, engine, index=False, if_exists="replace")

print(f"Banco criado: {db_file}")
print(f"Para acessar via SQLAlchemy use:")
print(f'engine = create_engine("sqlite:///{db_file}")')
print(f'Tabela: "{table_name}"')