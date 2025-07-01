from dotenv import load_dotenv
import os

load_dotenv()

from texttosql_localllm import *
from texttosql_localllm.workflows import Workflow_4

conn_info = {
    'database': os.getenv("PG_DB", "sewage"),
    'user': os.getenv("PG_USER", "reader"),
    'password': os.getenv("PG_PASSWORD", "data_reader"),
    'host': os.getenv("PG_HOST", "localhost"),
    'port': os.getenv("PG_PORT", '5432'),
    'schema': os.getenv("PG_SCHEMA", "default")
}

# Ollama server
ollama_host = os.getenv("OLLAMA_BASE_URL", "")
sql_llm_model_name="qwen2.5-coder:14b"

datadir="./"
schema_file = f"{datadir}/sewage_schema.sql"
# with open("/v/wfct0p/API-tokens/openai-api.token") as f:
        # token = f.read().strip()
# os.environ['OPENAI_API_KEY'] = token
example_path = f"{datadir}/example_qa_sewage.csv"
# f = open(example_path)
w4 = Workflow_4(sql_llm_model_name=sql_llm_model_name, 
                 conn_info=conn_info, 
                 ollama_host=ollama_host, 
                 token="fsdfs", 
                 openai_model=False,
                 schema_file=schema_file, 
                 examples_path=example_path)


w4.t2s.load_split_add_csv(f"{datadir}/table_description.txt", csv_args={'fieldnames': ['Table name', 'Description'], 'delimiter': '\t'})
#t2s.load_split_add_csv(f"{datadir}/hints.txt", csv_args={'fieldnames': ['tags', 'hint'], 'delimiter': '\t'})
w4.t2s.load_split_add_csv(f"{datadir}/table_column_description.txt", csv_args={'fieldnames': ['Table name', 'Column name', 'Variable type', 'Description'], 'delimiter': '\t'})
#t2s.load_split_add_text("sewage_data_descriptor.txt", split_on=["---"])

w4.run("How many sewage treatment plants are there in the city of Budapest?")