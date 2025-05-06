# Fewshot prompt with examples. The examples are from the training data.
# Additional description of the tables
# The SQL query is generated using the SQL LLM model with full database schema 
# If an error is received from the sql server, then the error message is used again to correct for the mistake, but it didn't help much
 
#from langchain_experimental.sql import SQLDatabaseChain #, SQLDatabaseSequentialChain
#from langchain_community.utilities.sql_database import SQLDatabaseChain
import os, sys

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import ChatOpenAI

from sqlalchemy.exc import SQLAlchemyError
import sql_metadata
import sqlite3

import warnings
warnings.filterwarnings("ignore")


#from tqdm import tqdm
import time
import pandas as pd
import logging

# These are own imports
from texttosql_localllm import utils
from texttosql_localllm.text2sql import Txt2Sql
import texttosql_localllm.problem_class as pc


# Setup logging
logformat = 'Txt2SQL: %(name)s - %(message)s - %(lineno)d'
logging.basicConfig(level=logging.INFO, format=logformat)
logger = logging.getLogger(__name__)

# Add function name to log records
class FunctionNameFilter(logging.Filter):
    def filter(self, record):
        record.funcName = record.funcName
        return True

logger.addFilter(FunctionNameFilter())

class Workflow_4():

    def __init__(self, sql_llm_model_name, 
                 db_name="sewage", 
                 conn_info={}, 
                 ollama_host="https://localhost:11434", 
                 token="fsdfs", 
                 openai_model=False, 
                 schema_file="",
                 examples_path=""):
        
        self.sql_llm_model_name = sql_llm_model_name  # SQL LLM model name to be used for generating SQL queries
        self.db_name = db_name # database name to be used for querying
        self.db = None
        self.conn_info = conn_info
        self.ollama_host = ollama_host
        self.token = token # API token for OpenAI if needed
        self.openai_model = openai_model
        self.schema_file = schema_file
        self.examples_path = examples_path
        self.ollama_sql = None


        # Load the database settings
        self.db = SQLDatabase.from_uri(f"postgresql+psycopg2://{self.conn_info['user']}:{self.conn_info['password']}@{self.conn_info['host']}/{self.conn_info['database']}", 
                                  schema=self.conn_info['schema'])
        
        logger.info(f"--- Using Model: {self.sql_llm_model_name} ----")
        if openai_model:
            self.ollama_sql = ChatOpenAI(openai_api_key=token, model=self.sql_llm_model_name)
        else:
            self.ollama_sql = OllamaLLM(base_url=self.ollama_host, model=self.sql_llm_model_name, temperature=0, keep_alive=15)

    def init_tool(self):        
        self.t2s = Txt2Sql(sql_llm_model=self.ollama_sql, db=self.db, schema_file=self.schema_file)
        # if self.examples_path:
        #     self.examples_dict = pd.read_csv(self.examples_path).to_dict(orient='records')
        #     self.t2s.add_examples(self.examples_dict)

        self.t2s._init_embedding_model()
        
        self.t2s._init_examples(reset=True)
        
        # if len(self.t2s.vectorstore_examples.get()['ids']) == 0:
        examples_dict = pd.read_csv(self.examples_path)#.to_dict(orient='records')
        self.t2s.add_examples(examples_dict)
        self.t2s._init_example_selector()
        
        self.t2s._init_docs(reset=True)
        # if len(self.t2s.vectorstore_docs.get()['ids']) == 0:
        # table_description = f"{self.ddir}/table_description.txt"
        # table_columns_description = f"{self.ddir}/table_column_description.txt"
        # self.t2s.load_split_add_csv(table_description, csv_args={'fieldnames': ['Table name', 'Description'], 'delimiter': '\t'})
        # self.t2s.load_split_add_csv(table_columns_description, csv_args={'fieldnames': ['Table name', 'Column name', 'Variable type', 'Description'], 'delimiter': '\t'})
        
        self.t2s._init_dbschema(reset=True)
        # if len(self.t2s.vectorstore_dbschema.get()['ids']) == 0:
        self.t2s.add_dbschema()

    def run(self, question, execute_query=False):
        if self.t2s is None:
            self.init_tool()

        # Load the database schema        
        self.t2s.set_question(question)        

        #if answered:
            #logger.info(f"Question already answered: {question}")
            # We could feed in the previous error message into the prompt to give a hint for correcting for the mistake
            # if error[0] != "None":
            #     print(f"But there was an error")
            #     error_split = error[0].split("\n")
            #     include_error_message.append(error_split[0])
            #     for error_line in error_split[1:]:
            #         if "HINT" in error_line:
            #             include_error_message.append(error_line)
            #             break

            #     print("\n".join(include_error_message))
        error = None
        
        #if not answered: #or there_is_error[0] != "None":
        # answer = "None"
        # p1 = pc.Problem(question)
        # p1.model_name = sql_llm_model_name
        # p1.answer = "None"           
        # p1.qid = index

        # We could feed in the previous error message into the prompt to give a hint for correcting for the mistake
        #for step in range(1, 4):
        # if error:
        #     error_split = error.split("\n")
        #     include_error_message.append("\n"+error_split[0])
        #     for error_line in error_split[1:]:
        #         if "HINT" in error_line:
        #             include_error_message.append("\n"+error_line)

        #     t2s.add_additional_info("".join(include_error_message))

        response, prompt = self.t2s.run_with_fewshot_prompt(return_prompt=False, full_dbschema=True)
        sql_gen = utils.postprocess_sql(response['result'])
        # p1.prompt = prompt
        # p1.sql_query.append(sql_gen)
        
        try:
            if execute_query:
                answer = self.db.run(sql_gen)
            else:
                return sql_gen
            # p1.answer = answer
            # p1.error.append("None")
            #sbreak
        except Exception as e:
            print(e)  
            # p1.error.append(str(e))
            error = str(e)

            
        logger.info(f"Answer: {answer}")
        
        # if utils.check_answer(Otrain, p1):
        #     p1.accuracy_score=pc.ACCURACY['CORRECT']
        #     logger.info(f"Correct SQL query, {p1.accuracy_score}")
        
        # if not p1.sql_query:
        #     p1.add_sql_queries(query=sql_gen['result'], answer="None", error='No SQL query generated')
        #     logger.info("No SQL query generated")          
        
        # try:
        #     p1.export(chromadb_prefix, param=param)
        # except Exception as e:
        #     logger.info(f"Error exporting {e}")

        return answer, error

