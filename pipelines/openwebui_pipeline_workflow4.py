#import psycopg2
#from psycopg2 import sql
import logging
from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel

#from sqlalchemy import create_engine, inspect
import aiohttp
import asyncio

from texttosql_localllm import *
#from texttosql_localllm import workflows as ws
from workflows import Workflow_4

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

class Pipeline:
    class Valves(BaseModel):
        DB_HOST: str
        DB_PORT: str
        DB_USER: str
        DB_PASSWORD: str
        DB_DATABASE: str
        DB_SCHEMA: str
        OLLAMA_BASE_URL: str
        OLLAMA_MODEL_NAME: str
        OPENAI_API_KEY: str
        TEXT_TO_SQL_MODEL: str

    def __init__(self):
        self.name = "01 Database RAG Pipeline OLlama"
        self.conn_info = {}
        self.nlsql_response = ""
        self.w4 = None

        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "DB_HOST": os.getenv("PG_HOST", "127.0.0.1"),
                "DB_PORT": os.getenv("PG_PORT", '5432'),
                "DB_USER": os.getenv("PG_USER", "reader"),
                "DB_PASSWORD": os.getenv("PG_PASSWORD", "data_reader"),
                "DB_DATABASE": os.getenv("PG_DB", "sewage"),
                "DB_SCHEMA": os.getenv("PG_SCHEMA", "default"),
                'OLLAMA_BASE_URL': os.getenv("OLLAMA_BASE_URL", "127.0.0.2:11434"),
                "OLLAMA_MODEL_NAME": os.getenv("OLLAMA_MODEL_NAME","llama3.2"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", 'abc-123'),
                "TEXT_TO_SQL_MODEL": os.getenv("TEXT_TO_SQL_MODEL", "qwen2.5-coder:14b")
            }
        )

    def init_text_to_sql_tool(self):
        self.conn_info = {
            'database': self.valves.DB_DATABASE,
            'user': self.valves.DB_USER,
            'password': self.valves.DB_PASSWORD,
            'host': self.valves.DB_HOST,
            'port': self.valves.DB_PORT,
            'schema': self.valves.DB_SCHEMA
        }
        self.w4 = Workflow_4(sql_llm_model_name=self.valves.TEXT_TO_SQL_MODEL, 
                 ollama_host=self.valves.OLLAMA_BASE_URL, 
                 conn_info=self.conn_info, 
                 token=self.valves.OPENAI_API_KEY, 
                 openai_model=False, 
                 )
            
        # datadir="/app/pipelines"
        datadir="./"
        self.w4.schema_file = f"{datadir}/data/sewage_schema.sql"
        self.w4.examples_path = f"{datadir}/data/example_qa_sewage.csv"
        self.w4.init_tool()
        self.w4.t2s.load_split_add_csv(f"{datadir}/data/table_description.txt", csv_args={'fieldnames': ['Table name', 'Description'], 'delimiter': '\t'})
        #t2s.load_split_add_csv(f"data/hints.txt", csv_args={'fieldnames': ['tags', 'hint'], 'delimiter': '\t'})
        self.w4.t2s.load_split_add_csv(f"{datadir}/data/table_column_description.txt", csv_args={'fieldnames': ['Table name', 'Column name', 'Variable type', 'Description'], 'delimiter': '\t'})
        #t2s.load_split_add_text("sewage_data_descriptor.txt", split_on=["---"])

        


    async def on_startup(self):
        self.init_text_to_sql_tool()

    async def on_shutdown(self):
        #self.cur.close()
        #self.conn.close()
        pass

    async def make_request_with_retry(self, url, params, retries=3, timeout=10):
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=timeout) as response:
                        response.raise_for_status()
                        return await response.text()
            except (aiohttp.ClientResponseError, aiohttp.ClientPayloadError, aiohttp.ClientConnectionError) as e:
                logging.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def extract_sql_query(self, response_object):
        for key, value in response_object.items():
            if isinstance(value, dict) and 'sql_query' in value:
                return value['sql_query']
            elif key == 'sql_query':
                return value
        return None

    def handle_streaming_response(self, response_gen):
        final_response = ""
        for chunk in response_gen:
            final_response += chunk
        return final_response

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # Use the established psycopg2 connection to create a SQLAlchemy engine
        

        try:
            self.nlsql_response = self.w4.run(user_message, execute_query=False)
            #sql_query = self.extract_sql_query(self.nlsql_response)
            # if hasattr(self.nlsql_response, 'response_gen'):
            #     final_response = self.handle_streaming_response(self.nlsql_response)
            #     result = f"Generated SQL Query:\n```sql\n{self.nlsql_response}\n```\nResponse:\n{final_response}"
            #     return result
            # else:
            final_response = self.nlsql_response
            result = f"Generated SQL Query:\n```sql\n{self.nlsql_response}\n```\nResponse:\n{final_response}"
            return result
        except aiohttp.ClientResponseError as e:
            logging.error(f"ClientResponseError: {e}")
            return f"ClientResponseError: {e}"
        except aiohttp.ClientPayloadError as e:
            logging.error(f"ClientPayloadError: {e}")
            return f"ClientPayloadError: {e}"
        except aiohttp.ClientConnectionError as e:
            logging.error(f"ClientConnectionError: {e}")
            return f"ClientConnectionError: {e}"
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return f"Unexpected error: {e}"
