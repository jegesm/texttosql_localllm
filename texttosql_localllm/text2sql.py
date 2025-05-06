#from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import  ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_experimental.sql import SQLDatabaseSequentialChain
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatOllama
#from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

import chromadb
#chromadb.api.client.SharedSystemClient.clear_system_cache()
from chromadb.config import Settings

import logging
logger = logging.getLogger(__name__)

import pandas as pd

class Txt2Sql():    
    """
    This class is used to generate SQL queries from natural language questions
    It initializes the LLM model, the database schema, the examples, the prompt and the SQL chain
    and then generates the SQL query based on the input question
    """

    def __init__(self, type="fewshot", sql_llm_model=None, db=None, schema_file=None):
        self.type = type
        self.sql_llm_model = sql_llm_model  # the LLM model used for generating SQL queries
        if not self.sql_llm_model:
            self.init_llm()
        self.db = db    # points to the SQLDatabase object
        self.schema_file = schema_file  # points to the file with the database schema, this contains views and other information as well        
        self.full_dbschema = None   # the full database schema from schema_file
        self.dbschema = None    # the database schema used in the prompt
        self.examples_dict = None   # the examples used in the fewshot prompt
        self.prompt = None
        self.sql_chain = None   # the SQL chain used to generate the SQL query including output parsing as well etc.
        self.sql_validation_chain = None    # Not sure if this is used
        self.collection = None  # the collection used for storing the documents in the Vector DB
        self.extra_documentation = ""   # additional information to be used in the prompt
        self.chroma_client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="mychroma.db")
        self.embeddings = None
        self.testing = False
        self.vectorstore_docs = None
        self.vectorstore_dbschema = None
        self.vectorstore_examples = None
        self.example_selector = None
        self.example_prompt = None

    def set_question(self, question):
        self.question = question

    def init_llm(self, host=None, temperature=0, keep_alive=1):
        """
        Init the LLM model for SQL generation if not already
        """
        self.sql_llm_model = OllamaLLM(base_url=host,
                        model=self.sql_llm_model,
                        temperature=temperature,
                        keep_alive=keep_alive)    
        
    def _init_embedding_model(self):
        # with local CPU or GPU Usage
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {'allow_download': 'True'}
        self.embeddings = GPT4AllEmbeddings(
            model_name=model_name,
            gpt4all_kwargs=gpt4all_kwargs
        )
        
    # def _init_collection(self, collection_name, reset):
    #     """
    #       Initialize the ChromaDB client and create a colection for storing the documents
    #     """
    #     if reset:
    #         self.chroma_client.delete_collection(collection_name)
        
    #     c = self.chroma_client.get_or_create_collection(collection_name)
    #     if not c.count():
    #         logger.info(f"Collection {collection_name} is empty")
    #     self.vectorstore[[collection_name] = Chroma(client=self.chroma_client, embedding_function=self.embeddings, collection_name=collection_name)
        
    # ADD DATABASE INFO
    def _init_dbschema(self, reset=False):   
        collection_name = "dbschema"

        # Reset collection if needed
        if reset:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
                
        # Init collection if not already
        if not self.vectorstore_dbschema:
            c = self.chroma_client.get_or_create_collection(collection_name)
            if not c.count():
                logger.info(f"Collection {collection_name} is empty")
                
        self.vectorstore_dbschema = Chroma(client=self.chroma_client, embedding_function=self.embeddings, collection_name=collection_name)

    def add_dbschema(self):   
        """
        Load the database schema from the schema file and split it into chunks to be stored in the collection
        it's reference tag is "dbschema"
        """
        from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDataBaseTool
        import sql_metadata, sqlparse
        
        # Get DB schema from database if no ddl file is provided
        dbschema = ""
        if not self.schema_file:
            tns = ListSQLDatabaseTool(db=self.db).invoke("")
            infoschema = InfoSQLDatabaseTool(db=self.db).invoke(tns)
            dbschema = sql_metadata.Parser(infoschema).without_comments
        else:  
            dbschema = open(self.schema_file, "r").read()

        self.full_dbschema = sqlparse.format(dbschema, reindent=True, keyword_case='upper')
        dbschema_splitter = RecursiveCharacterTextSplitter(separators=["CREATE"], chunk_overlap=0, chunk_size=30)
        dbschema_chunks = dbschema_splitter.split_text(self.full_dbschema)
        self.vectorstore_dbschema.add_texts(texts=dbschema_chunks, metadatas=[{"type":"dbschema"} for i in range(0,len(dbschema_chunks))])
            
    def set_dbschema(self, dbschema=None, full=True):
        """
        Set the database schema to be used in the prompt
        Either all of it, or just a part of it based on a similarity search
        """

        if not self.vectorstore_dbschema:
            self._init_dbschema()
        
        if dbschema:
            self.dbschema = dbschema
        else:
            if not full:   
                # init embedding model if not already
                if not self.embeddings:
                    self._init_embedding_model()
                resp = self.vectorstore_dbschema.similarity_search_with_relevance_scores(self.question,
                                                                       score_threshold=0, k=6)
                self.dbschema="\n".join([doc[0].page_content for doc in resp])
            else:
                self.dbschema = self.full_dbschema

    def _init_docs(self, reset=False):
        collection_name = "docs"

        # Reset collection if needed
        if reset:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
                
        # Init collection if not already
        if not self.vectorstore_docs:
            c = self.chroma_client.get_or_create_collection(collection_name)
            if not c.count():
                logger.info(f"Collection {collection_name} is empty")

        self.vectorstore_docs = Chroma(client=self.chroma_client, embedding_function=self.embeddings, collection_name=collection_name)
    
    # ADDITIONAL DOCUMENTATION
    def load_split_add_csv(self, file_path, csv_args={'delimiter': '\t'}):
        """
        Load the csv file and split it into chunks to be stored in the collection
        it's reference tag is "docs"
        """
        from langchain_community.document_loaders import CSVLoader

        if not self.vectorstore_docs:
            self._init_docs()    
        
        # Load csv
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_overlap=0)
        loader = CSVLoader(file_path=file_path, csv_args=csv_args)
        data = loader.load()
        all_splits = text_splitter.split_documents(data)
        metadatas=[{"type":"docs"} for i in range(0,len(all_splits))]
        # id_tag = file_path.split("/")[-1].split(".")[0]
        # self.collection.add(ids=[f"{id_tag}{d.model_dump()['metadata']['row']}" for d in all_splits], 
        #                     documents=[d.model_dump()['page_content'] for d in all_splits],
        #                     metadatas=[{"type":"docs"} for i in range(0,len(all_splits))])
        self.vectorstore_docs.add_texts(texts = [d.model_dump()['page_content'] for d in all_splits],
                                       metadatas=metadatas)
        
    def load_split_add_text(self, file_path, split_on=["\n"], chunk_size=300):
        """
        Load the text file and split it into chunks to be stored in the collection
        it's reference tag is "docs"
        """
        from langchain_community.document_loaders import TextLoader

        if not self.vectorstore_docs:
            self._init_docs()  

        # Load csv
        text_splitter = RecursiveCharacterTextSplitter(separators=split_on, chunk_overlap=0)
        loader = TextLoader(file_path=file_path)
        data = loader.load()
        all_splits = text_splitter.split_documents(data)
        metadatas=[{"type":"docs"} for i in range(0,len(all_splits))]
        # id_tag = file_path.split("/")[-1].split(".")[0]
        # self.collection.add(ids=[f"{id_tag}{i}" for i,d in enumerate(all_splits)], 
        #                     documents=[d.model_dump()['page_content'] for d in all_splits],
        #                     metadatas=[{"type":"docs"} for i in range(0,len(all_splits))])
        self.vectorstore_docs.add_texts(texts = [d.model_dump()['page_content'] for d in all_splits],
                                       metadatas=metadatas)
        
    def _retrieve_documents(self, top_k=30, filter={"type":"docs"}):
        """
            Retrieve the documents from the collection stored in the Vector DB
        """
        logger.info(f"Retrieving relevant document")
        # init embedding model if not already
        # if not self.embeddings:
        #     self._init_embedding_model()

        # Read the documents from the collection stored in the Vector DB
        if self.vectorstore_docs:
            retriever = self.vectorstore_docs.as_retriever(search_type="similarity_score_threshold", 
                                             search_kwargs={"score_threshold": 0.05, "k":top_k, "filter":filter}
                                             )
            resp = retriever.invoke(self.question)
            context = "\n".join([doc.page_content for doc in resp])
        else:
            context = ""
        return context

    def _init_examples(self, reset=False):
    # Init vectorstore if not already
        collection_name = "examples"

        # Reset collection if needed
        if reset:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
                
        # Init collection if not already
        if not self.vectorstore_examples:
            c = self.chroma_client.get_or_create_collection(collection_name)
            if not c.count():
                logger.info(f"Collection {collection_name} is empty")
        
        self.vectorstore_examples = Chroma(client=self.chroma_client, embedding_function=self.embeddings, collection_name=collection_name)

    def add_examples(self, examples_dict: pd.DataFrame):
        """
        Add examples to the examples dictionary
        It is used in the fewshot prompt through the example_selector
        """

        if not self.vectorstore_examples:
            self._init_examples()
        
        if self.examples_dict:
            self.examples_dict = pd.concat([self.examples_dict, examples_dict]).drop_duplicates().reset_index(drop=True)
        else:
            self.examples_dict = examples_dict

        # Make sure that there are no duplicates
        logger.info(f"Added examples")

        # Make sure that the question is not in the examples               
        if self.testing:
            temp_examples_dict = examples_dict[examples_dict['question'] != self.question]
        else:
            temp_examples_dict = self.examples_dict

        # init embedding model if not already
        if not self.embeddings:
            self._init_embedding_model()

        # # init collection if not already
        # if not self.collection:
        #     self._init_collection()

        # Add examples to the collection
        # self.collection.add(ids=[f"example{ix}" for ix, r in temp_examples_dict.iterrows()],
        #                     documents=[r['question'] + "\n" + r['sql'] for ix,r in temp_examples_dict.iterrows()],
        #                     metadatas=[{"type":"examples"} for ix,r in temp_examples_dict.iterrows()])
        # for ix, r in temp_examples_dict.iterrows():
            # self.collection.add(ids=[f"example{ix}" ],
            #                 documents=[r['question'] ],
            #                 #documents=[{'question': r['question'], 'sql': r['sql']} ],

            #                 metadatas=[{"type":"examples", "sql": r['sql'], "input": r['question']}])
        # texts = [r["question"] for ix, r in temp_examples_dict.iterrows()]
        # metadatas = [{"sql": r["sql"], "question":r["question"], "type":"examples"} for ix, r in temp_examples_dict.iterrows()]
        # self.vectorstore.add_texts(texts = texts, metadatas=metadatas)

        logger.info(f"Added examples to the vector store")


    
    
        
        # if not self.examples_dict:
        #     print("No examples to select from")
        #     return None, None

        # # init embedding model if not already
        # if not self.embeddings:
        #     self._init_embedding_model()
        
        # # Make sure that the question is not in the examples
        # temp_examples_dict = self.examples_dict
        # for i,ex in enumerate(self.examples_dict):
        #     if ex['question'] == self.question:
        #         temp_examples_dict = self.examples_dict[:i] + self.examples_dict[i+1:]
        #         break

        # example_selector = SemanticSimilarityExampleSelector.from_examples(
        #         # This is the list of examples available to select from.
        #         temp_examples_dict,
        #         # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        #         self.embeddings,
        #         # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        #         # Here I am using a different vector db, because it was a bit suspicious with Chroma
        #         FAISS,
        #         # This is the number of examples to produce.
        #         k=4,
        #     )

        # TODO Deduplicate
        # def add_unique_text(vectorstore, text, metadata, embedding_function):
        # Search for exact match (adjust threshold if needed)
            # results = vectorstore.similarity_search(text, k=1)

            # # Simple deduplication based on high similarity
            # if results and results[0].page_content.strip() == text.strip():
            #     print("Duplicate found, skipping.")
            # else:
            #     vectorstore.add_texts([text], [metadata])
            #     print("Added:", text)


        texts = [f"{r.question}" for ix, r in temp_examples_dict.iterrows()]
        metadatas = [{"sql": r.sql,  "question":r["question"], "type":"examples"} for ix, r in temp_examples_dict.iterrows()]
        self.vectorstore_examples.add_texts(texts = texts, metadatas=metadatas)
        
    def _init_example_selector(self):
        self.example_selector = SemanticSimilarityExampleSelector(
            # vectorstore=self.collection,
            vectorstore=self.vectorstore_examples,
            # vectorstore_kwargs={"collection_name": "tables",
            #                     "search_kwargs":{"score_threshold": 0.05, "k":5, "filter":{"type":"examples"}} },
        )

        # Create a prompt template for the examples
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Question: {question}"),
                ("ai", "SQL: {sql}"),
            ]
                )

        # example_prompt = PromptTemplate(
        #     input_variables=["question", "sql"],
        #     template="Q: {question}\nSQL: {sql}"
        # )


    
    def add_additional_info(self, how_to_use_text="", info=""):
        """
        Add additional information, hints or documentation to the prompt explicitly
        """
        self.extra_documentation = f"\n{how_to_use_text} \n {info}\n" 

    def create_fewshot_prompt(self):
        """
        This creates the prompt for the fewshot chat message
        """
        
        fewshot_prompt = ""
        if self.vectorstore_examples.get()['ids']:
            self._init_example_selector()

            fewshot_prompt = FewShotChatMessagePromptTemplate(
                example_selector=self.example_selector,
                example_prompt=self.example_prompt,
                input_variables=["question"],
            )
        else:
            fewshot_prompt = ChatPromptTemplate.from_messages(
                [           ]
            )

        # Add all other documentation with RAG from collection
        self.extra_documentation


        final_prompt = ChatPromptTemplate.from_messages(
            [ 
                ('system', """You are an assistant trained to generate SQL queries specifically 
        Given a question in natural language, your task is to generate an SQL query that retrieves the requested information from a specified database schema. Use the following database schema details to help craft your query:

        Database Schema: {dbschema}

    Guidelines:

        Analyze the question carefully to identify which tables and columns are relevant.
        Use JOIN statements where necessary, based on relationships between tables.
        Implement any filters, sorting, or grouping as described in the question.
        Do not change any nouns or words in the question that you do not understand.
        Output only the SQL query without additional commentary.
    
    {extra_documentation}
        
        Your Task: For each question provided, generate the SQL query that accurately retrieves the requested information based on the above schema.
    """),
        fewshot_prompt,
        ('human', "New Question: {question}"),
        ("assistant", "Based on the context: {rag_context}"),
            ]
        )

        self.prompt = final_prompt

    def run_with_fewshot_prompt(self, return_prompt=False, full_dbschema=True):
        """
        This puts together the full chain and runs the SQL generation
        For debug purposes we can choose to return only the prompt
        There is a validation prompt and use_query_checker=True
        But I don't know if it is used in the chain
        """

        # Check whether the question was already answered
        hits = self.vectorstore_examples.similarity_search_with_relevance_scores(self.question)
        for hit in hits:
            if hit[1]>0.99:
                # return with the related answer
                response = {'result': hit[0].metadata['sql']}
                return response, "" # gen_sql, final_prompt


        system = """Double check the user's {dialect} query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query.
        If there are no mistakes, just reproduce the original query with no further commentary.
        Remove all the markdown and comments from the query such as "```sql" and "###" ans "<s>"

        Output the final SQL query only."""
        validation_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{query}")]).partial(dialect=self.db.dialect)

        self.set_dbschema(full=full_dbschema)  
        
        self.create_fewshot_prompt()
        
        db_chain = SQLDatabaseSequentialChain.from_llm(self.sql_llm_model, self.db, verbose=False, use_query_checker=True, query_checker_prompt=validation_prompt, return_sql=True)


        final_prompt = self.prompt.format(question=self.question,
                                        dbschema=self.dbschema,
                                        format_description=f"syntactically correct SQL in {self.db.dialect} dialect",
                                        extra_documentation=self.extra_documentation,
                                        rag_context=self._retrieve_documents(),
                                        )
        #print(final_prompt)
        if return_prompt:
            return "", final_prompt
        gen_sql = db_chain.invoke(final_prompt, model_kwargs={"stop":["\n", ";", "Human"]})
        return gen_sql, final_prompt


    def _create_prompt_question_reformulator(self):
        """
        This creates the prompt for the question reformulator
        """

        self.prompt = ChatPromptTemplate.from_messages(
    [
            ("system", """You are a helpful assistant that reformulates the user's question. Your task is to generate {number} 
    different versions of the given user question that are easier to translate to a SQL query.   
    You can use the database schema details to help craft the reformulated questions. This is the schema of the database:  {db_schema}  
             The generated questions should contain keywords related to the column names, table names in this database schema:
             Respond only with the reformulated questions and do not include any explanation or extra information!"""),
        ("human"), "Here is more context for the question {rag_context}",
        ("human", "generate {number} different versions of the question: '{question}'"),
        ("ai", "Generated questions:"),
    ]
    )
        
    def reformulate_question(self, llm, number="three"):
        """
        Creates a number of reformulated questions based on the input question
        """

        self._create_prompt_question_reformulator()
        llm_chain = self.prompt | llm

        # Execute the chain with the input question and chat history
        response = llm_chain.invoke({
                                    "question": self.question, 
                                    "db_schema": self.dbschema,
                                    "number": number,
                                    #"extra_documentation=self.extra_documentation,
                                    "rag_context":self._retrieve_documents(),
                                    "extra_documentation": self.extra_documentation})

        # Print the response from the LLM
        return response



    def _create_prompt_question_splitter(self):
        """
        This creates the prompt for the question splitter
        """

        self.prompt = ChatPromptTemplate.from_messages(
    [
            ("system", """You are a helpful sentence analyst. Your task is to analyse the given question and split it into simpler subquestions.
            Example: 'How many samples contain antibiotic resistance genes linked to Amoxicillin resistance?'
             Answer: 'Which genes have Amoxicillin resistance? Which samples contain these resistance genes? How many of these samples are there?'
             """),
             ])