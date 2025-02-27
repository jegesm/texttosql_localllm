from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool, QuerySQLDataBaseTool
from langchain.prompts import PromptTemplate

from sqlalchemy.exc import OperationalError

class DatabaseInspector:
    def __init__(self, ollm, db_connection, extra_documentation=""):
        self.ollm = ollm
        self.connection = db_connection
        #self.cursor = self.connection.cursor()
        self.extra_documentation = extra_documentation

    def get_tables_str(self):
        """
        Use Langchain utils
        return a string with the table names
        """        
        return ListSQLDatabaseTool(db=self.connection).invoke("")

    def get_tables_list(self):
        """
        Use Langchain utils
        return a list with the table names
        """
        return [table for table in self.get_tables_str().split(", ") ]

    # Only for PostgreSQL
    def get_columns(self, table_name):
        if self.connection.dialect == "postgresql":
            """
            Simply send a query to do it
            """
            query= f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' and table_schema = '{self.connection._schema}';
            """
            try:
                db_ans = self.connection.run(query)
                #print(self.connection.dialect,query, db_ans)
                columns = [column[0] for column in eval(db_ans)]
                return columns
            except:
                return []
            
        else:
            first_row = self.connection.run(f"SELECT * FROM {table_name} LIMIT 1;", include_columns=True)
            
            try:
                import datetime
                columns = eval(first_row)[0].keys()
            except NameError:
                pass
            return columns
        
    def get_table_columns(self):
        tcl={}
        for t in self.get_tables_list():
            if t[0] == "_":
                continue
                
            tcl[t]=[]
            for c in self.get_columns(t):
                #print(f"{t} - {c}")
                tcl[t].append(c)
            if tcl[t] == []:
                del tcl[t]
        return tcl
        
    def get_foreign_keys(self, table_name):
        """
        Simply send a query to do it
        """
        foreign_keys = self.connection.run(f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name='{table_name}';
        """)
        
        return foreign_keys

    def describe_table(self, table_name):
        """
        Simply send a query to do it
        """
        columns = self.get_columns(table_name)
        foreign_keys = self.get_foreign_keys(table_name)
        description = f"Table '{table_name}' has columns: {', '.join(columns)}.\n"
        if foreign_keys:
            description += "Foreign keys:\n"
            for fk in foreign_keys:
                description += f"  - {fk[0]} references {fk[1]}({fk[2]})\n"
        else:
            description += "No foreign keys.\n"
        return description

    def describe_database(self, table_names=None):
        """
        Use Langchain utils
        return schema for given tables
        """
        if table_names is None:
            table_names = self.get_tables_str()
        
        return InfoSQLDatabaseTool(db=self.connection, description = 'Get the schema for the specified SQL tables.').invoke(table_names)

    def get_random_n_rows(self, table_name, n=5):
        """
        Simply send a query to do it
        """
        table_length_long = False
        # Check the number of rows in the table
        row_count_query = f"SELECT COUNT(*) FROM {table_name};"
        row_count = 10001
        try:
            row_count = eval(self.connection.run(row_count_query))[0][0]
        except OperationalError:
            table_length_long = True

        # Adjust 'n' if the table has fewer rows than 'n'
        if row_count < n:
            n = row_count
        
        if row_count > 10000:
            table_length_long = True

        if table_length_long:
            return self.connection.run(f"SELECT * FROM {table_name} TABLESAMPLE SYSTEM(1) LIMIT {n};")
        else:
            return self.connection.run(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {n};")


    def generate_description_table(self, table_name):
        """
        Use AI to generate a description of the table
        returns with textual description

        # Add any extra documentation or notes here.
        # For example, you can describe the purpose of the table, 
        # any specific naming conventions used, or any other relevant information.
        """

        description = self.describe_database()
        db_context = self.connection.get_context()

        prompt = PromptTemplate( template="""
        You are an AI agent tasked with providing a detailed description of a table in a PostgreSQL database. 
        The description should contain only the information about the table itself, not the data within it. 
        The information is intended to be used as by an sql query building AI agent and should be concise and informative to understand the database structure.

        The database contains the following tables and their respective columns and relationships:
        
        {db_context}
        
        {extra_documentation}
        
        Please provide a comprehensive and meaningful description of the {table_name} database structure, 
        If there is no relationship with other tables then do not mention it.
        Do not analyse the values in the column, just describe the table.
        Do not use any headers or titles in the response.
        Do not guess or assume any information.

        Answer:
        Table: {table_name}
        followed by the description        """,
        input_variables=["table_name", "extra_documentation", "db_context", "random_values"],
        )

        # Create a chain with the ollama model and the prompt template
        chain = prompt | self.ollm
        #chain = LLMChain(llm=self.ollm, prompt=template)

        random_values = self.get_random_n_rows(table_name, 50)    
    
        # Generate a response using the provided prompt
        response = chain.invove({"table_name": table_name,
                                  "db_context": db_context,
                                  "random_values": random_values,
                                  "extra_documentation": self.extra_documentation})

        return response

    def generate_description_column(self, table_name, column_name):
        """
        Use AI to generate a description of the table
        returns with textual description

        # Add any extra documentation or notes here.
        # For example, you can describe the purpose of the database, 
        # any specific naming conventions used, or any other relevant information.
        """

        db_context = self.connection.get_context()

        prompt = PromptTemplate(
            template="""
        You are an AI agent tasked with providing a detailed description of a column in a table in a PostgreSQL database. 
        The description should contain only the information about the column itself, not the data. 
        The information is intended to be used as by an sql query building AI agent and should be concise and informative to understand the database structure.
        The database contains the following tables and their respective columns and relationships:
        
        {db_context}
        
        {extra_documentation}

        Randomly selected values from the {table_name} table are as follows:
        {random_values}
        
        Please provide a comprehensive and meaningful description of the {column_name} in the {table_name} table.
        If there is no relationship with other tables then do not mention it.
        Do not analyse the values in the column, just describe the column.
        Do not use any headers or titles in the response.
        Do not guess or assume any information.

        Answer:
        Table: {table_name}, Column: {column_name}
        followed by the description
        """,
        input_variables=["table_name", "column_name", "extra_documentation", "db_context", "random_values"],
        )

        random_values = self.get_random_n_rows(table_name, 50)

        # Create a chain with the ollama model and the prompt template
        chain = prompt | self.ollm

        # Generate a response using the provided prompt
        response = chain.invoke({"table_name": table_name,
                                  "column_name": column_name,
                                  "db_context": db_context,
                                  "random_values": random_values,
                                  "extra_documentation": self.extra_documentation})

        return response

    def generate_description(self):
        """
        Generates a description for each table and column in the database
        returns with one row for each column
        """
        # Get all table names
        tables = self.get_tables_list()

        # Iterate over each table and get descriptions for each column
        with open(f"database_gen_description_{self.ollm.model}.txt", "w") as file:
            for table in tables:
                columns = self.get_columns(table)
                file.write(f"Table: {table}\n")
                for column in columns:
                    print(f"Generating description for column {column} in table {table}")
                    description = self.generate_description_column(table_name=table, column_name=column)
                    file.write(f"Column: {column}\n")
                    file.write(description + "\n")
        file.write("\n")

 
        
    def close(self):
        self.connection.close()
