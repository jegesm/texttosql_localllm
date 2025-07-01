from datetime import datetime
import logging

import pandas as pd
import sqlvalidator
import sqlite3

import warnings
warnings.filterwarnings("ignore")

# Setup logging
logformat = 'Txt2SQL-utils: %(name)s - %(message)s - %(lineno)d'
logging.basicConfig(level=logging.INFO, format=logformat)
logger = logging.getLogger(__name__)

# Add function name to log records
class FunctionNameFilter(logging.Filter):
    def filter(self, record):
        record.funcName = record.funcName
        return True

logger.addFilter(FunctionNameFilter())


def read_dict_from_file(file_path):
    """
    For reading some paramaters from a file
    """
    import json

    # Read the dictionary from a file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Print the dictionary
    return data

def init_test_db(prefix):
    """
    Initialize the SQLite database for storing the test results and train data if it does not exist
    """
    # Connect to SQLite database
    conn = sqlite3.connect(f"{prefix}_test.db")
    cursor = conn.cursor()

    # Create 'results' table
    cursor.execute(f'''
    CREATE TABLE result{param} (
        resultid INTEGER AUTO_INCREMENT PRIMARY KEY,
        question TEXT,
        number_of_max_steps INTEGER,
        model_name TEXT,
        sql_query TEXT,
        answer TEXT,
        error TEXT,
        accuracy_score INTEGER
    )
    ''')

    # Create 'train' table
    cursor.execute('''
    CREATE TABLE train (
        trainid INTEGER AUTO_INCREMENT PRIMARY KEY,
        question TEXT,
        sql_query TEXT,
        answer TEXT
        )
    ''')
    
    # Create 'test' table
    cursor.execute('''
    CREATE TABLE test (
        testid INTEGER AUTO_INCREMENT PRIMARY KEY,
        question TEXT,
        sql_query TEXT,
        answer TEXT
        )
    ''')

    # Commit changes and close connection
    conn.commit()
    conn.close()

def create_additional_tables(prefix, param=None):
    """
    In the SQlite database we can create additional tables to store the results
    """
    # Connect to SQLite database
    conn = sqlite3.connect(f"{prefix}_test.db")
    cursor = conn.cursor()

    cursor.execute(f'''
    CREATE TABLE result{param} (
        resultid INTEGER AUTO_INCREMENT PRIMARY KEY,
        question TEXT,
        number_of_max_steps INTEGER,
        model_name TEXT,
        sql_query TEXT,
        answer TEXT,
        error TEXT,
        accuracy_score INTEGER
    )
    ''')
        
    # Commit changes and close connection
    conn.commit()
    conn.close()

def upload_train_data_db(prefix, df):
    """
    Append existing train data to the SQLite database
    """
    # Connect to SQLite database
    conn = sqlite3.connect(f"{prefix}_test.db")

    df[['question', 'sql_query', 'answer']].to_sql( 'train', conn, if_exists='append', index=False)

    # Commit changes and close connection
    conn.commit()
    conn.close()

def upload_result(prefix, param, results):
    """
    Upload the results of the tests to the SQLite database
    """

    # Connect to SQLite database
    conn = sqlite3.connect(f"{prefix}_test.db")
    
    df = pd.DataFrame([results])
    df.sql_query = df.sql_query.iloc[0][-1]
    if "error" in df.columns:
        df.error = df.error.iloc[0][-1]
    else:
        df['error'] = ""

    if "prompt" not in df.columns:
        df['prompt'] = 0

    if "accuracy_score" not in df.columns:
        df['accuracy_score'] = 0

    if "number_of_max_steps" not in df.columns:
        df['number_of_max_steps'] = 0
    
    if "model_name" not in df.columns:
        df['model_name'] = "unknown"


    df[['qid', 'question', 'sql_query', 'answer', 'number_of_max_steps', 'prompt', 'model_name', 'error', 'accuracy_score']].to_sql( f"result{param}", conn, if_exists='append', index=False)

    # Commit changes and close connection
    conn.commit()
    conn.close()

def get_train_data(dbname):
    """
    Retrieve the train table from the Sqlite database 
    """
    import pandas as pd
    # Connect to the SQLite database
    conn = sqlite3.connect(dbname)

    # Retrieve the train table
    train_df = pd.read_sql_query("SELECT * FROM train", conn)

    # Close the connection
    conn.close()

    # Display the retrieved train table
    return train_df

def get_result_from_db(prefix, param):
    """
    Once a set of tests were run, we can retrieve the results from the database
    The results are saved in a table called 'result{param}' in the Sqlite database
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(f"{prefix}_test.db",  timeout=2)
    df = pd.read_sql(f"SELECT * FROM result{param}", conn)
    conn.commit()
    conn.close()
    #df.drop(columns=['number_of_max_steps'], inplace=True)

    return df

def check_if_already_answered(dbname, param, question, sql_llm_model_name, temperature=None, number_of_max_steps=None):
    """
    When rerunning a set of tests it is useful to check if a question has already been answered and if so,
    then we can skip it and move on to the next question
    """
    import pandas as pd

    # Connect to the SQLite database
    conn = sqlite3.connect(dbname)

    answered = False
    error = []
    # Retrieve the results table
    train_df = pd.DataFrame()
    where_query = f"WHERE question = \"{question}\" and model_name = \"{sql_llm_model_name}\""
    if temperature:
        #where_query += f" and temperature = {temperature}"
        pass
    if number_of_max_steps:
        where_query += f" and number_of_max_steps = {number_of_max_steps}"
    try:
        train_df = pd.read_sql_query(f"SELECT * FROM result{param} WHERE question = \"{question}\" and model_name = \"{sql_llm_model_name}\"", conn)
    except:
        answered = False

    # Close the connection
    conn.close()

    if train_df.size:
        answered = True
    else:
        answered = False

    if temperature:
        print(train_df)

    # Get error messages if any
    if answered:
        error = train_df['error'].values

    return answered, error


def update_accuracy_score_single(prefix, param, question, answer, accuracy_score=5):
    """
        This might be used to update the accuracy score of the results in the database 
        for a single question and answer pair
    """

    conn = sqlite3.connect(f"{prefix}_test.db")
    cursor = conn.cursor()
    # print(question, answer)
    cursor.execute(f'''
    UPDATE result{param}
    SET accuracy_score = {accuracy_score}
    WHERE question = "{question}"
    AND answer = "{answer}"
    ''')

    conn.commit()
    conn.close()

def update_accuracy_score(prefix, param, etalon):
    """
    This might be used to update the accuracy score of the results in the database
    but it is not used in the current implementation
    """
    df = get_result_from_db(prefix, param)
    for index, result_row in df.iterrows():
        if check_answer(etalon, result_row):
                update_accuracy_score_single(prefix, param, result_row['question'], result_row['answer'])
    
def check_answer_single(etalon, answer):
    """
    etalon is a pandas dataframe that contains the correct answers and this function checks 
    for a single question if the answer is correct
    The answers might be in a different format, so we need to check for that as well if the answer is not correct
    """
    if etalon == answer:
        return True
    try:
        logger.info(f"Answer: {answer} - Correct answer: {etalon}")
        answer = eval(answer)
        if answer:
            if str(etalon) == str(answer[0][0]):
                return True
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Problematic: Original: '{answer}' -- Etalon: '{etalon}'")
        

    return False

def check_answer(etalon, test):
    """
    If etalon contain the test answers then don't check accuracy score
    If we want to compare it with new results, where the answers have been manually checked, 
    then use the accuracy score==5 condition as well with the "with_score" parameter
    """
    result = False

    try:
        answer = test.answer
        question = test.question
    except:
        try:
            answer = test['answer']
            question = test['question']
        except: 
            result = etalon['answer'].apply(lambda x: x==test).any()
            return result
        
    #print(answer,test)
    for index, row in etalon.iterrows():
        if (row['question'] == question):
            return check_answer_single(row['answer'], answer)

    
def count_accuracy(df):
    """
    Simply reads the table into a dataframe and counts the number of correct answers
    for each model_name
    """

    return df.shape[0], df[df.accuracy_score==5].shape[0], df.accuracy_score.sum()/(len(df)*5)*100
      


def postprocess_sql(sql, debug=False):
    """
    Theoretically the outputs of the llm chains should be structured, but sometimes they are not
    and additional quotes or other characters are added to the beginning and end of the query so a 
    postprocessing step is needed to remove these characters and be able to submit the query to the database
    """

    try:
        sql = sql.split("```")[1][3:]
    except:
        pass

    try:
        sql = sql.split("###")[0]
    except:
        pass

    try:
        sql = sql.split("SQLQuery:")[1]   
    except:
        pass

    try:
        sql = sql.split("SQL:")[1]   
    except:
        pass

    try:
        sql = sql.split(";\n")[0]
    except:
        pass

    try:
        sql = sql.split("AI:")[1]
    except:
        pass
    
    sql = sql.replace("<s>", "")
    
    if debug:
        print(sql)
    psql = sqlvalidator.parse(sql)
    try:
        if psql.is_valid():
            return sql
        else:
            return psql.errors
    except:
        return sql

