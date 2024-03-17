from os.path import join, dirname
from dotenv import load_dotenv
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List
import sqlite3

conn = sqlite3.connect('db.sqlite')

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join([row[0] for row in rows if row[0]is not None])

dotenv_path = join(dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path)

def run_sqlite_query(query: str):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"
    
class RunQueryArgsSchema(BaseModel):
    query: str

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="runs a sqlite query",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join(f"'{table_name}'" for table_name in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="describes the schema of a table",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)
