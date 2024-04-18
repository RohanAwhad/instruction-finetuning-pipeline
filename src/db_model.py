import os

from peewee import *
from loguru import logger

DB_PATH = os.environ["DB_PATH"]

logger.info(f'Loading database {DB_PATH}...')
DB = SqliteDatabase(DB_PATH)

class BaseModel(Model):
    class Meta:
        database = DB


class DataInstance(BaseModel):
    input = TextField()
    target = TextField()
    task = TextField()
    dataset = TextField()


def create_tables():
    with DB:
        DB.create_tables([DataInstance])


def drop_all_tables():
    with DB:
        DB.drop_tables([DataInstance])


def load_data(dataset):
    '''Assumes dataset is list of dict with keys similar to column names'''
    if not DB.table_exists(DataInstance): create_tables()
    with DB.atomic():
        DataInstance.insert_many(dataset).execute()

