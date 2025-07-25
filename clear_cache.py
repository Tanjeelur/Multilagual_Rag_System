from src.storage import Storage
from src.config import DB_CONFIG, REDIS_CONFIG

def clear_cache():
    try:
        storage = Storage(DB_CONFIG, REDIS_CONFIG)
        with storage.conn.cursor() as cursor:
            cursor.execute('DELETE FROM chunks')
            storage.conn.commit()
        print('Database cache cleared successfully')
    except Exception as e:
        print(f'Error clearing cache: {str(e)}')

if __name__ == '__main__':
    clear_cache()