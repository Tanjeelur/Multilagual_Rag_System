import psycopg2
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres.sqnqmlvoyphdfcqplmyf",
    password="IZu8Dd7pItX6jzOt",
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    port="5432"
)
conn.close()
print("Connection successful!")