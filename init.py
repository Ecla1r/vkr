import psycopg2


SQL1 = '''CREATE TABLE IF NOT EXISTS test(act int DEFAULT 0,\
sat int DEFAULT 0, par_grad_lv varchar DEFAULT '',\
par_income int DEFAULT 0, hs_gpa float DEFAULT 0,\
col_gpa float DEFAULT 0, yrs_grad int DEFAULT 0);'''

SQL2 = '''CREATE TABLE IF NOT EXISTS test1(gender varchar DEFAULT '',\
height float DEFAULT 0, weight float DEFAULT 0);'''


if __name__ == '__main__':
    conn = psycopg2.connect("dbname=correlation_test user=postgres password=postgres")
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(SQL2)
    print("Table created!")

    with open('graduation_rate.csv', 'r') as f:
        next(f)
        cur.copy_from(f, 'test1', sep=',', null="")
    print("Values inserted!")

    conn.commit()
    conn.close()
