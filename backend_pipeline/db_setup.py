import mysql.connector
from mysql.connector import Error
import pandas as pd

def create_db_connection(host_name, user_name, user_password, db_name=None):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def setup_schema(connection):
    cursor = connection.cursor()
    # Create the table matching the Kaggle COVID-19 Clinical dataset
    create_table_query = """
    CREATE TABLE IF NOT EXISTS patient_records (
        id INT AUTO_INCREMENT PRIMARY KEY,
        USMER INT,
        MEDICAL_UNIT INT,
        SEX INT,
        PATIENT_TYPE INT,
        DATE_DIED VARCHAR(20),
        INTUBED INT,
        PNEUMONIA INT,
        AGE INT,
        PREGNANT INT,
        DIABETES INT,
        COPD INT,
        ASTHMA INT,
        INMSUPR INT,
        HIPERTENSION INT,
        OTHER_DISEASE INT,
        CARDIOVASCULAR INT,
        OBESITY INT,
        RENAL_CHRONIC INT,
        TOBACCO INT,
        CLASIFICACION_FINAL INT,
        ICU INT
    );
    """
    try:
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'patient_records' created successfully.")
    except Error as err:
        print(f"Error: '{err}'")

def insert_data_in_batches(connection, csv_path, batch_size=20000):
    cursor = connection.cursor()
    print("Starting data ingestion. This will take a few minutes for 1M+ rows...")
    
    # Read CSV in chunks
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        data = [tuple(x) for x in chunk.to_numpy()]
        
        # 21 clinical features
        placeholders = ', '.join(['%s'] * 21) 
        insert_query = f"""
        INSERT INTO patient_records (
            USMER, MEDICAL_UNIT, SEX, PATIENT_TYPE, DATE_DIED, INTUBED, PNEUMONIA, 
            AGE, PREGNANT, DIABETES, COPD, ASTHMA, INMSUPR, HIPERTENSION, 
            OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO, 
            CLASIFICACION_FINAL, ICU
        ) VALUES ({placeholders})
        """
        try:
            cursor.executemany(insert_query, data)
            connection.commit()
        except Error as err:
            print(f"Error inserting batch: '{err}'")
            
    print("Clinical data ingestion complete.")