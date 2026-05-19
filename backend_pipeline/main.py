from db_setup import create_db_connection, setup_schema, insert_data_in_batches
from etl_pipeline import extract_data, clean_and_normalize
from analytics import prepare_ml_data
from model_training import train_and_evaluate_models
import os

def main():
    # --- PHASE 1: Setup ---
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "passion"
    
    print("Connecting to MySQL...")
    conn = create_db_connection(DB_HOST, DB_USER, DB_PASSWORD) 
    
    if conn:
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS covid_risk_db;")
        conn.database = "covid_risk_db"
        
        setup_schema(conn)
        
        # Execute Batch Ingestion (Looking inside the 'data/' folder)
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Covid_Data.csv'))
        
        #insert_data_in_batches(conn, csv_path, batch_size=50000)
        
        print("Extracting clinical data from database...")
        raw_df = extract_data(conn)
        cleaned_df = clean_and_normalize(raw_df)
        
        # --- PHASE 3: ML Prep & Training ---
        X, y = prepare_ml_data(cleaned_df)
        
        print("Initiating Machine Learning Models...")
        train_and_evaluate_models(X, y)
        
        conn.close()
        print("Medical Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()