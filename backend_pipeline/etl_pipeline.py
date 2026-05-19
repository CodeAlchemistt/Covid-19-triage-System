import pandas as pd
import numpy as np

def extract_data(connection):
    query = "SELECT * FROM patient_records;"
    df = pd.read_sql(query, connection)
    df = df.drop('id', axis=1) 
    return df

def clean_and_normalize(df):
    print("Cleaning and normalizing clinical data...")
    # 1. Create Target Variable (Mortality Risk)
    # If DATE_DIED is '9999-99-99', patient survived (0). Else, they died (1).
    df['DEATH'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)
    df = df.drop('DATE_DIED', axis=1)

    # 2. Handle Binary Medical Features
    # Convert 2 (No) to 0. Treat missing data (97, 98, 99) as 0 (No) for simplicity.
    binary_columns = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 
                      'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                      'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'INTUBED']
    
    for col in binary_columns:
        df[col] = df[col].apply(lambda x: 0 if x == 2 else x)
        df[col] = df[col].apply(lambda x: 0 if x in [97, 98, 99] else x)

    # 3. Handle Gender (1 = Female, 2 = Male -> Map to 1 and 0)
    df['SEX'] = df['SEX'].apply(lambda x: 0 if x == 2 else 1)

    # 4. Normalize Age (Standard Scaling)
    age_mean = np.mean(df['AGE'])
    age_std = np.std(df['AGE'])
    df['AGE'] = (df['AGE'] - age_mean) / age_std
    
    print("Clinical data cleaned successfully.")
    return df