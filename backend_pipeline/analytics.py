def prepare_ml_data(df):
    # y is our Target (DEATH: 0 or 1)
    y = df['DEATH']
    
    # X is our Features (Everything else)
    X = df.drop('DEATH', axis=1)
    
    print(f"Data split into X (Features: {X.shape}) and y (Target: {y.shape}) sets.")
    return X, y