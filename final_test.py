import os
import joblib
import pandas as pd
from pipeline import mahoraga  # pulling in the brain factory logic
from sklearn.metrics import recall_score

def run_mahoraga_verification(old_model_path, data_path):
    # check if a model even exists. if not, force a training session to build one.
    if not os.path.exists(old_model_path):
        print(f"{old_model_path} not found. forcing initial training...")
        mahoraga.auto_retrain(data_path) 

    # 1. load the current brain.
    package = joblib.load(old_model_path)
    old_model = package['model']
    old_features = package['features']
    
    # 2. load the new data for testing.
    df_path = pd.read_csv(data_path)
    # hunt for the binary target (0/1) in the new file.
    target = [c for c in df_path.columns if df_path[c].nunique() == 2 and 'id' not in c.lower()][-1]
    
    # --- safety hatch: schema check ---
    # verifying if the old features actually exist in this new file.
    missing_features = [f for f in old_features if f not in df_path.columns]
    
    if missing_features:
        print(f"schema change detected! missing features: {missing_features}")
        print("old model is physically incompatible. mahoraga is evolving to the new schema...")
        # skip the test and go straight to rebuilding the model.
        mahoraga.auto_retrain(data_path) 
        old_recall = 0.0 # placeholder
    else:
        # 3. test old model. only runs if the column names match up.
        X_old = df_path[old_features]
        y_old_pred = old_model.predict(X_old)
        old_recall = recall_score(df_path[target], y_old_pred)
        print(f"old model recall on new data: {old_recall:.2f}")

    # 4. adaptation protocol.
    # if columns matched but the score is trash, we force an evolution.
    if not missing_features and old_recall < 0.90:
        print("warning! accuracy too low! mahoraga is adapting to the new trends...")
        mahoraga.auto_retrain(data_path) 
    elif old_recall >= 0.90:
        print("old model is still performing well. no adaptation needed.")
        return

    # 5. load the newly evolved brain.
    new_package = joblib.load('models/active_model.pkl')
    new_model = new_package['model']
    new_features = new_package['features']
    
    # 6. final verification run.
    # checking the new model against the same data to confirm improvement.
    X_new = df_path[new_features]
    y_new_pred = new_model.predict(X_new)
    new_recall = recall_score(df_path[target], y_new_pred)
    
    print(f"adapted model recall on new data: {new_recall:.2f}")
    print("adaptation complete. the system has evolved to the new dataset.")

# run the verification loop
run_mahoraga_verification('models/active_model.pkl', 'creditcard_2023.csv')