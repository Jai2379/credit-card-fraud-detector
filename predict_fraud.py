import joblib
import pandas as pd

def check_transaction(new_data_dict):
    # pull in the latest 'brain' mahoraga just cooked up.
    # we need the model + the specific feature list it decided was important.
    package = joblib.load('models/active_model.pkl')
    model = package['model']
    required_features = package['features']

    # safety net: if the incoming data is missing columns the model expects (common with new datasets), 
    # we patch the holes with 0.0 so the code doesn't explode.
    for feature in required_features:
        if feature not in new_data_dict:
            new_data_dict[feature] = 0.0  # neutral filler

    # flatten the dictionary into a dataframe so pandas can handle the math.
    df = pd.DataFrame([new_data_dict])

    # lockdown: ensure we're only feeding the model the exact features it was trained on. 
    # everything else is just noise at this point.
    X_live = df[required_features]

    # time for the verdict.
    prediction = model.predict(X_live)[0]

    # pulling the raw probability score because the 0/1 binary is too vague.
    # i want to see exactly how suspicious the ai is.
    probability = model.predict_proba(X_live)[0][1] 
    print(f"there is a {probability * 100:.2f}% chance this is fraud.")
    
    if prediction == 1:
        return "alert: this looks like fraud!"
    else:
        return "clear: this transaction is likely genuine."

# --- sandbox testing ---
# dumping some dummy data here to stress-test the model's logic.
'''test_transactions = [
    # 1. obvious fraud: hitting it with heavy negative v-scores (classic fraud indicators).
    {
        'V1': -2.31, 'V2': 1.65, 'V3': -3.12, 'V10': -4.52, 'V12': -5.11, 
        'V14': -6.21, 'V16': -2.33, 'V17': -5.01, 'Amount': 99.99
    },
    # 2. standard daily buy: normal amounts, clean positive v-scores.
    {
        'V1': 1.12, 'V2': -0.15, 'V3': 0.44, 'V10': 0.12, 'V12': 0.55, 
        'V14': 0.21, 'V16': 0.05, 'V17': 0.11, 'Amount': 12.50
    },
    # 3. high-ticket item: let's see if a massive 'amount' trips the alarm by itself.
    {
        'V1': -0.52, 'V2': 0.98, 'V3': 1.15, 'V10': -0.11, 'V12': 0.23, 
        'V14': 0.45, 'V16': -0.12, 'V17': 0.08, 'Amount': 2500.00
    },
    # 4. stealth fraud: moving the numbers slightly to see if the model has 'high sensitivity'.
    {
        'V1': -1.05, 'V2': 0.42, 'V3': -0.88, 'V10': -1.20, 'V12': -1.50, 
        'V14': -1.80, 'V16': -0.90, 'V17': -1.30, 'Amount': 45.00
    },
    # 5. grocery run: boring, genuine transaction pattern.
    {
        'V1': 2.05, 'V2': -0.01, 'V3': -1.22, 'V10': -0.05, 'V12': 0.88, 
        'V14': 0.12, 'V16': 0.44, 'V17': -0.21, 'Amount': 64.12
    }
]

# quick loop to cycle through the test cases.
for single_tx in test_transactions:
    print(f"\n--- running real-time scan ---")
    result = check_transaction(single_tx)
    print(result)'''