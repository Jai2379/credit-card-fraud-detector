import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class FraudPipeline:
    def __init__(self, model_path='models/active_model.pkl'):
        self.model_path = model_path

    def auto_retrain(self, data_path):
        print(f"trend detected! re-training on {data_path}...")
        df = pd.read_csv(data_path)

        # 1. ghost feature logic: find any numbers we can work with.
        # we pull only numeric columns and ditch anything that looks like noise or metadata.
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        
        # dna test: hunting for the binary target column (the fraud/genuine answer).
        target = [c for c in df.columns if df[c].nunique() == 2 and 'id' not in c.lower()][-1]
        
        # filter: keep the meat, cut the fat. no ids, row counts, or timestamps.
        exclude = [target, 'id', 'rowid', 'time', 'unnamed: 0']
        features = [col for col in all_numeric if col.lower() not in exclude]
        
        X = df[features]
        y = df[target]

        print(f"analyzing {len(features)} potential features...")

        # 2. dynamic feature selection.
        # if the dataset is tiny, we just take everything. otherwise, we grab the top 6 red flags.
        k_val = min(len(features), 6) 
        selector = SelectKBest(score_func=f_classif, k=k_val)
        
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        # 3. smote: balancing the scales.
        # we need to fake some fraud cases so the model doesn't get biased toward genuine transactions.
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_new, y)

        # 4. the evolved brain: training the random forest.
        # locked to max_depth 10 to stop it from just memorizing the noise.
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_res, y_res)

        # 5. boxing up the package.
        # we save the feature names here so the live check knows exactly what to look for later.
        joblib.dump({
            'model': model, 
            'features': selected_features,
            'target_name': target
        }, self.model_path)
        
        print(f"adaptation complete! mahoraga now relies on: {selected_features}")

# boot up the pipeline instance
mahoraga = FraudPipeline()