import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class RandomForestModel:

    def __init__(self, train_url, random_state=42):
        self.train_url = train_url

        self.random_state = random_state

        self.train_df = pd.read_csv(self.train_url, index_col=0)


        self.label_encoding = LabelEncoder()

        self.X_train = self.train_df.drop('target', axis=1).replace([np.inf, -np.inf], np.nan)
        self.y_train = self.label_encoding.fit_transform(self.train_df['target'])


    def create_pipeline(self):
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        clf = RandomForestClassifier(
            criterion='entropy',
            max_depth=10,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=3,
            n_estimators=30,
            random_state=self.random_state
        )

        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('classifier', clf)
        ])
        return pipeline

    def run(self):
        pipeline = self.create_pipeline()

        # Perform cross-validation
        cv_results = cross_val_score(pipeline, self.X_train, self.y_train, cv=5)
        cv_mean = cv_results.mean()



train_dataset_url = "https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/train_dataset.csv"


model = RandomForestModel(train_dataset_url)
model.run()
