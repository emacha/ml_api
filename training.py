from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from api import PredictRequest
import numpy as np

SEED = 1234


@dataclass(init=True)
class EnsembleModel:
    linear: LogisticRegression
    gbm: GradientBoostingClassifier
    forest: RandomForestClassifier
    columns: List[str]

    def predict(self, request: PredictRequest) -> float:
        """Return ensemble prediction positive class probability.
        The prediction is a simple mean of constituent models.
         """
        row = [[request.features[k] for k in self.columns]]

        linpred = self.linear.predict_proba(row)[0, 1]
        gbmpred = self.gbm.predict_proba(row)[0, 1]
        forpred = self.forest.predict_proba(row)[0, 1]

        return np.mean([linpred, gbmpred, forpred], axis=0)


def train_model() -> EnsembleModel:
    """Train Ensemble model."""
    data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
    train, test = train_test_split(data, test_size=.2, random_state=SEED, stratify=data.DEATH_EVENT)

    target = "DEATH_EVENT"
    predictors = data.columns.difference([target])

    clf_linear = LogisticRegression(random_state=SEED, max_iter=500)
    clf_gbm = GradientBoostingClassifier(random_state=SEED)
    clf_forest = RandomForestClassifier(random_state=SEED)

    clf_linear.fit(train[predictors], train[target])
    clf_gbm.fit(train[predictors], train[target])
    clf_forest.fit(train[predictors], train[target])

    model = EnsembleModel(linear=clf_linear, gbm=clf_gbm, forest=clf_forest, columns=predictors)
    return model


if __name__ == "__main__":
    model = train_model()
    test_row = {'age': 65.0, 'anaemia': 1.0, 'creatinine_phosphokinase': 52.0, 'diabetes': 0.0, 'ejection_fraction': 25.0, 'high_blood_pressure': 1.0, 'platelets': 276000.0, 'serum_creatinine': 1.3, 'serum_sodium': 137.0, 'sex': 0.0, 'smoking': 0.0, 'time': 16.0, 'DEATH_EVENT': 0.0}
    print(model.predict(PredictRequest(features=test_row)))



