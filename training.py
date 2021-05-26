from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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
        row = [[request.features[k] for k in self.predictors]]

        linpred = self.linear.predict_proba(row)[0, 1]
        gbmpred = self.gbm.predict_proba(row)[0, 1]
        forpred = self.forest.predict_proba(row)[0, 1]

        return np.mean([linpred, gbmpred, forpred], axis=0)


def train() -> EnsembleModel:
    """Train Ensemble model."""
    data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
    train, test = train_test_split(data, test_size=.2, random_state=SEED, stratify=data.DEATH_EVENT)

    target = "DEATH_EVENT"
    predictors = data.columns.difference([target])

    clf_linear = LogisticRegression(random_state=SEED, max_iter=500)
    clf_linear.fit(train[predictors], train[target])

    # .91 AUC, quite good
    #roc_auc_score(test[target], clf_linear.predict_proba(test[predictors])[:, 1])

    clf_gbm = GradientBoostingClassifier(random_state=SEED)
    clf_gbm.fit(train[predictors], train[target])
    print(roc_auc_score(test[target], clf_gbm.predict_proba(test[predictors])[:, 1]))

    clf_forest = RandomForestClassifier(random_state=SEED)
    clf_forest.fit(train[predictors], train[target])
    print(roc_auc_score(test[target], clf_forest.predict_proba(test[predictors])[:, 1]))

    model = EnsembleModel(linear=clf_linear, gbm=clf_gbm, forest=clf_forest, columns=predictors)
    return model






