from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 1234


class IncorrectColumnsError(Exception):
    pass


@dataclass(init=True)
class EnsembleModel:
    linear: LogisticRegression
    gbm: GradientBoostingClassifier
    forest: RandomForestClassifier
    columns: List[str]

    def predict(self, request: "PredictRequest") -> float:
        """Return ensemble prediction positive class probability.
        The prediction is a simple mean of constituent models.
        """
        try:
            row = [[request.features[k] for k in self.columns]]
        except KeyError:
            raise IncorrectColumnsError()

        linpred = self.linear.predict_proba(row)[0, 1]
        gbmpred = self.gbm.predict_proba(row)[0, 1]
        forpred = self.forest.predict_proba(row)[0, 1]

        return np.mean([linpred, gbmpred, forpred], axis=0)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        return cls(*joblib.load(path))

    def save(self, path: Path):
        joblib.dump([self.linear, self.gbm, self.forest, self.columns], path)


def train_model() -> EnsembleModel:
    """Train Ensemble model."""
    data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
    train, test = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=data.DEATH_EVENT
    )

    target = "DEATH_EVENT"
    predictors = data.columns.difference([target])

    clf_linear = LogisticRegression(random_state=SEED, max_iter=500)
    clf_gbm = GradientBoostingClassifier(random_state=SEED)
    clf_forest = RandomForestClassifier(random_state=SEED)

    clf_linear.fit(train[predictors], train[target])
    clf_gbm.fit(train[predictors], train[target])
    clf_forest.fit(train[predictors], train[target])

    model = EnsembleModel(
        linear=clf_linear, gbm=clf_gbm, forest=clf_forest, columns=predictors
    )
    return model


@lru_cache
def get_model() -> EnsembleModel:
    """Load trained model if saved, otherwise train it."""
    try:
        return EnsembleModel.load(Path("ensemble_model"))
    except FileNotFoundError:
        return train_model()


def main(save: bool = False):
    """Train the model. Use --save to save to disk."""
    typer.echo("Training model...")
    model = train_model()
    if save:
        model_path = Path("ensemble_model").absolute()
        typer.echo(f"Saving model to {model_path}")
        model.save(model_path)


if __name__ == "__main__":
    typer.run(main)
