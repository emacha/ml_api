from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import typer
from pathlib import Path
import joblib
from functools import lru_cache


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
        linear = joblib.load(path / "linear.joblib")
        gbm = joblib.load(path / "gbm.joblib")
        forest = joblib.load(path / "forest.joblib")
        columns = joblib.load(path / "columns.joblib")
        return cls(linear, gbm, forest, columns)

    def save(self, path: Path):
        path.mkdir(exist_ok=True)
        joblib.dump(self.linear, path / "linear.joblib")
        joblib.dump(self.gbm, path / "gbm.joblib")
        joblib.dump(self.forest, path / "forest.joblib")
        joblib.dump(self.columns, path / "columns.joblib")


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


@lru_cache
def get_model() -> EnsembleModel:
    """Load trained model if saved, otherwise train it."""
    try:
        return EnsembleModel.load(Path("ensemble_model"))
    except FileNotFoundError:
        return train_model()


def main(save: bool = False):
    typer.echo("Training model...")
    model = train_model()
    if save:
        model_path = Path("ensemble_model").absolute()
        typer.echo(f"Saving model to {model_path}")
        model.save(model_path)


if __name__ == "__main__":
    typer.run(main)
