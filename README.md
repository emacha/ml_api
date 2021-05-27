# ml_api

## Getting started

We'll use `conda` for environment management in this project. If you don't have it installed, I recommend
using the [miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution.

Create a new virtual environment with required packages by running:
`conda env create -f conda_env.yml`

Activate it with `conda activate ml_api`

## Commands

### Start the server

Run: `uvicorn api:app`

### Training the model

Train the model by running: `python training.py`

The default option does not save the model. Save it by running: `python training.py --save`





#### TODO: fixtures, instructions, lint

