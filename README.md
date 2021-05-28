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

### Predict

Use the command: 

`curl -X POST localhost:8000/predict --data "@sample_data.json"`

## Overview

The code was developed on python 3.9 using conda. I'm not 100% sure,
but I think it would run up from 3.6 as I only use f-strings. Testing would
be needed though.

All packages required to run the code and tests are in the `conda_env.yml` file.
`black` and `isort` were used for linting, but are not in the file as they are
not strictly necessary.

The API is running on top of the [uvicorn](https://www.uvicorn.org/) server. Which
I picked mostly because it's recommended by FastAPI. 

For the ML architecture, I didn't do anything too involved. 
A linear model (always a good choice), a random forest and one GBM.
2 tree based models are maybe too similar for an ensemble, and I'd rather
trade one for a neural network. But as I didn't do any parameter tuning
besides some _very_ light manual tuning, I thought it wasn't worth it.

All columns were already numeric so no encoding was needed. An 80%
train/test split was used. The data sample was quite small so it would
be better to have used K-fold cross validation, but the test AUC was
quite high already ~90% so I didn't invest the time.

I did not check for the data type when trying to predict, as FastAPI will throw
an error if the wrong data type is passed. I do check for columns though (and ensure they are in the correct order).

On the API side there wasn't much change. I've just implemented `get_model_predictions`.
I've used a `get_model` function to pull the model so that it handles both cases where
a model is already saved or not. Also memoizes the loading, so that expensive I/O or training
is not carried out after every call.

I didn't use any online code. Besides the documentation of packages, which I used extensively!
