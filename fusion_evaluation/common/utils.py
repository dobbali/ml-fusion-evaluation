import base64
from typing import Dict, List
import json
import pandas as pd
import numpy as np
from six.moves import cPickle as pickle
from sqlalchemy.engine import Engine


def qry_to_pd(qry: str, engine: Engine):
    return pd.read_sql(qry, engine)


def read_model(pickled_dict: Dict):
    """Returns model and features from pickled dict"""
    dict_model = json.loads(pickled_dict)  # type: ignore
    model = pickle.loads(base64.b64decode(dict_model['model_spec']))
    feats = dict_model['features']
    return model, feats


def get_model_and_features(path_to_model: str):
    """Read latest model from artifactory/file path"""
    with open(path_to_model, 'r') as reader:
        pickled_model = reader.read()
    model, features_string = read_model(pickled_model)  # type: ignore
    features = [s.strip() for s in features_string.split(',')]
    return model, features


def get_evaluation_dataset(model_name: str, engine: Engine) -> pd.DataFrame:
    """Returns sample evaluation dataset from Snowflake"""
    model_selection = {'bp': 'SELECT * FROM EBATES_PROD.TEMP.tas_basket_eval_final_e limit 100000',
                       'cvr': 'SELECT * FROM EBATES_PROD.TEMP.tas_cvr_eval_final_e limit 100000'}
    query = model_selection[model_name]
    return qry_to_pd(query, engine)


def get_clean_XY(df: pd.DataFrame, features: List, target_variable: str):
    """Returns X and y matrix"""
    y = df[target_variable]
    y = y.values.ravel()
    X = df[features]

    return X, y


def get_predictions(model, X) -> np.ndarray:
    """get predictions from the dataset and model as input"""
    return model.predict(X)
