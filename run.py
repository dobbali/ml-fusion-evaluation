from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from fusion_evaluation.common import utils
from fusion_evaluation.metrics import classification, regression
from fusion_evaluation.common import sf

if __name__ == "__main__":
    engine = create_engine(URL(**sf.fetch_snowflake_config()))
    # Basket Prediction
    bp_eval_df = utils.get_evaluation_dataset(model_name='bp', engine=engine)
    model, features = utils.get_model_and_features("/home/ec2-user/SageMaker/external_components/basket_model")
    X, y = utils.get_clean_XY(bp_eval_df, features, 'order_total')
    y_pred = utils.get_predictions(model, X)
    re = regression.RegressionEval(y_pred=y_pred, y_true=y, num_features=len(features))
    eval_results_bp = regression.metrics(re)
    print(eval_results_bp)
    # Conversion Prediction
    cvr_eval_df = utils.get_evaluation_dataset(model_name='cvr', engine=engine)
    model, features = utils.get_model_and_features("/home/ec2-user/SageMaker/external_components/cvr_model")
    X, y = utils.get_clean_XY(cvr_eval_df, features, 'label')
    y_pred = utils.get_predictions(model, X)
    ce = classification.ClassificationEval(y_pred=y_pred, y_true=y)
    eval_results_ce = classification.metrics(ce)
    print(eval_results_ce.accuracy)
    print(eval_results_ce.auc_roc)
    print(eval_results_ce.precision)
