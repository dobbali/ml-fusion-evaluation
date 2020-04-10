# # Save model to s3 bucket
# from sagemaker.session import Session
# from sagemaker import get_execution_role
# import boto3
#
# bucket_name = "sagemaker-fusion-evaluation"
# role = get_execution_role()
#
# session = boto3.Session()
# s3_resource = session.resource('s3')
#
# s3_resource.Bucket(bucket_name).upload_file(Filename='data/basket_eval_final.csv', Key='basket_pred_model.csv')
# s3_resource.Bucket(bucket).upload_file(Filename='basket_model',Key='bm')
#
