import os, logging, json, time, urllib.parse
import boto3, botocore
import numpy as np, cv2

logger = logging.getLogger()
logger.setLevel(logging.INFO)
client = boto3.client('lambda')

# S3 BUCKETS DETAILS
s3 = boto3.resource('s3')
BUCKET_NAME = "sage-made"
IMAGE_LOCATION = "img/path"

# INFERENCE ENDPOINT DETAILS
ENDPOINT_NAME = 'yolotensorprod'
config = botocore.config.Config(read_timeout=500)
runtime = boto3.client('runtime.sagemaker', config=config)
modelHeight, modelWidth = 640, 640

# RUNNING LAMBDA
def lambda_handler(event, context):

    body = event['body']
    s3url = body['key']
    bucket = body['bucket']

    #key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    # INPUTS - Download Image file from S3 to Lambda /tmp/
    #input_imagename = key.split('/')[-1]rl
    logger.info(f'Input Imagename: {s3url}')
    s3.Bucket(bucket).download_file(s3url, '/tmp/' + s3url)

    # INFERENCE - Invoke the SageMaker Inference Endpoint
    logger.info(f'Starting Inference ... ')
    orig_image = cv2.imread('/tmp/' + s3url)
    if orig_image is not None:
        start_time_iter = time.time()
        # pre-processing input image
        image = cv2.resize(orig_image.copy(), (modelWidth, modelHeight), interpolation = cv2.INTER_AREA)
        data = np.array(image.astype(np.float32)/255.)
        payload = json.dumps([data.tolist()])
        # run inference
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=payload)
        # get the output results
        result = json.loads(response['Body'].read().decode())
        end_time_iter = time.time()
        # get the total time taken for inference
        inference_time = round((end_time_iter - start_time_iter)*100)/100
    logger.info(f'Inference Completed ... ')

    # OUTPUTS - Using the output to utilize in other services downstream
    return {
        "statusCode": 200,
        "body": json.dumps({
                "message": "Inference Time:// " + str(inference_time) + " seconds.",
                "results": result
                }),
            }

# local testing
#e = {
#    "body": {
#    "key": "msm854_fol. 47(v,r) expell_fol1L.jpg",
#    "bucket": "msm854"
#    }
#}

#i = lambda_handler(event = e, context = None)
#print(i, 'i')
