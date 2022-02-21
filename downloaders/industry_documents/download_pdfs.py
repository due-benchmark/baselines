import os
import gzip

from tqdm import tqdm
from boto3 import client
from botocore import UNSIGNED
from botocore.client import Config

BUCKET_NAME = 'edu.ucsf.industrydocuments.artifacts'
TARGET_PATH = 'pdf'


client = client('s3', config=Config(signature_version=UNSIGNED))

if not os.path.exists(TARGET_PATH):
    os.makedirs(TARGET_PATH)

with gzip.open('file_ids.txt.gz', 'rt') as ids:
    for l in tqdm(ids):
        l = l.rstrip()
        target = f'{TARGET_PATH}/{l}.pdf'
        key = f'{l[0]}/{l[1]}/{l[2]}/{l[3]}/{l}/{l}.pdf'
        if not os.path.exists(target):
            with open(target, 'wb') as f:
                client.download_fileobj(BUCKET_NAME, key, f)
