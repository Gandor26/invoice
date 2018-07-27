from joblib import Parallel, delayed
from tqdm import tqdm
import pymongo as mg
import numpy as np
import threading
import logging
import shutil
import boto3
import json
import os
from utils import get_dir, get_logger

DB_NAME = 'invoice'
GUID_FIELD_NAME = 'attachment_guid'
VHOST_FIELD_NAME = 'vhost'
ACCOUNT_FIELD_NAME = 'account_name'
VENDOR_FIELD_NAME = 'vendor_id'
PROPERTY_FIELD_NAME = 'property_id'
TOTAL_FIELD_NAME = 'total'

def download_and_convert(*guids, n_jobs=-1, logger=get_logger(), training=True, image_format='png'):
    thread_storage = threading.local()
    logger.info('Downloading {} invoices in pdfs'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=int(logger.getEffectiveLevel() in [logging.DEBUG, logging.INFO]))\
            (delayed(_download_and_convert)(guid, account, training, thread_storage, image_format)\
            for guid, account in tqdm(zip(guids, ordered_lookup('attachment_guid', *guids, training=training, project_fields=['account_name'], flatten=True))))

def _download_and_convert(guid, account, training, thread_storage, image_format):
    if getattr(thread_storage, 's3_client', None) is None:
        thread_storage.s3client = boto3.client('s3', 'us-east-1')
    bucket_name = 'appfolio-ml-invoice-{}-set'.format('training' if training else 'testing')
    fname_prefix = 'attachmentsParallelized/{}/attachments/{}/original'.format(account, guid)
    objects = thread_storage.s3client.list_objects(Bucket=bucket_name, Prefix=fname_prefix).get('Contents', None)
    if objects is None:
        raise Exception('Didn\'t find {} in {}'.format(guid, account))
    elif len(objects) > 1:
        raise Exception('Expected one key to match prefix {}, but found {}'.format(fname_prefix, len(objects)))
    else:
        key = objects[0]['Key']
    pdf_path = os.path.join(get_dir('./data/pdf/{}'.format('training' if training else 'test')), '{}.pdf'.format(guid))
    img_path = os.path.join(get_dir('./data/img/{}'.format('training' if training else 'test')), '{}.png'.format(guid))
    thread_storage.s3client.download_file(bucket_name, key,
            os.path.join(get_dir('./data/pdf/{}'.format('training' if training else 'test')), '{}.pdf'.format(guid)))
    gs_device = '{}gray'.format(image_format)
    command = 'gs -q -dNOPAUSE -sDEVICE={} -r300 -dINTERPOLATE -dFirstPage=1 -dLastPage=1 -dGraphicsAlphaBits=4 -sOutputFile={} -c 30000000 setvmthreshold -f {} -c quit'
    os.system(command.format(gs_device, img_path, pdf_path))

def with_temp_client(func):
    def func_with_temp_client(*args, client=None, **kwargs):
        if client is None:
            try:
                client = mg.MongoClient('localhost', 27017)
                return func(*args, client=client, **kwargs)
            finally:
                client.close()
        else:
            return func(*args, client=client, **kwargs)
    return func_with_temp_client

@with_temp_client
def ordered_lookup(field_name, *field_values, client=None, db=DB_NAME, training=True, project_fields=None, flatten=False):
    pipeline = [{'$match':{field_name:{'$in':field_values}}},
            {'$addFields':{'__order':{'$indexOfArray':[field_values, '${}'.format(field_name)]}}},
            {'$sort':{'__order':1}}]
    if project_fields is None:
        projection = {'attachment_guid':1}
    else:
        projection = {'_id': 0}
        for field in project_fields:
            projection[field] = 1
    pipeline.append({'$project':projection})
    results = client[db]['{}_set'.format('training' if training else 'test')].aggregate(pipeline, allowDiskUse=True)
    if flatten:
        results = list(map(lambda d:tuple(d[field] for field in project_fields) if len(project_fields)>1 else d[project_fields[0]], results))
    else:
        results = list(results)
    return results

@with_temp_client
def get_frequent_vhosts(top_n=None, client=None, db=DB_NAME, training=True):
    pipeline = [{'$group':{'_id':'$vhost', 'count':{'$sum':1}}},
            {'$sort':{'count':-1}}]
    if top_n is not None:
        pipeline.append({'$limit': int(top_n)})
    results = [r['_id'] for r in client[db]['{}_set'.format('training' if training else 'test')].aggregate(pipeline, allowDiskUse=True)]
    return results

@with_temp_client
def get_guids_by_vhost(*vhosts, limit=100, client=None, db=DB_NAME, training=True):
    match = {'$match': {'vhost':{'$in':vhosts}}}
    group = {'$group': {'_id':{'vhost':'$vhost', 'vendor':'$vendor_id'}, 'count':{'$sum':1}, 'guids':{'$push': '$attachment_guid'}}}
    guids = {'$group': {'_id':'$_id.vhost', 'vendors': {'$push': {'vid': '$_id.vendor', 'guids': '$guids'}}, 'count':{'$sum': '$count'}}}
    limit = {'$match': {'count': {'$gt':limit}}}
    sort = {'$sort': {'count': -1}}
    proj = {'$project': {'_id':1, 'list':1}}
    pipeline = [match, group, guids, limit, sort, proj]
    guids = []
    labels = {}
    for r in client[db]['{}_set'.format('training' if training else 'test')].aggregate(pipeline, allowDiskUse=True):
        guids.extend(r['vendors']['guids'])
        for guid in r['vendors']['guids']:
            labels[guid] = '{}_{}'.format(r['_id'], r['vendors']['vid'])
    return guids, labels

@with_temp_client
def get_guids_by_top_vendor(limit=100, client=None, db=DB_NAME, training=True):
    group = {'$group': {'_id':{'vhost':'$vhost', 'vendor':'$vendor_id'}, 'count':{'$sum':1}, 'guids':{'$push': '$attachment_guid'}}}
    limit = {'$limit': limit}
    sort = {'$sort': {'count': -1}}
    proj = {'$project': {'_id':1, 'guids':1}}
    pipeline = [group, sort, limit, proj]
    guids = []
    labels = {}
    for r in client[db]['{}_set'.format('training' if training else 'test')].aggregate(pipeline, allowDiskUse=True):
        guids.extend(r['guids'][:1000])
        for guid in r['guids'][:1000]:
            labels[guid] = '{}_{}'.format(r['_id']['vhost'], r['_id']['vendor'])
    return guids, labels

@with_temp_client
def get_labels(*guids, client=None, vendor=False, prop=False, total=False, db=DB_NAME, training=True):
    if not (vendor or prop or total):
        raise ValueError('At least return one attribute as label, but none is enabled')
    project_fields = []
    if vendor:
        project_fields.append(VENDOR_FIELD_NAME)
    if prop:
        project_fields.append(PROPERTY_FIELD_NAME)
    if total:
        project_fields.append(TOTAL_FIELD_NAME)
    results = ordered_lookup(GUID_FIELD_NAME, *guids, client=client, db=db, training=training,
            project_fields=project_fields)
    return results

if __name__ == '__main__':
    guids, labels = get_guids_by_top_vendor(limit=10)
    download_and_convert(*guids, n_jobs=4)
    with open(os.path.join(get_dir('./data/training'), 'labels.json'), 'w') as f:
        json.dump(labels, f)
    for guid in label:
        cls = label[guid]
        src_path = './data/img/training/{}.png'.format(guid)
        dst_path = os.path.join(get_dir('./data/img/training/{}'.format(cls)), '{}.png'.format(guid))
        shutil.move(src_path, dst_path)

