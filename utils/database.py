from .configs import *
import pymongo as mg
from joblib import Parallel, delayed

__all__ = ['aggregate', 'ordered_lookup', 'get_top_vhosts', 'get_guids_by_vhost', 'get_guids_by_top_vendor', 'get_labels']

def get_collection(train):
    return TRAIN_COLLECTION if train else TEST_COLLECTION

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

def lookup_by_chunks(chunk_size=0, n_jobs=8):
    def func_wrapper(func):
        def func_by_chunks(*args, **kwargs):
            if chunk_size <= 0 or chunk_size > len(args):
                return func(*args, **kwargs)
            else:
                num_chunks = len(args)//chunk_size + int(len(args)%chunk_size>0)
                res = Parallel(n_jobs=n_jobs, backend='threading', verbose=False)\
                        (delayed(func)(*args[i*chunk_size:(i+1)*chunk_size], **kwargs) for i in range(num_chunks))
                return sum(res, [])
        return func_by_chunks
    return func_wrapper

@with_temp_client
def aggregate(*pips, client=None, db=DB_NAME, train=True):
    collection = client[db][get_collection(train)]
    return list(collection.aggregate(list(pips), allowDiskUse=True))

def ordered_lookup(field_name, *field_values, client=None, db=DB_NAME, train=True, project_fields=None, flatten=False):
    if project_fields is None:
        cursor =[{field_name:v} for v in field_values]
        project_fields = [field_name]
    else:
        pipeline = [{'$match':{field_name:{'$in':field_values}}},
                {'$addFields':{'__order':{'$indexOfArray':[field_values, '${}'.format(field_name)]}}},
                {'$sort':{'__order':1}}]
        projection = {field:1 for field in project_fields}
        projection['_id'] = 0
        pipeline.append({'$project':projection})
        cursor = aggregate(*pipeline, client=client, db=db, train=train)
    if flatten:
        results = list(map(lambda d:tuple(d[field] for field in project_fields) if len(project_fields)>1 else d[project_fields[0]], cursor))
    else:
        results = list(cursor)
    return results

def get_top_vhosts(top_n=None, client=None, db=DB_NAME, train=True):
    pipeline = [{'$group':{'_id':'$vhost', 'count':{'$sum':1}}},
            {'$sort':{'count':-1}}]
    if top_n is not None:
        pipeline.append({'$limit': int(top_n)})
    results = [r['_id'] for r in aggregate(*pipeline, client=client, db=db, train=train)]
    return results

def get_guids_by_vhost(*vhosts, least=0, client=None, db=DB_NAME, train=True):
    match = {'$match': {'vhost':{'$in':vhosts}, 'num_properties':{'$lt':2}}}
    group = {'$group': {'_id':{'vhost':'$vhost', 'vendor':'$vendor_id'}, 'count':{'$sum':1}, 'guids':{'$push': '$attachment_guid'}}}
    limit = {'$match': {'count': {'$gt':least}}}
    proj = {'$project': {'_id':0, 'guids':1}}
    pipeline = [match, group, limit, proj]
    guids = []
    for r in aggregate(*pipeline, client=client, db=db, train=train):
        guids.extend(r['guids'])
    return guids

def get_guids_by_top_vendor(limit=100, client=None, db=DB_NAME, train=True):
    match = {'$match': {'num_properties':{'$lt':2}}}
    group = {'$group': {'_id':{'vhost':'$vhost', 'vendor':'$vendor_id'}, 'count':{'$sum':1}, 'guids':{'$push': '$attachment_guid'}}}
    limit = {'$limit': limit}
    sort = {'$sort': {'count': -1}}
    proj = {'$project': {'_id':0, 'guids':1}}
    pipeline = [match, group, sort, limit, proj]
    guids = []
    for r in aggregate(*pipeline, client=client, db=db, train=train):
        guids.extend(r['guids'])
    return guids

@lookup_by_chunks(chunk_size=100000)
def get_labels(*guids, client=None, account=False, vendor=False, prop=False, total=False, db=DB_NAME, train=True, flatten=False):
    project_fields = [ACCOUNT_FIELD_NAME] if account else [VHOST_FIELD_NAME]
    if vendor:
        project_fields.append(VENDOR_FIELD_NAME)
    if prop:
        project_fields.append(PROPERTY_FIELD_NAME)
    if total:
        project_fields.append(TOTAL_FIELD_NAME)
    results = ordered_lookup(GUID_FIELD_NAME, *guids, client=client, db=db, train=train,
            project_fields=project_fields, flatten=flatten)
    return results
