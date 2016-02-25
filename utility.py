import os
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import preprocessing as pre

def draw_ct(join, colName):
    df = pd.crosstab(join[colName], join['fault_severity'])
    ndf = df.div(df.sum(1).astype(float), axis=0)

    # Normalize the cross tab to sum to 1:
    ndf = ndf.sort(columns=[2.0, 1.0, 0.0])
    return ndf, ndf.plot(kind='bar', stacked=True, title=('Fault Severity by ' + colName))


def empty_join(datasets):
    join = pd.DataFrame({'id': []})
    datasets['train']['sample'] = 'train'
    datasets['test']['sample'] = 'test'
    datasets['test']['fault_severity'] = np.nan
    join = pd.concat([datasets['train'], datasets['test']], ignore_index=True)
    return join


def cat_join():
    datasets = get_datasets()

    notCat = ['id', 'volume']
    cat_cols = get_cat_cols()
    samples = get_sample_names()

    join = pd.DataFrame({'id': []})
    datasets['train']['sample'] = 'train'
    datasets['test']['sample'] = 'test'
    datasets['test']['fault_severity'] = np.nan
    join = pd.concat([datasets['train'], datasets['test']], ignore_index=True)

    datasets['log_feature']['volume'] = pd.cut(datasets['log_feature']['volume'], bins=[0, 1, 2, 7, 1310], labels=[1, 2, 3, 4]).astype(str)
    for key, dataset in datasets.items():
        if key not in samples:
            join = pd.merge(join, dataset, on='id', how='inner')

    return join

def get_cat_cols():
    return ['location', 'log_feature', 'severity_type', 'resource_type', 'event_type']
def get_sample_names():
    return ['train', 'test']

def get_datasets():
    path = os.getcwd() + '/data/'
    #TODO: make a look from this with imput as file names
    trainFile = "train.csv"
    testFile = "test.csv"
    resourceFile = "resource_type.csv"
    eventTypeFile = "event_type.csv"
    logFeatureFile = "log_feature.csv"
    severityTypeFile = "severity_type.csv"

    test = pd.read_csv(filepath_or_buffer=path + testFile, delimiter=",", header=0)
    train = pd.read_csv(filepath_or_buffer=path + trainFile, delimiter=",", header=0)
    resource = pd.read_csv(filepath_or_buffer=path + resourceFile, delimiter=",", header=0)
    event = pd.read_csv(filepath_or_buffer=path + eventTypeFile, delimiter=",", header=0)
    feature = pd.read_csv(filepath_or_buffer=path + logFeatureFile, delimiter=",", header=0)
    severity = pd.read_csv(filepath_or_buffer=path + severityTypeFile, delimiter=",", header=0)

    datasets = {'train': train, 'test': test, 'resource_type': resource, 'event_type': event, 'log_feature': feature,
                'severity_type': severity}
    return datasets

def get_prior_features(join):
    cat_cols=get_cat_cols()
    ds=get_datasets()

    prior_col_names=['rt_f1','rt_f2','et_f1','et_f2','lf_f1','lf_f2','lf_f3','lf_f4','volume_y']
    for cat in cat_cols:
        if cat is not 'location':
            df=pd.merge(ds['train'],ds[cat],on='id')
            del df['id']
            df.rename(columns={'fault_severity':(cat+'_prior')}, inplace=True)
            prior_col_names.append(cat+'_prior')
            dfg=df.groupby(['location',cat]).mean()
            dfg.reset_index(inplace=True)
            join=pd.merge(join,dfg,on=['location',cat],how='left')


    def rt_f1(x):
        if x=='resource_type_5':
            return 0
        else:
            return 1

    def rt_f2(x):
        if x=='resource_type_2':
            return 0
        else:
            return 1

    join['rt_f1']=join['resource_type'].map(rt_f1)
    join['rt_f2']=join['resource_type'].map(rt_f2)

    et_l1=['event_type_7','event_type_27','event_type_50','event_type_15','event_type_8','event_type_47','event_type_46','event_type_9'
          ,'event_type_5','event_type_38','event_type_3','event_type_49','event_type_39','event_type_53','event_type_19']
    et_l2=['event_type_30','event_type_29','event_type_21','event_type_28','event_type_2','event_type_31']

    def et_f1(x):
        if x in et_l1:
            return 1
        else:
            return 0

    def et_f2(x):
        if x in et_l2:
            return 1
        else:
            return 0

    join['et_f1']=join['event_type'].map(et_f1)
    join['et_f2']=join['event_type'].map(et_f2)

    df=pd.crosstab(join['log_feature'],join['fault_severity'])
    ndf = df.div(df.sum(1).astype(float), axis=0)

    i1=ndf[ndf[0.0]==1.0].index
    i21=ndf[ndf[2.0]==0.0].index
    i22=ndf[ndf[1.0]>=0.5].index
    i3=ndf[ndf[2.0]>=0.5].index
    i41=ndf[ndf[2.0]>0.0].index
    i42=ndf[ndf[2.0]<0.5].index

    def lf_f1(x):
        if x in i1:
            return 1
        else:
            return 0

    def lf_f2(x):
        if (x in i21) & (x in i22):
            return 1
        else:
            return 0

    def lf_f3(x):
        if x in i3:
            return 1
        else:
            return 0

    def lf_f4(x):
        if (x in i41) & (x in i42):
            return 1
        else:
            return 0

    join['lf_f1']=join['log_feature'].map(lf_f1)
    join['lf_f2']=join['log_feature'].map(lf_f2)
    join['lf_f3']=join['log_feature'].map(lf_f3)
    join['lf_f4']=join['log_feature'].map(lf_f4)


    prior_fs=join.loc[:,prior_col_names]
    prior_fs['id']=join['id']
    prior_fs=prior_fs.groupby('id').mean()
    prior_fs=prior_fs.fillna(value=0)
    prior_fs['id']=prior_fs.index
    for column in prior_fs.columns:
        if column is not 'id':
            prior_fs[column]=pre.scale(prior_fs[column])

    return prior_fs

def get_stats_features(join):
    from scipy.stats import chisquare
    dict_stats={}
    cat_cols=get_cat_cols()
    stat_col_names=[]

    for column in cat_cols:
        df=pd.crosstab(join[column],join['fault_severity'])
        df[column+'_sum']=df.sum(axis=1)
        stat_col_names.append(column+'_sum')
        df.columns=['0','1','2',column+'_sum']

        df[column+'_std']=df[['0','1','2']].std(axis=1)
        stat_col_names.append(column+'_std')
        f,_=chisquare(df[['0','1','2']],axis=1)
        df[column+'_pchisqr']=f
        stat_col_names.append(column+'_pchisqr')

        fs=[column+'_sum',column+'_std',column+'_pchisqr']
        df=df[fs]
        df[column]=df.index
        dict_stats[column]=df
        join=pd.merge(join,df,on=column,how="left") #safer than inner

    stats=join.loc[:,stat_col_names]
    stats['id']=join['id']
    stats=stats.groupby('id').mean()
    stats=stats.fillna(value=0)
    stats['id']=stats.index
    for column in stats.columns:
        if column is not 'id':
            stats[column]=pre.scale(stats[column])

    return stats

