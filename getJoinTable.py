def joinTable():
	import import_datasets as imp
	import pandas as pd
	import numpy as np

	datasets=imp.importData()

	notCat=['id','volume']
	cat_cols=['location','log_feature','severity_type','resource_type','event_type']
	samples=['train','test']

	join=pd.DataFrame({'id':[]})
	datasets['train']['sample']='train'
	datasets['test']['sample']='test'
	datasets['test']['fault_severity']=np.nan
	join=pd.concat([datasets['train'],datasets['test']],ignore_index=True)

	datasets['feature']['volume']=pd.cut(datasets['feature']['volume'],bins=[0,1,2,7,1310],\
										 labels=[1,2,3,4]).astype(str)
	for key, dataset in datasets.items():
		if key not in samples:
			join=pd.merge(join,dataset,on='id',how='inner')
	
	return join