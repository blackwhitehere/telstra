import os
import pandas as pd

def importData():
	path=os.getcwd()+'/data/'

	trainFile="train.csv"
	testFile="test.csv"
	resourceFile="resource_type.csv"
	eventTypeFile="event_type.csv"
	logFeatureFile="log_feature.csv"
	severityTypeFile="severity_type.csv"

	test=pd.read_csv(filepath_or_buffer=path+testFile,delimiter=",",header=0)
	train=pd.read_csv(filepath_or_buffer=path+trainFile,delimiter=",",header=0)
	resource=pd.read_csv(filepath_or_buffer=path+resourceFile,delimiter=",",header=0)
	event=pd.read_csv(filepath_or_buffer=path+eventTypeFile,delimiter=",",header=0)
	feature=pd.read_csv(filepath_or_buffer=path+logFeatureFile,delimiter=",",header=0)
	severity=pd.read_csv(filepath_or_buffer=path+severityTypeFile,delimiter=",",header=0)

	datasets={'train':train,'test':test,'resource':resource,'event':event,'feature':feature,'severity':severity}
	return datasets
