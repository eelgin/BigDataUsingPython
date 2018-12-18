import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

directory = 'stats'
files = os.listdir(directory)

output_file = 'compiled_data.csv'

stats_data = pd.DataFrame(columns=['epoch','batch'])

for file in files:

	if file == output_file:
		continue

	tmp = pd.read_csv(directory+'/'+file, index_col=0)

	name = file.rpartition('.')[0]

	print ('Loading '+name)

	tmp[name+'_train'] = tmp['train']
	tmp[name+'_test'] = tmp['test']
	tmp = tmp.drop(columns=['train','test'])

	stats_data = pd.merge(stats_data, tmp, how='outer', on=['epoch','batch'])

	del tmp

stats_data.to_csv('stats/'+output_file)

stats_data = stats_data.drop(['epoch','batch'],axis=1)


plt.title('NNs: Batch Size 128')
plt.plot(stats_data['nn_2L_relu_b128_train'],'b--',label='2L ReLu Train')
plt.plot(stats_data['nn_2L_relu_b128_test'],'b-',label='2L ReLu Test')
plt.plot(stats_data['nn_3L_relu_b128_train'],'g--',label='3L ReLu Train')
plt.plot(stats_data['nn_3L_relu_b128_test'],'g-',label='3L ReLu Test')
plt.plot(stats_data['nn_4L_relu_b128_train'],'r--',label='4L ReLu Train')
plt.plot(stats_data['nn_4L_relu_b128_test'],'r-',label='4L ReLu Test')
plt.plot(stats_data['nn_5L_relu_b128_train'],'c--',label='5L ReLu B128 Train')
plt.plot(stats_data['nn_5L_relu_b128_test'],'c-',label='5L ReLu B128 Test')
plt.plot(stats_data['nn_2L_softplus_b128_train'],'m--',label='2L SoftPlus B128 Train')
plt.plot(stats_data['nn_2L_softplus_b128_test'],'m-',label='2L SoftPlsu B128 Test')
plt.plot(stats_data['nn_3L_softplus_b128_train'],'y--',label='3L SoftPlus B128 Train')
plt.plot(stats_data['nn_3L_softplus_b128_test'],'y-',label='3L SoftPlsu B128 Test')
plt.plot(stats_data['nn_4L_softplus_b128_train'],'k--',label='4L SoftPlus B128 Train')
plt.plot(stats_data['nn_4L_softplus_b128_test'],'k-',label='4L SoftPlsu B128 Test')

plt.legend(loc='best')
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

#####

plt.title('NNs: Inconsistent Training')
plt.plot(stats_data['nn_3L_relu_b128_train'],'g--',label='3L ReLu Train')
plt.plot(stats_data['nn_4L_relu_b128_train'],'r--',label='4L ReLu Train')
plt.plot(stats_data['nn_5L_relu_b128_train'],'c--',label='5L ReLu Train')
plt.plot(stats_data['nn_2L_relu_b128_train'],'b--',label='2L ReLu Train')


plt.legend(loc='best')
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

####

plt.title('NNs: Consistent Training')
plt.plot(stats_data['nn_2L_softplus_b128_train'],'m--',label='2L SoftPlus B128 Train')
plt.plot(stats_data['nn_3L_softplus_b128_train'],'y--',label='3L SoftPlus B128 Train')
plt.plot(stats_data['nn_4L_softplus_b128_train'],'k--',label='4L SoftPlus B128 Train')

plt.legend(loc='best')
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

###

minimum = stats_data.min(axis=0)
minimum = minimum.sort_index()
minimum_train = minimum.drop(['nn_2L_softplus_b128_test',
							  'nn_2L_relu_b128_test',
							  'nn_3L_relu_b128_test',
							  'nn_3L_softplus_b128_test',
							  'nn_4L_relu_b128_test',
							  'nn_4L_softplus_b128_test',
							  'nn_4L_relu_b256_test',
							  'nn_5L_relu_b128_test',
							  'nn_5L_relu_b256_test'])

train_list = minimum_train.index.tolist()
for indx in range(0,len(train_list)):
	#indx = indx.index
	train_list[indx] = train_list[indx].replace('_', ' ')
	train_list[indx] = train_list[indx].replace('nn ',' ')
	train_list[indx] = train_list[indx].replace(' train','')

minimum_train.index = train_list
ind = np.arange(0,minimum_train.shape[0])
width = 0.35

plt.figure(1)
plt.subplot(1,2,1)
plt.title('NN MSE Training Minimum')
plt.bar(ind, minimum_train, width,color='orange')

ax = plt.gca()
ax.set_yscale('log')
plt.xticks(ind, train_list, rotation=30,ha='right')
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#plt.show(block=True)

####

minimum_test  = minimum.drop(['nn_2L_softplus_b128_train',
							  'nn_2L_relu_b128_train',
							  'nn_3L_relu_b128_train',
							  'nn_3L_softplus_b128_train',
							  'nn_4L_relu_b128_train',
							  'nn_4L_softplus_b128_train',
							  'nn_4L_relu_b256_train',
							  'nn_5L_relu_b128_train',
							  'nn_5L_relu_b256_train'])

test_list = minimum_test.index.tolist()
for indx in range(0,len(train_list)):
	#indx = indx.index
	test_list[indx] = test_list[indx].replace('_', ' ')
	test_list[indx] = test_list[indx].replace('nn ',' ')
	test_list[indx] = test_list[indx].replace(' test','')

minimum_test.index = test_list

ind = np.arange(0,minimum_test.shape[0])
width = 0.35

plt.subplot(1,2,2)
plt.title('NN MSE Testing Minimum')
plt.bar(ind, minimum_test, width)

ax = plt.gca()
ax.set_yscale('log')
plt.xticks(ind, test_list, rotation=30,ha='right')
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

"""
plt.plot(stats_data)
plt.legend(list(stats_data))
plt.ylim(0,1)
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

plt.plot(stats_data)
plt.legend(list(stats_data))
plt.ylim(0,.1)
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

plt.plot(stats_data)
plt.legend(list(stats_data))
plt.ylim(0,.01)
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)

plt.plot(stats_data)
plt.legend(list(stats_data))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('Epoch',fontsize=18)
plt.xticks(range(0,stats_data.shape[0],28),(i for i in range(0,10)))
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=True)
"""