from __future__ import division
import gensim
import numpy as np
import ast

# ifile = open('Train/ADV_file_Restaurant_Train_multiWordAspect.txt')
ifile = open('Test/ADV_file_Restaurant_Test.txt')
doc = ifile.read()
ifile.close()


#Read model
model = gensim.models.Word2Vec.load_word2vec_format('../raw_data/GoogleNews-vectors-negative300.bin',binary=True)
polarities = {'positive':0,'negative':1,'neutral':2,'conflict':3}

lines = doc.split('\n')
lines = lines[1:-1]
num_asps = len(lines)
output1 = np.zeros((num_asps,300))
output2 = np.zeros(num_asps)

for idx,line in enumerate(lines):
	print idx, line
	line = line.split('\t')
	# Now, line = [SenID,SenIdx,AspIdx,Aspect,ADVList,Polarity]
	
	pol = polarities[line[5]]

	# #Preprocess the ADVList to get all ADV words list
	# adv_list = line[4].split('\'') 
	# num_advs = int((len(adv_list)-1)/2)	#checked for 0
	# for i in range(num_advs-1):
	# 	adv_list.remove(', ')
	# adv_list = adv_list[1:-1] #remove the braces from both ends
	# # Now we have the full adv_list as a list
	adv_list = ast.literal_eval(line[4])

	if 'ROOT' in adv_list:
		adv_list.remove('ROOT')
		# num_advs -= 1

	adv = np.zeros(300)
	N = 0
	#add each word to adv and take average
	for word in adv_list:
		if word in model:
			adv += model[word]
			N += 1
	if not N == 0:
		adv /= N
	output1[idx] = adv
	output2[idx] = pol

# np.savetxt('Train/x_train.csv',output1,delimiter=',')
# np.savetxt('Train/y_train.csv',output2,delimiter=',')

np.savetxt('Test/x_test.csv',output1,delimiter=',')
np.savetxt('Test/y_test.csv',output2,delimiter=',')

#################################################
# To read:
