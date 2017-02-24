from __future__ import division
import gensim
import numpy as np
import ast
import pickle

ifile = open('Train/ADV_file_Restaurant_Train_multiWordAspect.txt')
# ifile = open('Test/ADV_file_Restaurant_Test.txt')
doc = ifile.read()
ifile.close()


#Read model
model = gensim.models.Word2Vec.load_word2vec_format('../raw_data/GoogleNews-vectors-negative300.bin',binary=True)
polarities = {'positive':0,'negative':1,'neutral':2,'conflict':3}

# FOR RESTAURANTS ----------------------------------------------------------------------------------------__EDIT ACCORDINGLY__------------------------------
categories = ['excellent','poor'] # to maintain order, we do not access directly from dictionary
# Identify 20 seeds per category

# seeds = {}
# for cat in categories:
# 	print "Category: ",cat
# 	cat_seeds = model.similar_by_word(cat, topn=20)
# 	seeds[cat] = [cat_seed[0] for cat_seed in cat_seeds]

# with open('restaurants_pns_seeds.pickle','wb') as handle:
# 	pickle.dump(seeds,handle)

# to read
# with open('restaurants_pns_seeds.pickle','rb') as handle:
	# seeds = pickle.load(handle)
seeds = {}
seeds['excellent'] = ['delicious','impeccable','exceptional','exquisite','outstanding','spot-on','phenomenal','terrific','fabulous','fantastic','incredible','awesome','good','wonderful','amazing','great','delightful','delightful','delightful','delightful']
seeds['poor'] = ['pathetic','deplorable','appalling','horrific','horrendous','horrible','horrid','abysmal','atrocious','awful','terrible','lousy','crappy','bad','substandard','shitty','subpar','subpar','subpar','subpar']

lines = doc.split('\n')
lines = lines[1:-1]
num_asps = len(lines)
output1 = np.zeros((num_asps,303))
output2 = np.zeros(num_asps)

for idx,line in enumerate(lines):
	# print idx, line
	
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
	sims = np.zeros((len(adv_list),len(categories),20))
	#add each word to adv and take average
	for idx_word,word in enumerate(adv_list):
		if word in model:
			adv += model[word]
			N += 1
			for idx_cat,cat in enumerate(categories):
				for idx_seed,seed in enumerate(seeds[cat]):
					if seed in model:
						sims[idx_word,idx_cat,idx_seed] = model.similarity(word,seed)
	
	if len(adv_list) is 0:
		pns = [0,0]
	else:
		pns = np.max(np.max(sims,axis=0),axis=1) # shape should be len(categories) = 2
	
	if not N == 0:
		adv /= N

	output1[idx] = np.append(adv,np.append(pns,[pns[1]-pns[0]]))
	output2[idx] = pol

np.savetxt('Train_pns/x_train.csv',output1,delimiter=',')
np.savetxt('Train_pns/y_train.csv',output2,delimiter=',')

# np.savetxt('Test_pns/x_test.csv',output1,delimiter=',')
# np.savetxt('Test_pns/y_test.csv',output2,delimiter=',')

#################################################
# To read:
# np.loadtxt('filename.csv',delimiter=',')

############## TEST  ##################################################################################

ifile = open('Test/ADV_file_Restaurant_Test.txt')
doc = ifile.read()
ifile.close()

lines = doc.split('\n')
lines = lines[1:-1]
num_asps = len(lines)
output1 = np.zeros((num_asps,303))
output2 = np.zeros(num_asps)

for idx,line in enumerate(lines):
	# print idx, line
	
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
	sims = np.zeros((len(adv_list),len(categories),20))
	#add each word to adv and take average
	for idx_word,word in enumerate(adv_list):
		if word in model:
			adv += model[word]
			N += 1
			for idx_cat,cat in enumerate(categories):
				for idx_seed,seed in enumerate(seeds[cat]):
					if seed in model:
						sims[idx_word,idx_cat,idx_seed] = model.similarity(word,seed)
	if len(adv_list) is 0:
		pns = [0,0]
	else:
		pns = np.max(np.max(sims,axis=0),axis=1) # shape should be len(categories) = 2
	
	if not N == 0:
		adv /= N

	output1[idx] = np.append(adv,np.append(pns,[pns[1]-pns[0]]))
	output2[idx] = pol

np.savetxt('Test_pns/x_test.csv',output1,delimiter=',')
np.savetxt('Test_pns/y_test.csv',output2,delimiter=',')
