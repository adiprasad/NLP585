import bs4
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
import numpy as np 
import re 
import string

def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

# Importing the XML file to xml_doc
xml_file = open("../../Datasets/SemEval2014/Extracted/Restaurants/train/Restaurants_Train.xml",'r')
xml_doc = xml_file.read()
xml_file.close()
#print xml_doc

cat_idx_map = {'food':0,'service':1,'ambience':2,'price':3,'anecdotes/miscellaneous':4}
soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

num_sents = len(sentences)
dict_of_sets = {k : set() for k in range(len(sentences))}

sent_idx_array = []

words_set_file = "words_set_file_train.dict"
#sent_idx_file = "sent_ids_test_gold.npy"
idx = 0

for senten in sentences:
	
	senten_id = senten['id']				# Finding the sentence id 
	sent_idx_array.append(senten_id)   		# Array ... index = row number in tfidf 2d matrix, output = sentence id
	
	# Extracting the pos tags out of reviews
	text =  senten.find('text')				# Text tag (tag that contains the review)
	#text = text.string
	#text = text.translate(None,string.punctuation)   # Removing the punctuation words
	tokens = word_tokenize(text.string)
	tokens = filter(lambda x : re.match('\w+',x),tokens)
	
	dict_of_sets[idx] = set(tokens)    # Distinct words set for sentence index = 0

	idx+=1

#np.savetxt(words_set_file,dict_of_sets,delimiter=',')
#np.savetxt(sent_idx_file,sent_idx_array,delimiter=',')

xml_file = open("../../Test_Data_Gold/ABSA_Gold_TestData/Restaurants_Test_Gold.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

num_sents = len(sentences)
cat_mat_test = np.zeros((num_sents,5))
cat_mat_train = np.loadtxt("tfidf_category_train.npy",delimiter=',')    # Loading the category matrix
idx = 0 

cat_mat_test_file = "cat_mat_predicted.npy"

for senten in sentences:
	dic_coeff_dict = {}
	
	# Extracting the pos tags out of reviews
	text =  senten.find('text')				# Text tag (tag that contains the review)
	#text = text.string
	#text = text.translate(None,string.punctuation)   # Removing the punctuation words
	tokens = word_tokenize(text.string)
	tokens = filter(lambda x : re.match('\w+',x),tokens)
	
	set_of_words = set(tokens)    # Distinct words set for sentence index = 0

	for i in range(len(dict_of_sets)):    # i is the train sentence index
		dic_coeff_dict[i] = dice_coefficient(set_of_words,dict_of_sets[i])    # dic_coeff_dict[i] = dice coefficient of the current sentence with ith training sentence

	similar_sents = sorted(dic_coeff_dict.items(),key = lambda (k,v) : (v,-k),reverse = True)
	# First index for first K items will give the sentence id of the similar sentences
	# Use those sentence ids to assign categories

	cats_count_over_k_sents = np.zeros(5)		# This will contain the category counts over the 10 most similar sentences 
	m = 0

	for i in range(10):   		# Doing it for 10 most similar sentences
		sent_idx = similar_sents[i][0]    # Gives the index
		
		num_cats = np.sum(cat_mat_train[sent_idx])
		if num_cats > m:
			m = num_cats

		cats_count_over_k_sents+=cat_mat_train[sent_idx]
				
	for i in range(int(m)):
		cat = np.argmax(cats_count_over_k_sents)
		cat_mat_test[idx,cat] += 1 
		cats_count_over_k_sents[cat] = -100				# Making it negative such that next argmax gets the next category


	idx +=1

np.savetxt(cat_mat_test_file,cat_mat_test,delimiter=',')


#######################
# To calculate the baseline accuracy
# Load tfidf_category_test_gold and cat_mat_pred
# Equate them column wise for category wise accuracies
# 
#
#######################