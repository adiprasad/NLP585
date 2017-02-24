import bs4
from bs4 import BeautifulSoup
import numpy as np 
import collections 
from collections import defaultdict

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
polarities = {'positive':0,'negative':1,'neutral':2,'conflict':3}
soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

aspect_set = set()    # Containing all the aspect terms of the training set
aspect_polarity_cnt = defaultdict()		# Count of every aspect term across the training set across all polarities

for senten in sentences:

	aspect_terms  = senten.find_all('aspectterm')

	# Adding all aspect terms present in the sentence to aspect_set
	for aspect_term in aspect_terms:
		term = aspect_term['term']
	  	aspect_set.add(term)

	  	if aspect_term not in aspect_polarity_cnt.keys():
	  		aspect_polarity_cnt[term] = np.zeros(4)

	  	asp_polarity = aspect_term['polarity']
	  	aspect_polarity_cnt[term][polarities[asp_polarity]] += 1  		# Incrementing the polarity count for that aspect term

	
xml_file = open("../../Test_Data_Gold/ABSA_Gold_TestData/Restaurants_Test_Gold.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

aspect_polarities_pred = []
y_pred_file = 'y_pred_baseline.npy'

for senten in sentences:
	dic_coeff_dict = {}

	aspect_terms  = senten.find_all('aspectterm')
	
	for aspect_term in aspect_terms:
		term = aspect_term['term']
		if term not in aspect_set:
			aspect_polarities_pred.append(polarities['positive'])   # If not in aspect set then assign the most frequent polarity
		else:   # Assign the highest encountered polarity for this aspect term in the training set
			highest_p = np.argmax(aspect_polarity_cnt[term])
			aspect_polarities_pred.append(highest_p)


np.savetxt(y_pred_file,aspect_polarities_pred,delimiter=',')

#######################
# To calculate the baseline accuracy
# Load tfidf_category_test_gold and cat_mat_pred
# Equate them column wise for category wise accuracies
# 
#
#######################