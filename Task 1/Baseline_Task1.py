from __future__ import division
import bs4
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from collections import defaultdict

# Importing the Training XML file to xml_doc
xml_file = open("../raw_data/Restaurants_Train.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')
aspect_set = set()
for senten in sentences:
	

	aspect_terms  = senten.find_all('aspectterm')

	# Adding all aspect terms present in the sentence to aspect_set
	for aspect_term in aspect_terms:
	  	aspect_set.add(aspect_term['term'])

	# Extracting the pos tags out of reviews
	# text =  senten.find('text')				# Text tag (tag that contains the review)
	# tokens = word_tokenize(text.string)
	# pos_tags = nltk.pos_tag(tokens)
# print aspect_set
# Training done. We've extracted all aspect terms and stored them in aspect_set
# TESTING

xml_file = open("../raw_data/Restaurants_Test_Data_phaseB.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')
tp, fp, tn, fn = 0, 0, 0, 0
for senten in sentences:
	# aspect_set_pred = defaultdict(int)
	aspect_set_gold = defaultdict(int)
	
	
	aspect_terms_gold  = senten.find_all('aspectterm')
	for aspect_term in aspect_terms_gold:
		aspect_set_gold[aspect_term['term']]+=1
	# print aspect_set_gold
	text = senten.find('text')
	tokens = word_tokenize(text.string)
	for token in tokens:
		if token in aspect_set: # i.e. we predict is at POSITIVE
			# aspect_set_pred[token]+=1
			if token in aspect_set_gold:
				tp += 1
				aspect_set_gold[token]-=1
			else:
				fp += 1
		else: # i.e. we predict NEGATIVE
			if token in aspect_set_gold:
				fn += 1
			else:
				tn += 1

		if aspect_set_gold[token] is 0:
			aspect_set_gold.pop(token,None)		
print tp,fp,tn,fn
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*precision*recall/(precision+recall)
print "precision = ", precision
print "recall = ", recall
print "F1 score = ", f1



