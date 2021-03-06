import bs4
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize

# Importing the XML file to xml_doc
xml_file = open("../Datasets/SemEval2014/Extracted/Restaurants/train/Restaurants_Train.xml",'r')
xml_doc = xml_file.read()
xml_file.close()
#print xml_doc

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

pos_tag_file = open("../Datasets/SemEval2014/Extracted/Restaurants/train/Restaurants_Train_pos_tagged2.txt",'w+')

for senten in sentences:
	aspect_set = set()

	aspect_terms  = senten.find_all('aspectterm')

	# Adding all aspect terms present in the sentence to aspect_set
	for aspect_term in aspect_terms:
	  	aspect_set.add(aspect_term['term'])

	# Extracting the pos tags out of reviews
	text =  senten.find('text')				# Text tag (tag that contains the review)
	tokens = word_tokenize(text.string)
	pos_tags = nltk.pos_tag(tokens)
	
	
	#For writing Aspect/Non-Aspect[tab_sep] pos_tag to file for all the tokens in the sentence
	cntr = 1
	num_tokens = len(pos_tags)

	for key,val in pos_tags:
		if key in  aspect_set:
			aspect_bool = "A"
		else:
			aspect_bool = "NA"

		if cntr==1:
			pos_tag_file.write(aspect_bool+"\t"+"pos_tag="+val+"\t"+"__BOS__"+"\n")
		elif cntr == num_tokens:
			pos_tag_file.write(aspect_bool+"\t"+"pos_tag="+val+"\t"+"__EOS__"+"\n")
		else:
			pos_tag_file.write(aspect_bool+"\t"+"pos_tag="+val+"\n")

		cntr+=1

	pos_tag_file.write("\n")

pos_tag_file.close()

