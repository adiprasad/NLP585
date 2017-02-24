import bs4
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
import gensim

# Importing the XML file to xml_doc
xml_file = open("../Datasets/SemEval2014/Extracted/Restaurants/test/Restaurants_Test_Data_phaseB.xml",'r')
xml_doc = xml_file.read()
xml_file.close()
#print xml_doc

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')
model = gensim.models.Word2Vec.load_word2vec_format('/Volumes/Data/School/Study/585/Final_Project/Datasets/Google_News_Word2Vec/GoogleNews-vectors-negative300.bin',binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin',binary=True)

# print model['computer'][1]

pos_tag_file = open("../Datasets/SemEval2014/Extracted/Restaurants/test/Restaurants_Test_pos_tagged3.txt",'w+')

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
	num_tokens = len(pos_tags)

	for key,val in pos_tags:
		if key in  aspect_set:
			aspect_bool = "A"
		else:
			aspect_bool = "NA"

		pos_tag_file.write(aspect_bool+"\t")
		
		if key in model.vocab:
			word_vec = model[key]
			
			for i in range(len(word_vec)):
				pos_tag_file.write("dim"+str(i)+":"+str(word_vec[i])+"\t")

		pos_tag_file.write("pos_tag="+val+"\n")

		

	pos_tag_file.write("\n")

pos_tag_file.close()

