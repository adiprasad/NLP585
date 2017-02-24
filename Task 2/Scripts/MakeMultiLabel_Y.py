"""
Input : Takes a train/test XML file with sentences and their aspect categories
Output : Writes a 2-D multi-label boolean Y matrix of size (num_sentences X num_categories) for that file
"""

import bs4
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np

# Importing the XML file to xml_doc
xml_file = open("../../Datasets/SemEval2014/Extracted/Restaurants/test/Restaurants_Test_Data_phaseB.xml",'r')
xml_doc = xml_file.read()
xml_file.close()
#print xml_doc

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

category_to_index_map = {'food':0,'service':1,'ambience':2,'price':3,'anecdotes/miscellaneous':4}

output_file = "../Restaurants/Y_test_categories_matrix/y_test.npy"

y = np.zeros(5)

for senten in sentences:
	category_bool_list = np.zeros(5)

	aspect_categories  = senten.find_all('aspectcategory')

	# Lighting up the zeros array for the categories encountered in this sentence
	for aspect_category in aspect_categories:
	  	category_bool_list[category_to_index_map[aspect_category['category']]] = 1

	# V-stacking the multi-label array for the current sentence with the main
	y = np.vstack((y,category_bool_list))	


# Shave the first row for y, since it was a dummy row just to facilitate the functioning of vstack
np.savetxt(output_file,y[1:,],delimiter=',')
print "File writing completed"
#################################################
# To read:
# np.loadtxt('filename.csv',delimiter=',')

