import bs4
from bs4 import BeautifulSoup
import collections 
from collections import defaultdict
import re 

# Importing the XML file to xml_doc
xml_file = open("../../Test_Data_Gold/ABSA_Gold_TestData/Restaurants_Test_Gold.xml",'r')
dep_tree_file = open("../restaurant_dependency_tree_zip/Test/Restaurants_Test_Sentencies_Dependencies.txt",'r')
xml_doc = xml_file.read()
dep_doc = dep_tree_file.read()
xml_file.close()
dep_tree_file.close()
#print xml_doc

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')
depen_trees_list = dep_doc.split("\n\n")

senten_to_aspect_terms = defaultdict(list)			# key = sentence number, value = list of aspect terms
senten_counter = 0				# Sentence index to be used as the key in the dictionary
senten_id_array = list()		

for senten in sentences:
	aspect_polarity_list = list() 

	senten_id = senten['id']				# Finding the sentence id 
	senten_id_array.append(senten_id)		# Storing in the sentence id in a separate array

	aspect_terms  = senten.find_all('aspectterm')

	# Adding all aspect terms present in the sentence to aspect_set
	for aspect_term in aspect_terms:
		asp_term = aspect_term['term']				# Extract the aspect 
		asp_polarity = aspect_term['polarity']		# Extract its polarity
		term_polarity = asp_term + "#" + asp_polarity		# Club them together separated by a hash symbol
	  	aspect_polarity_list.append(term_polarity)			# Append to the aspect list which will be stored at key = sentence_index 

	senten_to_aspect_terms[senten_counter] = aspect_polarity_list

	senten_counter += 1


# After this loop ends, we have a dictionary of lists with key = sentence number (0 indexing) and value = [list of aspects in it]

crude_data_set_file = open("../restaurant_dependency_tree_zip/Test/ADV_file_Restaurant_Test.txt",'w')
crude_data_set_file.write("Sentence_ID\t" + "Sentence_Index\t" + "Aspect_Index\t" + "Aspect\t" + "Dep_Words\t" + "Aspect_Polarity\n") 		# Populating the headers

for senten_idx, aspect_list in senten_to_aspect_terms.items():
	senten_id = senten_id_array[senten_idx]
	aspect_term_to_dep_words = defaultdict(list)			# Creating a dictionary for each sentence with key = aspect term and value = list of dependency words
	

	aspect_polarity_list = senten_to_aspect_terms[senten_idx]		# Load the aspect#polaritiy list for this sentence
	aspect_list = map(lambda x : x.split("#")[0], aspect_polarity_list)		# Extract the list of aspects for this sentence
	polarity_list = map(lambda x : x.split("#")[1], aspect_polarity_list)	# Extract the list of polarities for this sentence

	dep_tree_entries = depen_trees_list[senten_idx].split("\n") 			# Storing the tree rules for the current sentence's dependies in a list
	dep_tree_entries = map(lambda x : re.search('\((.+)\)',x).group(1) , dep_tree_entries)			# Extracting the rules/relationships in each bracket and discarding the rest
	dep_tree_entry_tuples = map(lambda x : x.split(" ,"), dep_tree_entries)		# Converting the rules into tuples 


	for tup in dep_tree_entry_tuples :
		comm_sep_pair = tup[0]			#tup is a list in itself with only one element (implementation issues)
		#print "Comm sep pair: " + comm_sep_pair 		# for debugging
		left, right = comm_sep_pair.split(", ")

		#print left, right						# debug
		word1 = left.split("-")[0]
		word2 = right.split("-")[0]
		
		# Creating a dictionary with key = word, value = list of dependency words for that word
		aspect_term_to_dep_words[word1].append(word2)
		aspect_term_to_dep_words[word2].append(word1)


	for i in range(len(aspect_list)):			# Write to file
		aspect = aspect_list[i]
		
		aspect_entities = aspect.split(" ")		# All of this added to handle multi-word aspects
		
		combined_dep_words = []				# Will be filled with dependency words for all the words in a multi-word aspect
		
		for ent_t in aspect_entities:
			combined_dep_words.extend(aspect_term_to_dep_words[ent_t])

		set_combined_dep_words = set(combined_dep_words)			# Converting to set to kill duplicates
		set_aspect_entities = set(aspect_entities)					

		set_combined_dep_words  = set_combined_dep_words - set_aspect_entities		# If aspect entities present in the dependency words set, kill them
		list_dep_words = list(set_combined_dep_words)
		
		crude_data_set_file.write(str(senten_id) + "\t" + str(senten_idx) + "\t" + str(i) + "\t" + str(aspect) + "\t" + str(list_dep_words) + "\t" + polarity_list[i] + "\n")






		
