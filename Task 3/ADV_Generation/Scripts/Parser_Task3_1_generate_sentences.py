import bs4
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
import re

# Importing the XML file to xml_doc
xml_file = open("../raw_data/Restaurants_Train.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

outfile = open("Restaurants_Train_Sentences.txt",'w+')
for senten in sentences:
	text = senten.find('text').string				# Text tag (tag that contains the review)
	text = re.sub(r'\.\.+',',',text) # replace multiple periods by comma
	text = re.sub(r'([?!])+',r'\1',text) # replace multiple ? or ! with itself
	text = re.sub('\.+','',text) # remove all periods
	# if re.match('.+[a-zA-Z0-9\)][.!?]\Z',text.string) is not None: # ends with punct
	# if re.match('.+[a-zA-Z0-9\)][.!?]\Z',text.string) is not None: # ends with punct
	# 	outfile.write(text.string + "\n")
	# else:
	# 	outfile.write(text.string + " .\n")
	outfile.write(text + " .\n")

outfile.close()

# SAME FOR TEST ----------------------------------------------------------------------------------------------------------------------------------
# Importing the XML file to xml_doc
xml_file = open("../raw_data/Restaurants_Test_Data_phaseB.xml",'r')
xml_doc = xml_file.read()
xml_file.close()

soup = BeautifulSoup(xml_doc,"html.parser")
sentences = soup.find_all('sentence')

outfile = open("Restaurants_Test_Sentences.txt",'w+')

for senten in sentences:
	text =  senten.find('text').string				# Text tag (tag that contains the review)
	text = re.sub(r'\.\.+',',',text)
	text = re.sub(r'([?!])+',r'\1',text)
	text = re.sub('\.+','',text)
	# if re.match('.+[a-zA-Z0-9\)][.!?]\Z',text.string) is not None: # ends with punct
	# 	outfile.write(text.string + "\n")
	# else:
	# 	outfile.write(text.string + " .\n")
	outfile.write(text + " .\n")
outfile.close()

#######################################
# scriptdir=`dirname $0`
# java -mx1024m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ../task3/Restaurants_Train_Sentences.txt >> ../task3/Restaurants_Train_Sentencies_Dependencies.txt
# java -mx1024m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ../task3/Restaurants_Test_Sentences.txt >> ../task3/Restaurants_Test_Sentencies_Dependencies.txt
#
#######################################
