'''
Description : Script to do multilabel multiclass classification for Task 2 (Category Detection)
Number of categories/labels : 6

Set of different features used :-
A) NAV 
B) NAV + TN 
C) NAV + TN + CS

Input:-

Case A :- 
X (3044,300), Y(3044,6)

Case B :- 
X (3044,301), Y(3044,6)

Case C :-
X (3044,306), Y(3044,6)

Output :-

3 Folders for each case, 5 folders inside each folder for every classifer. 
Model dump, CV Results, Best Params, classification report, 

APPROACH :-

1. Train 6 different SVMs (one for each label) i.e split the training data into 5 cases for 5 labels [(X_Train,Y_Train[,0]),(X_Train,Y_Train[,1]) and so on..]
2. Apply GridSearchCV for each of the 5 classifiers, for every case. 
3. Log in/Pickle dump models/Write CV Results to files, neatly in different folders with neat naming convention for each classifier, for each case 
4. Analyse peacefully. 
'''


import numpy as np 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import os 
from os import listdir
from os.path import isfile, join
import pandas
from pandas import DataFrame


class MultiClass_MultiLabel_Task2(object):

	
	# Returns a fit Grid Search CV object
	def fitAndReturnModel(self,prediction_pipeline,prediction_parameters,X_train,Y_train,cv=5):
		clf = GridSearchCV(prediction_pipeline, prediction_parameters, cv=5,
                           scoring='accuracy',verbose=5)
		clf.fit(X_train,Y_train)
		return clf

	# Takes a fit model and joblib dumps it to the specified location
	def dumpModelAndStats(self,clf,output_path,category_map,feature_map,cat_idx,feat_idx):
		
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		folder = join(output_path,feature_map[feat_idx])
		if not os.path.exists(folder):
			os.makedirs(folder)

		folder = join(folder,category_map[cat_idx])
		if not os.path.exists(folder):
			os.makedirs(folder)

		model_file_path = join(folder,"SVM_clf.pkl")
		cv_file_path =join(folder,"SVM_clf_CV_Results.csv")
		other_stats_path = join(folder,"SVM_clf_other_stats.txt")

		feat_tag = feature_map[feat_idx]
		
		X_test = self.__dict__['X_'+feat_tag+'_Test']
		y_pred = clf.predict(X_test)

		# CV results
		cv_res = DataFrame(clf.cv_results_)
		cv_res.to_csv(cv_file_path,sep=',')

		# Other stats
		class_rep = str(classification_report(self.y_test[:,cat_idx],y_pred))
		acc = str(np.mean(y_pred == self.y_test[:,cat_idx]))
		best_par = str(clf.best_params_)
		best_est = str(clf.best_estimator_)
		best_idx = str(clf.best_index_)
		best_scr = str(clf.best_score_)

		f_stats = open(other_stats_path,'wb')
		f_stats.write(class_rep + "\nAccuracy on test set : "+ acc + "\nBest params : " + best_par + "\nBest estimator : " + best_est + "\nBest index : " + best_idx + "\nBest score" + best_scr)
		f_stats.close()

		# Model dump
		joblib.dump(clf,model_file_path)
		print "Dumped model for feature = " + str(feat_tag) + " and category = " + str(category_map[cat_idx])



	# Loads all kinds of training, test data etc
	def loadData(self,train_path,test_path):
		self.y_train = np.loadtxt(join(train_path, 'y_train.npy'),delimiter=',')
		self.y_test = np.loadtxt(join(test_path, 'y_test.npy'),delimiter=',')
		# self.X_NAV_Train = np.loadtxt(join(train_path, 'train_nav.csv'),delimiter=',')
		# self.X_NAV_TN_Train = np.loadtxt(join(train_path, 'train_nav_tn.csv'),delimiter=',')
		self.X_NAV_TN_CS_Train = np.loadtxt(join(train_path, 'train_nav_tn_cs_aomax.csv'),delimiter=',')
		# self.X_NAV_Test = np.loadtxt(join(test_path, 'test_nav.csv'),delimiter=',')
		# self.X_NAV_TN_Test = np.loadtxt(join(test_path, 'test_nav_tn.csv'),delimiter=',')
		self.X_NAV_TN_CS_Test = np.loadtxt(join(test_path, 'test_nav_tn_cs_aomax.csv'),delimiter=',')

	# Returns the needed Train Data Pair out of all combinations according to indices 
	def getTrainData(self,category_map,feature_map,cat_idx,feat_idx):
		y = self.y_train[:,cat_idx]
		feat_tag = feature_map[feat_idx]
		X = self.__dict__['X_'+feat_tag+'_Train']

		return X,y 




if __name__ == '__main__':
	train_path = "/Volumes/Data/School/Study/585/Final_Project/Task_2/Restaurants/Train_Features"
	test_path = "/Volumes/Data/School/Study/585/Final_Project/Task_2/Restaurants/Test_Features"

	model_dump_path = "/Volumes/Data/School/Study/585/Final_Project/Task_2/Restaurants/MC_ML_SVM_new"

	# category_map = ['food','service','ambience','price','anecdotes_miscellaneous']
	# feature_map = ['NAV','NAV_TN','NAV_TN_CS']

	category_map = ['food']
	feature_map = ['NAV_TN_CS']

	MLC_MLB_SVM = MultiClass_MultiLabel_Task2()
	MLC_MLB_SVM.loadData(train_path,test_path) 


	# gamma_range = [0.5,0.1,0.01,0.001,0.0001]
	# C_range = [0.01, 0.1, 1, 10, 100, 1000]
	gamma_range = [0.5,0.1,0.01,0.001]
	C_range = [1, 10, 100, 1000]
	dec_func = ['ovr']

	prediction_pipeline = Pipeline([
            ('prediction', SVC())
        ])

	prediction_parameters = {
            'prediction__C': C_range,
            'prediction__kernel': ['rbf', 'linear'],
            'prediction__gamma': gamma_range,
	    	'prediction__decision_function_shape': dec_func
    	}

	for j in range(len(feature_map)):
		for i in range(len(category_map)):
			X_tr,Y_tr = MLC_MLB_SVM.getTrainData(category_map,feature_map,i,j)
			print X_tr.shape, Y_tr.shape
			clf = MLC_MLB_SVM.fitAndReturnModel(prediction_pipeline,prediction_parameters,X_tr,Y_tr,5)
			MLC_MLB_SVM.dumpModelAndStats(clf,model_dump_path,category_map,feature_map,i,j)







