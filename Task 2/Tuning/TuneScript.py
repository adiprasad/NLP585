import numpy as np 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib

gamma_range = [0.01,0.001,0.0001]
C_range = [1,10,100,1000]
dec_func = ['ovr']

y_train = np.loadtxt('../Y_train_categories_matrix/y_train.npy',delimiter=',')
y_test = np.loadtxt('../Y_test_categories_matrix/y_test.npy',delimiter=',')
X_NAV_Train = np.loadtxt('../Train_Features/train_nav.csv',delimiter=',')
X_NAV_TN_Train = np.loadtxt('../Train_Features/train_nav_tn.csv',delimiter=',')
X_NAV_TN_CS_Train = np.loadtxt('../Train_Features/train_nav_tn_cs.csv',delimiter=',')
X_NAV_Test = np.loadtxt('../Test_Features/test_nav.csv',delimiter=',')
X_NAV_TN_Test = np.loadtxt('../Test_Features/test_nav_tn.csv',delimiter=',')
X_NAV_TN_CS_Test = np.loadtxt('../Test_Features/test_nav_tn_cs.csv',delimiter=',')


svc_nav = SVC()

prediction_pipeline = Pipeline([
            ('prediction', svc_nav)
        ])

prediction_parameters = {
            'prediction__C': C_range,
            'prediction__kernel': ['rbf', 'linear'],
            'prediction__gamma': gamma_range,
	    'prediction__decision_function_shape': dec_func
        }


gr_nav = GridSearchCV(prediction_pipeline, prediction_parameters, scoring='accuracy',verbose=5,cv=5)
clf_nav = OneVsRestClassifier(gr_nav) 

clf_nav_tn = OneVsRestClassifier(GridSearchCV(prediction_pipeline, prediction_parameters, scoring='accuracy',verbose=5,cv=5))
clf_nav_tn_cs = OneVsRestClassifier(GridSearchCV(prediction_pipeline, prediction_parameters, scoring='accuracy',verbose=5,cv=5))


clf_nav.fit(X_NAV_Train, y_train)
clf_nav_tn.fit(X_NAV_TN_Train, y_train)
clf_nav_tn_cs.fit(X_NAV_TN_CS_Train, y_train)

joblib.dump(clf_nav, 'clf_nav.pkl')
joblib.dump(clf_nav_tn, 'clf_nav_tn.pkl')
joblib.dump(clf_nav_tn_cs, 'clf_nav_tn_cs.pkl')


'''
clf_nav = joblib.load('clf_nav.pkl')
clf_nav_tn = joblib.load('clf_nav_tn.pkl')
clf_nav_tn_cs = joblib.load('clf_nav_tn_cs.pkl')
'''


