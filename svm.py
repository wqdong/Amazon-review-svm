import numpy as np
import json
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from string import punctuation
import random


def read_vector_file(fname):
    """
      Reads and returns a vector from a file specified by fname.
      Input:
        fname- string specifying a filename
      Returns an (n,) array where n is the number of lines in the file.
    """
    return np.genfromtxt(fname)


def load_data(fname):
    """
      Reads and returns data from a file specified by fname.
      Input:
        fname- string specifying a filename
      Returns a list of n reviews and
      an (n,) array of ratings where n is the number of json objects in the file.
    """
    with open(fname) as input_file:
        review_data = json.load(input_file)
    reviewText = [x['reviewText'] for x in review_data]
    y = [int(x['rating']) for x in review_data]
    return (reviewText, np.array(y))

def load_heldout_data(fname):
    """
      Reads and returns data from a file specified by fname.
      Input:
        fname- string specifying a filename
      Returns a list of n reviews and
      where n is the number of json objects in the file.
    """
    with open(fname) as input_file:
        review_data = json.load(input_file)
    reviewText = [x['reviewText'] for x in review_data]
    return reviewText
    
def write_scores(vec, outfile):
    """
      Writes your label vector the a given file.
      The vector must be of shape (70, ) or (70, 1),
      i.e., 100 rows, or 100 rows and 1 column.
      Input:
          vec- (100,) or (100,1) array containing predicted scores,
          outfile- string with filename fto write to
    """

    if(vec.shape[0] != 100):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    np.savetxt(outfile, vec)


def extract_dictionary(reviews):
    """
      Reads a list of reviews (strings), and returns a dictionary of distinct words
      mapping from the distinct word to its index (ordered by when it was found).
      Input:
        reviews- list of reviews (strings)        
      Returns: 
        a dictionary of distinct words
        mapping from the distinct word to its index (ordered by when it was found).
    """
    word_list = {}
    i = 0
    for rev in reviews:
    	for char in punctuation:
            rev = rev.replace(char," "+char+" ")

        rev_list = rev.split()
        for word in rev_list:
            word = word.lower()
            if word not in word_list.values():
                word_list[i] = word
                i = i+1
    	 
    return word_list

def extract_dictionary_2(reviews):
    """
      Reads a list of reviews (strings), and returns a dictionary of distinct words
      mapping from the distinct word to its index (ordered by when it was found).
      Input:
        reviews- list of reviews (strings)        
      Returns: 
        a dictionary of distinct words
        mapping from the distinct word to its index (ordered by when it was found).
    """
    word_list = {}
    rev_list = []
    i = 0
    for rev in reviews:
    	rev = rev.lower()
    	for char in punctuation:
            rev = rev.replace(char," "+char+" ")
        rev_list = rev_list + rev.split()
    for word in rev_list:
        if (rev_list.count(word) > 2) and (word not in word_list.values()):
            word_list[i] = word
            i = i+1
    	 
    return word_list

def extract_feature_vectors(reviews, word_list):
    """
      Reads a list of reviews (strings) and the dictionary of words in the reviews
      to generate {1, 0} feature vectors for each review. The resulting feature
      matrix should be of dimension (number of reviews, number of words).
      Input:
        reviews- list of reviews (strings)
        word_list- dictionary of words mapping to indices
      Returns: 
        a feature matrix of dimension (number of reviews, number of words)
    """
    feature_matrix = []
    	   		
    for rev in reviews:
    	for char in punctuation:
    		rev = rev.replace(char," "+char+" ")
    		
    	rev = rev.lower()
    	rev_list = rev.split()
        vec = []
        for v in word_list.values():
            if v in rev_list:
                vec.append(1)
            else:
                vec.append(0)
        feature_matrix.append(vec)
    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """
        Calculates the performance metric based on the agreement between the
        true labels and the predicted labels
        Input:
          y_true- (n,) array containing known labels
          y_pred- (n,) array containing predicted scores
          metric- string option used to select the performance measure
        Returns: the performance as a np.float64
    """
    labels = [1,-1]
    num = len(y_true)
    if metric == 'f1-score':
        performance_score = metrics.f1_score(y_true,y_pred)
    elif metric == "auroc":
        performance_score = metrics.roc_auc_score(y_true,y_pred)
    elif metric == "precision":
        performance_score = metrics.precision_score(y_true,y_pred)
    elif metric == "sensitivity":
        conf = metrics.confusion_matrix(y_true,y_pred,labels)
        performance_score = np.float64(conf[0][0])/(conf[0][0]+conf[0][1])
    elif metric == "specificity":
        conf = metrics.confusion_matrix(y_true,y_pred,labels)
        performance_score = np.float64(conf[1][1])/(conf[1][0]+conf[1][1])
    else:
        performance_score = metrics.accuracy_score(y_true,y_pred,labels)
    return performance_score

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
        Splits the data, X and y, into k-folds and runs k-fold crossvalidation:
        training a classifier on K-1 folds and testing on the remaining fold.
        Calculates the k-fold crossvalidation performance metric for classifier
        clf by averaging the performance across folds.
        Input:
          clf- an instance of SVC()
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specificyin the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns: average 'test' performance across the k folds as np.float64
    """
    skf = StratifiedKFold(y,k)
    sum = 0
    num = len(y)
    for train_index, test_index in skf:
        trainx = []
        trainy = []
        testx = []
        testy = []
        for i in range(num):
            if i in train_index:
                trainx.append(X[i])
                trainy.append(y[i])
            else:               
            	testx.append(X[i])
                testy.append(y[i])
        clf.fit(trainx,trainy)
        dis = clf.decision_function(testx)
        if metric == 'auroc':
        	predy = dis
        else:
        	predy = np.sign(dis)
        score = performance(testy,predy,metric)
       
        sum = sum + score

    avg_performance_score = np.float64(sum/k)

    return avg_performance_score
 

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
        Sweeps different settings for the hyperparameter of a linear-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns the parameter value for linear-kernel SVM, that 'maximizes' the
        average 5-fold CV performance.
    """
    bestC = 0
    bestP = 0
    for c in C_range:
    	p = cv_performance(SVC(kernel='linear', C = c),X,y,5,metric)
    	"print c,p"
    	if p > bestP:
			bestP = p
			bestC = c
			
    return bestC
    
def select_param_linear_l1(X, y, k=5, metric="accuracy", C_range = [], penalty='l1'):
    """
        Sweeps different settings for the hyperparameter of a linear-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specifying the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns the parameter value for linear-kernel SVM, that 'maximizes' the
        average 5-fold CV performance.
    """
    bestC = 0
    bestP = 0
    for c in C_range:
    	
		clf = LinearSVC(penalty='l1',dual=False,C=c)
		
		p = cv_performance(clf,X,y,5,metric)
		"print c,p"
		if p > bestP:
			bestP = p
			bestC = c
			
    return bestC

def select_param_quadratic(X, y, k=5, metric="accuracy", C_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          k- int specificyin the number of folds (default=5)
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns the parameter value(s) for an quadratic-kernel SVM, that 'maximize'
        the average 5-fold CV performance.
    """
    ranger = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]
    bestC = 0
    bestR = 0
    bestP = 0
    
    for c in C_range:
    	for r in ranger:
    		clf = SVC(kernel='poly',degree=2,C=c,coef0=r)
    		p = cv_performance(clf,X,y,5,metric)
    		"print c,r,p"
    		if p > bestP:
    			bestP = p
    			bestC = c
    			bestR = r
    		
    return bestR, bestC

def performance_CI(clf, X, y, metric="accuracy"):
    """
        Estimates the performance of clf on X,y and the corresponding 95%CI
        (lower and upper bounds)
        Input:
          clf-an instance of SVC() that has already been fit to data
          X- (n,d) array of feature vectors, where n is the number of examples
             and d is the number of features
          y- (n,) array of binary labels {1,-1}
          metric- string specifying the performance metric (default='accuracy',
                   other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                   and 'specificity')
        Returns:
            a tuple containing the performance of clf on X,y and the corresponding
            confidence interval (all three values as np.float64's)
    """
    dis = clf.decision_function(X)
    if metric == 'ausoc':
    	predy = dis
    else:
    	predy = np.sign(dis)
    score = performance(y,predy,metric)
    s_list = []
    for i in range(1000):
		setx = []
		sety = []
		for j in range(len(y)):
			r = random.randint(0,len(y)-1)
			setx.append(X[r])
			sety.append(y[r])
		dis = clf.decision_function(setx)
		if metric == 'ausoc':
			predy = dis
		else:
			predy = np.sign(dis)
		s = performance(sety,predy,metric)
		s_list.append(s)
    s_list = np.sort(s_list)
    lower = s_list[25]
    upper = s_list[975]
    return score, lower, upper
    
def main(filename,heldoutfile):
	
	data,rate = load_data(filename)
	"2 Feature Extraction"
	"part a"
	dic = extract_dictionary(data)
	
	"part b"
	mat = extract_feature_vectors(data,dic)
	
	"part c"
	trainv = []
	testv = []
	trainr = []
	testr = []
	for i in range(400):
		if i < 300:
			trainv.append(mat[i])
			trainr.append(rate[i])
		else:
			testv.append(mat[i])
			testr.append(rate[i])
	
	
	print "#3 Hyperparameter#"
	
	met = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
	
	rangec = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]
	rangec2 = [0.0001,0.001,0.01,0.1,0.5,1.0,5.0,10.0,50.0,100.0,1000.0,10000.0]
	
	print "#3.1 linear#" 
  	
	for m in met:
		c_value = select_param_linear(trainv, trainr,5,m,rangec)
		print m,"bestC", c_value
	
	print "#3.2 quadratic#"
	
	for m in met:
		r,c = select_param_quadratic(trainv,trainr,5,m,rangec)
		print m, "bestR, bestC",r,c
	
	print "#3.4 L1 penalty and Squared Hinge Loss#"		
	
	for m in met:
		c_value = select_param_linear_l1(trainv, trainr,5,m,rangec)
		print m,"bestC", c_value
	
	
	print "#3.5 Evaluation#"
	
	clf_linear = SVC(kernel = 'linear',C = 10.0)
	clf_linear = clf_linear.fit(trainv,trainr)
	clf_quad = SVC(kernel = 'poly',degree = 2, C = 10000.0, coef0 = 0.01)
	clf_quad = clf_quad.fit(trainv,trainr)
	clf_l1 = LinearSVC(penalty='l1',dual=False,C=100.0)
	clf_l1 = clf_l1.fit(trainv,trainr)
	
	print "#4 Bootstrap Confidence Intervals#"
	
	print "#linear#"
	for m in met:
		p = performance_CI(clf_linear, testv, testr, m)
		print m, p
	
	print "#quad#"
	for m in met:
		p = performance_CI(clf_quad, testv, testr, m)
		print m, p
	
	print "#l1#"
	for m in met:
		p = performance_CI(clf_l1, testv, testr, m)
		print m, p
	
	print "#6 Heldout#"
  	clf_linear = SVC(kernel = 'linear',C = 0.1)
  	clf_linear = clf_linear.fit(mat,rate)
  	clf_quad = SVC(kernel = 'poly',degree = 2, C = 0.01, coef0 = 10000.0)
  	clf_quad = clf_quad.fit(mat,rate)
  	clf_l1 = LinearSVC(penalty='l1',dual=False,C=1000.0)
  	clf_l1 = clf_l1.fit(mat,rate)

	heldoutdata = load_heldout_data(heldoutfile)
	heldoutmat = extract_feature_vectors(heldoutdata,dic)
	vec_linear = clf_linear.decision_function(heldoutmat)
  	vec_quad = clf_quad.decision_function(heldoutmat)
  	vec_l1 = clf_l1.decision_function(heldoutmat)
  	
  	print "linear mean",np.mean(vec_linear)
  	print "quad mean",np.mean(vec_quad)
  	print "l1 mean",np.mean(vec_l1)
	
	write_scores(vec_linear,"wqdong.txt")
    write_scores(vec_quad,"wqdong.txt")
    write_scores(vec_l1,"wqdong.txt")
	
main("reviews.json","held_out_reviews.json")

