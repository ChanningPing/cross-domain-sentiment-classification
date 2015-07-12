__author__ = 'qingping'
import csv
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
from numpy import genfromtxt
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.svm import NuSVC


from sklearn.pipeline import Pipeline

#read in reversed data file
with open('/Users/qingping/Documents/CS 613/final project/data/4 domain unlabeled reversed by java.csv', 'rb') as f:
    doc_list = f.read().splitlines()
#change to doc-term count matrix
count_vectorizer = CountVectorizer(min_df=1,max_features=5000)
term_freq_matrix = count_vectorizer.fit_transform(doc_list)

#vocabulary extracted
output = open('/Users/qingping/Documents/CS 613/final project/results/4 domain vocabulary-5000 features.csv', 'w')
for (k, v) in count_vectorizer.vocabulary_.items():
    output.write('%s, %d \n' % (k, v))


#PCA
pca = decomposition.PCA(n_components=100)
pca_estimator=pca.fit(term_freq_matrix.todense())
numpy.savetxt("/Users/qingping/Documents/CS 613/final project/results/variance_ratio.csv", pca.explained_variance_ratio_, delimiter=",")
numpy.savetxt("/Users/qingping/Documents/CS 613/final project/results/variance.csv", pca.explained_variance_, delimiter=",")
numpy.savetxt("/Users/qingping/Documents/CS 613/final project/results/loadings.csv", pca.components_, delimiter=",")
numpy.savetxt("/Users/qingping/Documents/CS 613/final project/results/frequency.csv",  term_freq_matrix.todense().sum(axis=0), delimiter=",")
numpy.savetxt("/Users/qingping/Documents/CS 613/final project/results/100 component membership.csv", numpy.argmax(abs(pca.components_), axis=0), delimiter=",")


#read the training data for book
with open('/Users/qingping/Documents/CS 613/final project/data/book train 1600 reversed.csv', 'rb') as f:
    book_train_list = f.read().splitlines()
book_train_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
book_train_matrix = book_train_vector.fit_transform(book_train_list)
book_train_new_fit=pca_estimator.transform(book_train_matrix.todense())

#read the training data for DVD
with open('/Users/qingping/Documents/CS 613/final project/data/DVD train 1600 reversed.csv', 'rb') as f:
    DVD_train_list = f.read().splitlines()
DVD_train_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
DVD_train_matrix = DVD_train_vector.fit_transform(DVD_train_list)
DVD_train_new_fit=pca_estimator.transform(DVD_train_matrix.todense())


#read the training data for electronics
with open('/Users/qingping/Documents/CS 613/final project/data/electronic train 1600 reversed.csv', 'rb') as f:
    electronic_train_list = f.read().splitlines()
electronic_train_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
electronic_train_matrix = electronic_train_vector.fit_transform(electronic_train_list)
electronic_train_new_fit=pca_estimator.transform(electronic_train_matrix.todense())


#read the training data for kitchen
with open('/Users/qingping/Documents/CS 613/final project/data/kitchen train 1600 reversed.csv', 'rb') as f:
    kitchen_train_list = f.read().splitlines()
kitchen_train_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
kitchen_train_matrix = kitchen_train_vector.fit_transform(kitchen_train_list)
kitchen_train_new_fit=pca_estimator.transform(kitchen_train_matrix.todense())


#read the test data of book
with open('/Users/qingping/Documents/CS 613/final project/data/book test 400 reversed.csv', 'rb') as f:
    book_test_list = f.read().splitlines()
book_test_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
book_test_matrix = book_test_vector.fit_transform(book_test_list)
book_test_new_fit=pca_estimator.transform(book_test_matrix.todense())

#read the test data of DVD
with open('/Users/qingping/Documents/CS 613/final project/data/DVD test 400 reversed.csv', 'rb') as f:
   DVD_test_list = f.read().splitlines()
DVD_test_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
DVD_test_matrix = DVD_test_vector.fit_transform(DVD_test_list)
DVD_test_new_fit=pca_estimator.transform(DVD_test_matrix.todense())


#read the test data of electronics
with open('/Users/qingping/Documents/CS 613/final project/data/electronics test 400 reversed.csv', 'rb') as f:
   electronics_test_list = f.read().splitlines()
electronics_test_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
electronics_test_matrix = electronics_test_vector.fit_transform(electronics_test_list)
electronics_test_new_fit=pca_estimator.transform(electronics_test_matrix.todense())



#read the test data of kitchen
with open('/Users/qingping/Documents/CS 613/final project/data/kitchen test 400 reversed.csv', 'rb') as f:
   kitchen_test_list = f.read().splitlines()
kitchen_test_vector = CountVectorizer(min_df=1,vocabulary=count_vectorizer.vocabulary_)
kitchen_test_matrix = kitchen_test_vector.fit_transform(kitchen_test_list)
kitchen_test_new_fit=pca_estimator.transform(kitchen_test_matrix.todense())



#print book_test_new_fit
train_label=genfromtxt('/Users/qingping/Documents/CS 613/final project/data/train label 1600.csv',delimiter=',')
test_label=genfromtxt('/Users/qingping/Documents/CS 613/final project/data/test label 400.csv',delimiter=',')

print "----PCA + linear SVM"

#output results book->all
BookSVC = svm.LinearSVC()
BookSVC.fit(book_train_new_fit, train_label)
#numpy.savetxt("/Users/qingping/Documents/CS 613/final project/data/variance.csv", pca.explained_variance_, delimiter=",")
#print linearSVC.predict(book_test_new_fit)
print "In-domain error for book-book (trained PCA book data to predict test PCA book data)",BookSVC.score(book_test_new_fit,test_label)
print "Cross-domain error for book-DVD (trained PCA book data to predict test PCA DVD data)",BookSVC.score(DVD_test_new_fit,test_label)
print "Cross-domain error for book-electronics (trained PCA book data to predict test PCA electroincs data)",BookSVC.score(electronics_test_new_fit,test_label)
print "Cross-domain error for book-kitchen (trained PCA book data to predict test PCA kitchen data)",BookSVC.score(kitchen_test_new_fit,test_label)

#output results DVD->all
DVDSVC = svm.LinearSVC()
DVDSVC.fit(DVD_train_new_fit, train_label)
print "Cross-domain error for DVD-book (trained PCA DVD data to predict test PCA book data) ",DVDSVC.score(book_test_new_fit,test_label)
print "In-domain error for DVD-DVD (trained PCA DVD data to predict test PCA DVD data)",DVDSVC.score(DVD_test_new_fit,test_label)
print "Cross-domain error for DVD-electronics (trained PCA DVD data to predict test PCA electroincs data)",DVDSVC.score(electronics_test_new_fit,test_label)
print "Cross-domain error for DVD-kitchen (trained PCA DVD data to predict test PCA kitchen data)",DVDSVC.score(kitchen_test_new_fit,test_label)

#output results electronic->all
electronicSVC = svm.LinearSVC()
electronicSVC.fit(electronic_train_new_fit, train_label)
print "Cross-domain error for electronic-book (trained PCA electronic data to predict test PCA book data) ",electronicSVC.score(book_test_new_fit,test_label)
print "Cross-domain error for DVelectronicD-DVD (trained PCA electronic data to predict test PCA DVD data)",electronicSVC.score(DVD_test_new_fit,test_label)
print "In-domain error for electronics-electronics (trained PCA electronic data to predict test PCA electroincs data)",electronicSVC.score(electronics_test_new_fit,test_label)
print "Cross-domain error for electronic-kitchen (trained PCA electronic data to predict test PCA kitchen data)",electronicSVC.score(kitchen_test_new_fit,test_label)

#output results kitchen->all
kitchenSVC = svm.LinearSVC()
kitchenSVC.fit(kitchen_train_new_fit, train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchenSVC.score(book_test_new_fit,test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchenSVC.score(DVD_test_new_fit,test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchenSVC.score(electronics_test_new_fit,test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchenSVC.score(kitchen_test_new_fit,test_label)

print "----raw feature + linear SVM"

#output baseline book->all
kitchenSVC = svm.LinearSVC()
kitchenSVC.fit(book_train_matrix.todense(), train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchenSVC.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchenSVC.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchenSVC.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchenSVC.score(kitchen_test_matrix.todense(),test_label)

#output baseline DVD->all
kitchenSVC = svm.LinearSVC()
kitchenSVC.fit(DVD_train_matrix.todense(), train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchenSVC.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchenSVC.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchenSVC.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchenSVC.score(kitchen_test_matrix.todense(),test_label)

#output baseline electronics->all
kitchenSVC = svm.LinearSVC()
kitchenSVC.fit(electronic_train_matrix.todense(), train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchenSVC.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchenSVC.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchenSVC.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchenSVC.score(kitchen_test_matrix.todense(),test_label)

#output baseline kitchen->all
kitchenSVC = svm.LinearSVC()
kitchenSVC.fit(kitchen_train_matrix.todense(), train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchenSVC.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchenSVC.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchenSVC.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchenSVC.score(kitchen_test_matrix.todense(),test_label)



print "----PCA + NON-linear SVM"

#NuSVC results book->all
book_clf = NuSVC()
book_clf.fit(book_train_new_fit, train_label)
print "In-domain error for book-book (trained PCA book data to predict test PCA book data)",book_clf.score(book_test_new_fit,test_label)
print "Cross-domain error for book-DVD (trained PCA book data to predict test PCA DVD data)",book_clf.score(DVD_test_new_fit,test_label)
print "Cross-domain error for book-electronics (trained PCA book data to predict test PCA electroincs data)",book_clf.score(electronics_test_new_fit,test_label)
print "Cross-domain error for book-kitchen (trained PCA book data to predict test PCA kitchen data)",book_clf.score(kitchen_test_new_fit,test_label)

#NuSVC results DVD->all
DVD_clf = NuSVC()
DVD_clf.fit(DVD_train_new_fit, train_label)
print "Cross-domain error for DVD-book (trained PCA DVD data to predict test PCA book data) ",DVD_clf.score(book_test_new_fit,test_label)
print "In-domain error for DVD-DVD (trained PCA DVD data to predict test PCA DVD data)",DVD_clf.score(DVD_test_new_fit,test_label)
print "Cross-domain error for DVD-electronics (trained PCA DVD data to predict test PCA electroincs data)",DVD_clf.score(electronics_test_new_fit,test_label)
print "Cross-domain error for DVD-kitchen (trained PCA DVD data to predict test PCA kitchen data)",DVD_clf.score(kitchen_test_new_fit,test_label)

#NuSVC results electronic->all
electronic_clf = NuSVC()
electronic_clf.fit(electronic_train_new_fit, train_label)
print "Cross-domain error for electronic-book (trained PCA electronic data to predict test PCA book data) ",electronic_clf.score(book_test_new_fit,test_label)
print "Cross-domain error for DVelectronicD-DVD (trained PCA electronic data to predict test PCA DVD data)",electronic_clf.score(DVD_test_new_fit,test_label)
print "In-domain error for electronics-electronics (trained PCA electronic data to predict test PCA electroincs data)",electronic_clf.score(electronics_test_new_fit,test_label)
print "Cross-domain error for electronic-kitchen (trained PCA electronic data to predict test PCA kitchen data)",electronic_clf.score(kitchen_test_new_fit,test_label)

#NuSVC results kitchen->all
kitchen_clf = NuSVC()
kitchen_clf.fit(kitchen_train_new_fit, train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchen_clf.score(book_test_new_fit,test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchen_clf.score(DVD_test_new_fit,test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchen_clf.score(electronics_test_new_fit,test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchen_clf.score(kitchen_test_new_fit,test_label)


print "----raw feature + non-linear SVM"
#NuSVC results book->all
book_clf = NuSVC()
book_clf.fit(book_train_matrix.todense(), train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",book_clf.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",book_clf.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",book_clf.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",book_clf.score(kitchen_test_matrix.todense(),test_label)

#NuSVC results DVD->all
DVD_clf = NuSVC()
DVD_clf.fit(DVD_train_new_fit, train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",DVD_clf.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",DVD_clf.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",DVD_clf.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",DVD_clf.score(kitchen_test_matrix.todense(),test_label)

#NuSVC results electronic->all
electronic_clf = NuSVC()
electronic_clf.fit(electronic_train_new_fit, train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",electronic_clf.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",electronic_clf.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",electronic_clf.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",electronic_clf.score(kitchen_test_matrix.todense(),test_label)

#NuSVC results kitchen->all
kitchen_clf = NuSVC()
kitchen_clf.fit(kitchen_train_new_fit, train_label)
print "Cross-domain error for kitchen-book (trained PCA kitchen data to predict test PCA book data) ",kitchen_clf.score(book_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-DVD (trained PCA kitchen data to predict test PCA DVD data)",kitchen_clf.score(DVD_test_matrix.todense(),test_label)
print "Cross-domain error for kitchen-electronics (trained PCA kitchen data to predict test PCA electroincs data)",kitchen_clf.score(electronics_test_matrix.todense(),test_label)
print "In-domain error for kitchen-kitchen (trained PCA kitchen data to predict test PCA kitchen data)",kitchen_clf.score(kitchen_test_matrix.todense(),test_label)
