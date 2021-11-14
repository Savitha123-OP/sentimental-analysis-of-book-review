#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
import random
from nltk.corpus import sentiwordnet as swn
Adata_train=pd.read_csv("C:/Users/ADMIN/Documents/flipkartrainDataset.csv",encoding="ISO-8859-1")
Adata_test=pd.read_csv("C:/Users/ADMIN/Documents/flipkartTestDataset.csv",encoding="ISO-8859-1")

training_data=[]
test_data=[]

l1=Adata_train['REVIEWS'].values
l2=Adata_train['SENTIMENTS'].values

for i in range(1,len(l1)):
        l=[]
        l.append(l1[i])
        l.append(l2[i])
        training_data.append(l)
l3=Adata_test['REVIEWS'].values
l4=Adata_test['SENTIMENTS'].values

for i in range(1,len(l3)):
        l=[]
        l.append(l3[i])
        l.append(l4[i])
        test_data.append(l)


# In[28]:


# Examples from training data
print(training_data[1])
print(len(training_data), len(test_data))


# In[29]:


#Required for Bag of words (unigram features) creation
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
#lemmatizer.lemmatize
import re
vocabulary = []
lemmatizer = WordNetLemmatizer()
for tup in training_data:
    words = re.sub("[^\w]", " ",tup[0]).split() #remove special charater and tokeniation
    cleaned_text = [lemmatizer.lemmatize(w.lower()) for w in words if w not in set(stopwords.words('english'))] #remove stop words  and stemming 
    vocabulary.extend(cleaned_text)
    

print(len(vocabulary))
vocabulary = list(set(vocabulary))
vocabulary.sort() #sorting the list

print(len(vocabulary))
print(vocabulary)


# In[30]:


def get_unigram_features(data,vocab):
    fet_vec_all = []
    for tup in data:
        single_feat_vec = []
        sent = tup[0].lower() #lowercasing the dataset
    
        for v in vocab:
            if sent.__contains__(v):
                single_feat_vec.append(1)
            else:
                single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)       

    return fet_vec_all


# In[31]:


def get_senti_wordnet_features(data):
    fet_vec_all = []
    for tup in data:
        sent = tup[0].lower()
        words = sent.split()
        pos_score = 0
        neg_score = 0
        for w in words:
           
            senti_synsets = swn.senti_synsets(w.lower())
            
            for senti_synset in senti_synsets:
                p = senti_synset.pos_score()
                n = senti_synset.neg_score()
                
                pos_score+=p
                neg_score+=n
                break #take only the first synset (Most frequent sense)
        fet_vec_all.append([float(pos_score),float(neg_score)])
    return fet_vec_all


# In[32]:


def merge_features(featureList1,featureList2):
    # For merging two features
    if featureList1==[]:
        return featureList2
    merged = []
    for i in range(len(featureList1)):
        m = featureList1[i]+featureList2[i]
        merged.append(m)
    return merged


# In[33]:


#extract the sentiment labels by making positive reviews as class 1 and negative reviews as class 2
def get_lables(data):
    labels = []
    for tup in data:
        if tup[1].lower()=="negative":
            labels.append(-1)
        else:
            labels.append(1)
        
    return labels


# In[34]:


def calculate_precision(prediction, actual):
    prediction = list(prediction)
    correct_labels = [predictions[i]  for i in range(len(predictions)) if actual[i] == predictions[i]]
    precision = float(len(correct_labels))/float(len(prediction))
    return precision


# In[35]:


def real_time_test(classifier,vocab):
    print("Enter a sentence: ")
    inp = input()
    print(inp)
    feat_vec_uni = get_unigram_features(inp,vocab)
    feat_vec_swn =get_senti_wordnet_features(test_data)
    feat_vec = merge_features(feat_vec_uni, feat_vec_swn)

    predict = classifier.predict(feat_vec)
    if predict[0]==1:
        print("The sentiment expressed is: positive")
    else:
        print("The sentiment expressed is: negative")   


# In[36]:


training_unigram_features = get_unigram_features(training_data,vocabulary) # vocabulary extracted in the beginning
training_swn_features = get_senti_wordnet_features(training_data)
training_features = merge_features(training_unigram_features,training_swn_features)
training_labels = get_lables(training_data)

test_unigram_features = get_unigram_features(test_data,vocabulary)
test_swn_features=get_senti_wordnet_features(test_data)
test_features= merge_features(test_unigram_features,test_swn_features)

test_gold_labels = get_lables(test_data)


# In[37]:


# Naive Bayes Classifier 
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB().fit(training_features,training_labels) #training process
predictions = nb_classifier.predict(test_features)

print("Precision of NB classifier is")
predictions = nb_classifier.predict(training_features)
precision = calculate_precision(predictions,training_labels)
print("Training data\t" + str(precision))
predictions = nb_classifier.predict(test_features)
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))


# In[38]:


# SVM Classifier
#Refer to : http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC(penalty='l2', C=0.01).fit(training_features,training_labels)
predictions = svm_classifier.predict(training_features)

print("Precision of linear SVM classifier is:")
precision = calculate_precision(predictions,training_labels)
print("Training data\t" + str(precision))
predictions = svm_classifier.predict(test_features)
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))


# In[ ]:




