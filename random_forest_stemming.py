import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

# stemming:
# go, goes, going = go, but NOT went - requires lemmatization
# eat, eats, eaten = eat, but NOT eaten - requires lemmatization 
# snowball is a language widly used to implement stemming algorithms
stemmer = SnowballStemmer('english')

df_train = pd.read_csv('input\\train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('input\\test.csv', encoding="ISO-8859-1")

df_pro_desc = pd.read_csv('input\\product_descriptions.csv')

num_train = df_train.shape[0] # number of inputs in training set

# take a sentence/string, stem all words within it
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

# return the number of common words in two sentences/strings
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

# vertically combine training set and testing set
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# inner join train-test set with "product description" on key "product_uid"
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

# create additional columns/features for random forest classification
#
# 1 - perform stemming on features "search_term", "product_title" and "product_description"
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

#
# 2 - calculate length of query sent by user
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

#
# 3 - create "product_info" feature that's a tab delimited
# concatination of stemmed "search_term", "product_title" and "product_description"
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

#
# 4 - count the common words between stemmed "search_term" and "product_title"
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
#
# 5 - count the common words between stemmed "search_term" and "product_description"
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

#
# 6 - drop features not to be used in classifier
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

#
# 7 - split dataset back to train/test sets
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

#
# 8 - split predictor/output and features in train/test sets
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#
# 9 - train random forest model and use it for prediction
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('output\\submission.csv',index=False)