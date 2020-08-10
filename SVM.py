import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import joblib
data = pd.read_csv('Airline.csv',encoding='utf-8')
data = data[['airline_sentiment','text']]

def convert(polarity):
	if polarity == 'positive':
		return 1
	elif polarity == 'neutral':
		return 0
	else:
		return -1

data['polarity'] = data['airline_sentiment'].apply(convert)

X = data.drop('airline_sentiment',axis=1)
y = data['polarity']

# bow_transformer = CountVectorizer().fit(data['text'])
# messages_bow = bow_transformer.transform(data['text'])
# tfidf_transformer = TfidfTransformer().fit(messages_bow)
# messages_tfidf = tfidf_transformer.transform(messages_bow)

X_train,X_test,y_train,y_test = train_test_split(data['text'],data['polarity'],test_size=0.3,random_state=101)

pipeline = Pipeline([
	('bow',CountVectorizer()),
	('tfidf',TfidfTransformer()),
	('classifier',SVC())
	])

print(X_test)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)

print('classification_report for SVM')
print(classification_report(y_test,predictions))

_ = joblib.dump(pipeline, "svm.pkl", compress=9)

report_path = "report.txt"

text_file = open(report_path, "w")
n = text_file.write(classification_report(y_test,predictions))
text_file.close()