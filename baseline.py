from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def baseline_model(X_train,y_train,X_test,y_test, ngrams = False, tfidf = False, verbose=True):
	"""
	param X = list of documents, represented as strings
	param y: list of labels
	param ngrams: set to True if you want the n-gram version of the baseline model, will be 5-gram model
	param tfidf: wether you want tfidf or not
	returns: accuracy on test set
	"""


	#first preprocess the input, namely vectorize it
	vectorizer = None

	if ngrams:
		n_range = 5
		max_features = 500000
		if tfidf:
			vectorizer = TfidfVectorizer(ngram_range=(1,n_range),max_features=max_features)
		else:
			vectorizer = CountVectorizer(ngram_range=(1,n_range),max_features=max_features)
	else:
		n_range = 1
		max_features = 50000
		if tfidf:
			vectorizer = TfidfVectorizer(ngram_range=(1,n_range),max_features=max_features)
		else:
			vectorizer = CountVectorizer(ngram_range=(1,n_range),max_features=max_features)


	#fit the data
	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)

	"""now we are ready to train"""
	#first prepare the model
	model = LogisticRegression() #no details provided on hyperparameters
	if verbose:
		print("Starting training...")
	model.fit(X_train,y_train)

	if verbose:
		print("Training done. Starting predictions...")
	acc = model.score(X_test,y_test)

	if verbose:
		print("Accuracy on test set:" + str(acc))
	return acc