

def preprocessDocument(doc):
	"""
	param doc: a string
	returns: preprocesse string
	"""
	#first apply lowercasing
	doc = doc.lower()

	#next we replace '\' with ' '
	doc = doc.replace('\\',' ')

	return doc