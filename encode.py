import numpy as np
from operator import itemgetter
import preprocess
from scipy import sparse

class my_encoding():

	def __init__(self):
		self.max_number_of_words_per_document = 128
		self.max_number_of_coordinates_per_word = 256



	def fit(self,X):
		"""
		param X: for now assumed to be a list of document, each document being a string
		returns: a list containing the matrix representation of each document, as described in the paper
		"""


		#first we preprocess the data
		X_new = []
		for document in X:
			document = preprocess.preprocessDocument(document)
			X_new.append(document)


		#first we rank of each character in the corpus
		ranked_characters = self._getRankOfCharacters(X_new)
		print("Got rank")
		#now get mapping from character to compress code
		mapping = self._getMappingCharToCompressCode(ranked_characters)

		
		#now we can build the representation of each document
		representations = []
		for document in X_new:
			document_representation = []
			words = document.split()
			for word in words:
				word_representation = np.array([])
				for char in word:
					word_representation = np.concatenate((word_representation,mapping[char]))
					if len(word_representation) > self.max_number_of_coordinates_per_word:
						break
				word_representation = self._fixSizeOfWordRepresentation(word_representation)
				document_representation.append(word_representation)
				if len(document_representation)>self.max_number_of_words_per_document:
					break
			document_representation = self._fixSizeOfDocumentRepresentation(document_representation)
			document_representation = np.asarray(document_representation)

			##Addition###
			document_representation = sparse.csr_matrix(document_representation)
			representations.append(document_representation)

		return representations


	def _getRankOfCharacters(self,X):
		
		#first we get count of each character
		count = {}
		for document in X:
				for charac in document:
					if charac in count:
						count[charac] += 1
					else:
						count[charac] = 1


		list_count = []
		for charac in count:
			list_count.append((charac,count[charac]))

		#sort in decreasing order
		list_count.sort(reverse=True, key=itemgetter(1))

		ranked_characters = []
		for i in range(len(list_count)):
			ranked_characters.append(list_count[i][0])

		return ranked_characters


	def _getMappingCharToCompressCode(self, ranked_characters):
		"""
		param ranked_characters: list of characters in decreasing order of frequency in the corpus
		returns: a dictionnary mapping each character to its compress code
		"""

		mapping = {}
		for rank in range(len(ranked_characters)):
			mapping[ranked_characters[rank]] = self._createCompressCode(rank)
		return mapping


	def _createCompressCode(self,rank):
		"""
		param rank: rank of the character in the corpus
		returns: compress code corresponding to that character, as a numpy array
		"""

		compress_code = [1]
		for i in range(rank):
			compress_code.append(0)
		compress_code.append(1)
		return np.asarray(compress_code)




	def _fixSizeOfWordRepresentation(self,word_representation):
		"""
		param word_representation: numpy array representing the code of a word, either too long or too short
		returns: the word representation with the good size
		"""
		if len(word_representation) > self.max_number_of_coordinates_per_word:
			word_representation = word_representation[:self.max_number_of_coordinates_per_word]
			return word_representation
		if len(word_representation) < self.max_number_of_coordinates_per_word:
			word_representation = np.concatenate((word_representation,np.zeros(self.max_number_of_coordinates_per_word - len(word_representation))))
		return word_representation


	def _fixSizeOfDocumentRepresentation(self,document_representation):
		if len(document_representation) >= self.max_number_of_words_per_document:
			document_representation = document_representation[:self.max_number_of_words_per_document]
			return document_representation
		else:
			while len(document_representation) < self.max_number_of_words_per_document:
				zero_line = np.zeros(self.max_number_of_coordinates_per_word)
				document_representation = np.concatenate((document_representation,[zero_line]))
			return document_representation

