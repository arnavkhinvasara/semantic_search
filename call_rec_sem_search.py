import glob
import os
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
import numpy as np
 
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))

print("\n\n AUDIO FILES MUST BE .WAV files! Please give the program a second to load. Model is being loaded.\n\n")

#Converting audio files to transcripts (until line 31)

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Obtain ONLY WAV files because only WAV files work with speech_recognition library
list_of_recordings = glob.glob(os.path.join(current_directory, '*.wav'))

#Get transcripts for all the audio files
r = sr.Recognizer()
transcripts_list = []
for recording in list_of_recordings:
	with sr.AudioFile(recording) as source:
		audio_data = r.record(source)
		transcript = r.recognize_google(audio_data)
		transcripts_list.append(transcript)

#Preprocessing the transcripts
def transcript_preprocesser(transcripts_list):
	#1. Convert all text to lowercase
	transcripts_list = [t.lower() for t in transcripts_list]

	#2. Remove stop words
	new_transcripts_list = []
	for i in range(len(transcripts_list)):
		t = transcripts_list[i]
		word_tokens = word_tokenize(t)
		filtered_word_list = [word for word in word_tokens if word not in stop_words]
		new_transcripts_list.append(filtered_word_list) #this tokenizes the transcript by word, which is step 3
	return new_transcripts_list

processed_transcripts_list = transcript_preprocesser(transcripts_list)
"""
extended_transcripts_list = []
for element_list in processed_transcripts_list:
	extended_transcripts_list.extend(element_list)
model = gensim.models.Word2Vec(extended_transcripts_list, min_count=1, vector_size=100, window=5)
"""
model = KeyedVectors.load("word2vec.model")

#Returns a vector embedding for any list of words
def obtain_embedding(some_list_of_words):
	vector_embeddings = [model[a_word] for a_word in some_list_of_words if a_word in model]
	#average embeddings for each word to get one embedding for entire transcript
	embedding = np.mean(vector_embeddings, axis=0)
	return embedding

#Obtain embeddings (vectors in semantic space) for each transcript
transcript_embeddings = [obtain_embedding(a_transcript) for a_transcript in processed_transcripts_list]

while True:
	#Obtain query from user
	query = input(" \n Enter a phrase you'd like to semantically search: \n")
	query_list = query.split(" ")

	#Preprocess query: Convert all text to lowercase and remove stop words in one step
	query_list = [word.lower() for word in query_list if word.lower() not in stop_words]

	#Gets embedding for query
	query_embedding = obtain_embedding(query_list)

	#Returns Cauchy Schwarz Ineqaulity cosine similarity between any 2 vectors
	def cosine_similarity(v1, v2):
		return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

	#Gets list of cosine similarity between query embedding and each transcript embedding
	cosine_similarities = [cosine_similarity(query_embedding, transcript_embedding) for transcript_embedding in transcript_embeddings]

	print(f'Cosine similarities for files in the order they appear in the directory: {cosine_similarities}')
	#Index of transcript with maximum similarity
	index_of_max = cosine_similarities.index(max(cosine_similarities))

	#return file_name associated with max similarity
	print(f'The most similar file to your input is: {os.path.basename(list_of_recordings[index_of_max])}')




