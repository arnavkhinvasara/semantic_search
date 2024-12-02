# semantic_search
Semantic Search in call recordings for Bland.AI

My process: Firstly, the model can only take WAV files due to speech_recognition library only accepting WAV files. I initially tried to train the model on just the transcripts of the recordings I had, but I did not have very many transcripts so the model wouldn't train as expected. I do not know how many recordings and transcripts Bland has so I am not sure if the model will train properly with your recordings or not. So for right now I trained the model on general words in the English language, and the model is fairly accurate. However, if you would like to see if the model trains accurately based on the call recordings you have instead of just general words, please do the following steps:
1. Uncomment lines 51-54, and comment out line 56
2. Uncomment line 9 and comment out line 8
3. In line 60, "vector_embeddings = ..." change "a_word in model" to "a_word in model.wv.index_to_key"

OVERALL TO TEST THE PROGRAM:
1. Clone the repository
2. Please import your audio files (IN .WAV FORMAT) to the same directory as the python program.
3. Run the python program with "python call_rec_sem_search.py"
