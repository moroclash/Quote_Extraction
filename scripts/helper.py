
import math
import numpy as np
# Library to import pre-trained model for sentence embeddings
from sentence_transformers import SentenceTransformer
# Calculate similarities between sentences
from sklearn.metrics.pairwise import cosine_similarity
# package for finding local minimas
from scipy.signal import argrelextrema
import spacy



class docTokenizer:

    def __init__(self) -> None:
        # Loading a model - don't try it at home, it might take some time - it is 420 mb
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')

    def rev_sigmoid(self, x:float)->float:
        return (1 / (1 + math.exp(0.5*x)))

    def activate_similarities(self, similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum 
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(self.rev_sigmoid) 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weghted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities


    def process(self, text):
        self.sentences = [str(i) for i in self.nlp(text).sents]

        # Get the length of each sentence
        sentece_length = [len(each.split(" ")) for each in self.sentences]
        # Determine longest outlier
        long = abs(np.mean(sentece_length) + np.std(sentece_length)) *2
        # Determine shortest outlier
        short = abs(np.mean(sentece_length) - np.std(sentece_length)) *2


        # Shorten long sentences
        text = ''
        for each in self.sentences:
            if len(each.split(" ")) > long:
                # let's replace all the commas with dots
                text+= each.replace(',', '.')
            else:
                text+= f'{each}###'

        self.sentences = text.split('###')
    
        # Now let's concatenate short ones
        text = ''
        for each in self.sentences:
            if len(each.split(" ")) < short:
                text+= f'{each}'
            else:
                text+= f'{each}###'

        self.sentences = text.split('###')

        # Embed sentences
        embeddings = self.model.encode(self.sentences)

        # Create similarities matrix
        similarities = cosine_similarity(embeddings)

        # Lets apply our function. For long sentences i reccomend to use 10 or more sentences
        activated_similarities = self.activate_similarities(similarities, p_size=7)

        ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
        #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
        minmimas = argrelextrema(activated_similarities, np.less, order=2) 

        split_points = [each for each in minmimas[0]]
        # Create empty string
        text = ''
        
        for num, each in enumerate(self.sentences):
            # Check if sentence is a minima (splitting point)
            if num in split_points:
                # If it is than add a dot to the end of the sentence and a paragraph before it.
                text+=f'####{each}'
            else:
                # If it is a normal sentence just add a dot to the end and keep adding sentences.
                text+=f'{each}'
      
        return text.split("####")