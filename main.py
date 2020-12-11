# Tom Jarman, 10/12/2020
import pandas as pd
import numpy as np
import string

# Preprocessing Constants
MAP_SENTIMENT = False
LOWERCASE = True
NOISE_REMOVAL = True

# Baysian Classifier Constants
ALL_WORDS = True

# Preprocessing Constants
PUNCTUATION = string.punctuation.replace('!','')

def MapSentimentScore(score):
    """ 
    Maps input score value from 5 value system to 3 value system:
        0 or 1 -> 0
        2      -> 1
        3 or 4 -> 2 
    """
    return_value = score

    if score == 1:
        return_value = 0

    elif score == 2:
        return_value = 1

    elif score == 3 or score == 4:
        return_value = 2
    
    return return_value


def RemoveNoise(phrase):
    """
    Removes punctuation and replaces double spaces with a singular space
    """
    return phrase.translate(str.maketrans('', '', PUNCTUATION)).replace('  ', ' ')


def Preprocessing(structure):
    """
    Applies various preprocessing methods depending on CONSTANTS set above
    """
    ids = structure["SentenceId"]
    phrases = structure["Phrase"]
    sentiments = structure["Sentiment"]

    if MAP_SENTIMENT:
        sentiments = sentiments.apply(MapSentimentScore)

    if LOWERCASE:
        phrases = phrases.apply(str.lower)

    if NOISE_REMOVAL:
        phrases = phrases.apply(RemoveNoise)

    data = {"SentenceId": ids,
            "Phrase": phrases,
            "Sentiment": sentiments}

    return pd.concat(data, axis=1)
    

def ComputePriorProbabilities(structure):
    frequency = structure['Sentiment'].value_counts()
    return frequency / structure.shape[0]


def ComputeLikelihoods(structure):

    # Spliting phrase column into individual words
    split_phrases = structure.assign(Phrase=structure.Phrase.str.split(' ')).explode('Phrase')

    # Removing rows with empty cell in phrase column
    split_phrases['Phrase'].replace('', np.nan, inplace=True)
    split_phrases.dropna(subset=['Phrase'], inplace=True)

    # Counts occurrences of each word in each sentiment class
    likelihoods = split_phrases.groupby(['Phrase','Sentiment']).size().reset_index(name='Counts')
    sentiment_counts = split_phrases['Sentiment'].value_counts()
    likelihoods['Total Sentiment Terms'] = likelihoods['Sentiment'].map(sentiment_counts)
    likelihoods['Likelihood'] = likelihoods['Counts'] / likelihoods['Total Sentiment Terms']

    return likelihoods


if __name__ == "__main__":
    train_data = pd.read_csv("train.tsv", sep='\t')
    preprocessed_train_data = Preprocessing(train_data)
    priors = ComputePriorProbabilities(preprocessed_train_data)

    dev_data = pd.read_csv("dev.tsv",sep='\t')
    preprocessed_dev_data = Preprocessing(dev_data)
    likelihoods = ComputeLikelihoods(preprocessed_dev_data)
    

