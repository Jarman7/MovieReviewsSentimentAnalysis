# Tom Jarmran, 10/12/2020

import pandas as pd
import string

MAP_SENTIMENT = False
LOWERCASE = True
NOISE_REMOVAL = True

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


if __name__ == "__main__":
    data_structure = pd.read_csv("dev.tsv", sep='\t')
    preprocessed_data = Preprocessing(data_structure)