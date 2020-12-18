# Tom Jarman, 10/12/2020
# Stop list from https://gist.github.com/sebleier/554280 Accessed: 18/12/2020
# Adjective list from https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913 Accessed: 18/12/2020
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string

# Constants
MAP_SENTIMENT     = False
LOWERCASE         = True
NOISE_REMOVAL     = False
LAPLACE_SMOOTHING = True
STOPWORD_LIST     = False
ADJECTIVES_ONLY   = False
PUNCTUATION = string.punctuation

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


def Preprocessing(structure, split=True, sentiment=True):
    """
    Applies various preprocessing methods depending on CONSTANTS set above
    """
    # Takes ids and phrases from initial structure
    ids = structure["SentenceId"]
    phrases = structure["Phrase"]

    # If LOWERCASE constant set then make all phrases lowercase
    if LOWERCASE:
        phrases = phrases.apply(str.lower)

    # If NOISE_REMOVAL is set then remove punctuation
    if NOISE_REMOVAL:
        phrases = phrases.apply(RemoveNoise)

    data = {"SentenceId": ids, "Phrase": phrases}

    # If sentiment is true then include the sentiment values in the final structure
    if sentiment:
        sentiments = structure["Sentiment"]
    
        # Is MAP_SENTIMENT is set then reduce sentiment score from 5 to 3 values
        if MAP_SENTIMENT:
            sentiments = sentiments.apply(MapSentimentScore)
    
        data['Sentiment'] = sentiments

    # Recombine processed data
    cleaned_data = pd.concat(data, axis=1)

    # If STOPWORD_LIST is set only use words in stop_list.txt
    if STOPWORD_LIST:
        stop_list = pd.read_csv('stop_list.txt', sep=" ", header=None)[0].tolist()
        #cleaned_data = cleaned_data[~cleaned_data['Phrase'].isin(stop_list[0].tolist())]
        cleaned_data['Phrase'] = [' '.join([item for item in x.split() 
                                if item not in stop_list]) 
                                for x in cleaned_data['Phrase']]

    # If ADJEVTIVES_ONLY is set only use words in english-adjectives.txt
    if ADJECTIVES_ONLY:
        adjective_list = pd.read_csv('english-adjectives.txt', sep=" ", header=None)[0].tolist()
        #cleaned_data = cleaned_data[cleaned_data['Phrase'].isin(adjective_list[0].tolist())]
        cleaned_data['Phrase'] = [' '.join([item for item in x.split() 
                                if item not in adjective_list]) 
                                for x in cleaned_data['Phrase']]


    if split:
        # Spliting phrase column into individual words
        split_phrases = cleaned_data.assign(Phrase=cleaned_data.Phrase.str.split(' ')).explode('Phrase')

        # Removing rows with empty cell in phrase column
        split_phrases['Phrase'].replace('', np.nan, inplace=True)
        split_phrases.dropna(subset=['Phrase'], inplace=True)
        cleaned_data = split_phrases

        

    return cleaned_data
    

def ComputePriorProbabilities(structure):
    """
    Computes prior probabilities of passed "structure"
    """
    # Count occurences of each sentiment class, order the results by class number, divide total count to get priors
    frequency_series = structure['Sentiment'].value_counts().sort_index() / structure.shape[0]
    return frequency_series


def ComputeLikelihoods(data_set):
    """
    Computes likelihoods of each term in "data_set"
    """
    # Counts occurrences of each word in each sentiment class
    likelihoods = data_set.groupby(['Phrase','Sentiment']).size().reset_index(name='Counts')
    sentiment_counts = data_set['Sentiment'].value_counts()
    likelihoods['Total Sentiment Terms'] = likelihoods['Sentiment'].map(sentiment_counts)

    # Calculates number of unique phrases
    num_terms = data_set['Phrase'].nunique()

    # Duplicating column and changing name
    likelihoods['Likelihood'] = likelihoods['Counts']

    # Trimming columns, pivoting sentiment values and filling NaN with 0
    # Calculates denominator for likelihood formula for each class
    # With Laplase smoothing
    if LAPLACE_SMOOTHING: 
        class_denominator = (1 / (sentiment_counts + num_terms)).sort_index()
        likelihoods = likelihoods[['Phrase', 'Sentiment', 'Likelihood']].pivot_table('Likelihood',['Phrase'],'Sentiment').fillna(0) + 1

    # Without Lapslase smoothing
    else:
        class_denominator = (1 / (sentiment_counts)).sort_index()
        likelihoods = likelihoods[['Phrase', 'Sentiment', 'Likelihood']].pivot_table('Likelihood',['Phrase'],'Sentiment').fillna(0)

    # Applies final stage of Calculating Likelihoods
    for i in sentiment_counts.index:
        likelihoods[i] = likelihoods[i] * class_denominator[i]

    return likelihoods


def Classify(data_set, priors, likelihoods):
    """
    Classifies "data_set" with senitment score according to "priors" and "likelihoods"
    """
    # Depending on sentiment score mapping determine column names
    columns =  ['SentenceId', 'Phrase', 0, 1, 2] if MAP_SENTIMENT else ['SentenceId', 'Phrase', 0, 1, 2, 3, 4]
    # Times the counts by the likelihoods for each term in each sentence
    data_set = (data_set.merge(likelihoods, left_on='Phrase', right_on='Phrase')
                        .reindex(columns=columns)
                        .set_index('SentenceId')).groupby('SentenceId').prod()
    # Multiply each sentiment class by its corresponding prior
    for i in range(priors.shape[0]):
        data_set[i] *= priors[i]
    # Find the max value for each row
    classified_series = data_set.idxmax(axis=1)
    # Create datafram with 'SentenceId' and 'Sentiment' value
    classified_frame = pd.DataFrame({'SentenceId':classified_series.index, 'Sentiment':classified_series.values}).set_index('SentenceId')
    return classified_frame


def ComputeConfusionMatrix(predicted_series, actual_series):
    """
    Computes confusion matrix between 'predicted_series' and 'actual_series'
    """
    # Create a dictionaly of the two series
    data = {'actual': actual_series, 'predicted': predicted_series}
    data_frame = pd.DataFrame(data, columns=['actual', 'predicted'])
    # Compute confusion matrix of the two columns
    confusion_matrix = pd.crosstab(data_frame['actual'], data_frame['predicted'], rownames=['Actual'], colnames=['Predicted'])
    return confusion_matrix


def PrepDataAndConfusionMatrix(priors, likelihoods, filename):
    """
    Takes in test data set 'filename', prepares it, classifies it and returns the confusion matrix
    """
    # Loads data and preprocesses for classifying and classifies
    data = pd.read_csv(filename, sep='\t')
    preprocessed_classify_data = Preprocessing(data, sentiment=False)
    classified_results = Classify(preprocessed_classify_data, priors, likelihoods)

    # If sentiment needs to be mapped, do so on the unprocessed data
    if MAP_SENTIMENT:
        data['Sentiment'] = data['Sentiment'].apply(MapSentimentScore)

    # Remove unclassified sentences from the intial data set, due to training corpus being too small
    unclassified_sentences = list(set(data['SentenceId'].tolist()).difference(set(classified_results.index.tolist())))
    cleaned_data = data[~data['SentenceId'].isin(unclassified_sentences)]

    #Computes confusion matrix between 'classified_results' and 'cleaned_data'
    confusion_matrix = ComputeConfusionMatrix(classified_results['Sentiment'].tolist(), cleaned_data['Sentiment'].tolist())

    return confusion_matrix


if __name__ == "__main__":
    # Computes priors
    train_data = pd.read_csv("train.tsv", sep='\t')
    preprocessed_train_data = Preprocessing(train_data, split=False)
    priors = ComputePriorProbabilities(preprocessed_train_data)

    # Computes likelihoods
    train_data_likelihoods = pd.read_csv("train.tsv",sep='\t')
    preprocessed_train_data_likelihoods = Preprocessing(train_data_likelihoods)
    likelihoods = ComputeLikelihoods(preprocessed_train_data_likelihoods)

    matrix = PrepDataAndConfusionMatrix(priors, likelihoods, "dev.tsv")
    heat_map = sns.heatmap(matrix/np.concatenate(matrix).sum(), fmt='.2%', annot=True, cmap="YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Save results
    #data = pd.read_csv("test.tsv", sep='\t')
    #preprocessed_classify_data = Preprocessing(data, sentiment=False)
    #classified_results = Classify(preprocessed_classify_data, priors, likelihoods)
    #classified_results.to_csv("test_predictions_3classes_Tom_JARMAN_ALL.tsv", sep='\t')

    # Block used to generate test data and output percentage accuracy
    #for aw in ADJECTIVES_ONLYS:
    #    ADJECTIVES_ONLY = aw
    #    for sw in STOPWORD_LISTS:
    #        STOPWORD_LIST = sw
    #        for ls in LAPLACE_SMOOTHINGS:
    #            LAPLACE_SMOOTHING = ls
    #            for nr in NOISE_REMOVALS:
    #                NOISE_REMOVAL = nr
    #                for l in LOWERCASES:
    #                    LOWERCASE = l
    #                    for ms in MAP_SENTIMENTS:
    #                        MAP_SENTIMENT = ms
    #                        if not aw and not sw:
    #                            print("aw ", end="")
    #                        if aw:
    #                            print("adw ", end="")
    #                        if sw:
    #                            print("sw ", end="")
    #                        if ls:
    #                            print("ls ", end="")
    #                        if nr:
    #                            print("nr ", end="")
    #                        if l:
    #                            print("l ", end="")
    #                        if ms:
    #                            print("3: ", end="")
    #                        else:
    #                            print("5: ", end="")
    #                        train_data = pd.read_csv("train.tsv", sep='\t')
    #                        preprocessed_train_data = Preprocessing(train_data, split=False)
    #                        priors = ComputePriorProbabilities(preprocessed_train_data)

    #                        dev_data = pd.read_csv("train.tsv",sep='\t')
    #                        preprocessed_dev_data = Preprocessing(dev_data)
    #                        likelihoods = ComputeLikelihoods(preprocessed_dev_data)

    #                        matrix = PrepDataAndConfusionMatrix(priors, likelihoods, "dev.tsv")
    #                        print(sum(matrix[i][i] for i in range(matrix.shape[0]))/np.concatenate(matrix).sum())
