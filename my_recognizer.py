import warnings
from asl_data import SinglesData
import logging

##logger basic config

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(name)s:  %(message)s')



def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for i in range(test_set.num_items):
        x, length = test_set.get_item_Xlengths(i)
        
        temp_prob = {}
        temp_guess = None
        temp_guess_score = float('-inf')
        
        for word in models:
            try:
                model = models[word]
                temp_score = model.score(x, length)
                temp_prob[word] = temp_score
                if temp_score > temp_guess_score:
                    temp_guess_score = temp_score
                    temp_guess = word
            except Exception as er:
                logger.info('Recognize: Error word={} error={}'.format(word,er))
                temp_prob[word] = None
        
        probabilities.append(temp_prob)
        guesses.append(temp_guess)
    
    return probabilities, guesses
