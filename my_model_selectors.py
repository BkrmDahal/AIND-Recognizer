import math
import statistics
import warnings
import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

#logger basic basicConfig
logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(name)s:  %(message)s')



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information DIC: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        hmm_model = None
        BIC = float('inf')

        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_hmm_model = self.base_model(i)
                temp_logL = temp_hmm_model.score(self.X, self.lengths)
                num_feature = self.X[0]
                num_param = 2*i*num_feature + i*i
                temp_BIC = (-2)*temp_logL + len(num_param)*math.log(len(self.X))
                if temp_BIC < BIC:
                    BIC = temp_BIC
                    logging.info("SelectorBIC: model created for {} with {} states".format(self.this_word, i))
                    hmm_model = temp_hmm_model
            except Exception as er:
                logging.info("SelectorBIC: failure on {} with {} error".format(self.this_word, er))
                pass

        if self.verbose and hmm_model == None:
            logging.info("SelectorBIC: failure on {} with {} states".format(self.this_word, i))

        return hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        hmm_model = None
        DIC = float('-inf')

        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_hmm_model = self.base_model(i)
                temp_logL = temp_hmm_model.score(self.X, self.lengths)
                temp_logL2 = 0

                for word2 in self.hwords:
                    try:
                        if word2 == self.this_word:
                            continue
                        x2, l2 = self.hwords[word2]
                        temp_logL2_t = temp_hmm_model.score(x2, l2)
                        temp_logL2 += temp_logL2_t

                    except Exception as er:
                        logging.info("SelectorDIC: failure on {} with {} states, and word2: {}".format(self.this_word, i, word2))
                        pass
                
                temp_DIC = temp_logL - temp_logL2 / (len(self.hwords)-1)

                if temp_DIC > DIC:
                    DIC = temp_DIC
                    hmm_model = temp_hmm_model
            except Exception as er:
                logging.info("SelectorDIC: failure on {} with {} states, and error: {}".format(self.this_word, i, er))
                pass
            
        if self.verbose and hmm_model == None:
            logging.info("SelectorDIC: failure on {} with {} states".format(self.this_word, i))
        
        return hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        n_components = None
        logL = float('-inf')
        
        try:
            split_method = KFold(n_splits=min(3,len(self.sequences)))
        except:
            return None

        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_logL = 0
                for j, k in split_method.split(self.sequences):
                    train_x, train_length = combine_sequences(j, self.sequences)
                    test_x, test_length = combine_sequences(k, self.sequences)
                    temp_hmm_model = self.base_model(i)
                    temp_logL += temp_hmm_model.score(test_x, test_length)
                logging.info("SelectorCV: model created for {} with {} states and logl {}.".format(self.this_word, i, temp_logL))
                if temp_logL > logL:
                    logL = temp_logL
                    n_components = i
            except Exception as ex:
                logging.info("SelectorCV: failure on {} with {} states, and error: {}".format(self.this_word, i, ex))
                pass
        
        if self.verbose and n_components == None:
            logging.info("SelectorCV: failure on {} with {} states, and error: {}".format(self.this_word, i, ex))
            return None
        
        try:
            return self.base_model(n_components)
        except:
            return None
