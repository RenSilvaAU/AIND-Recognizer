import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        split_method = KFold()

        # print('Number of sequences: {} '.format(len(self.sequences)))

        best_model = None
        best_num_components = self.min_n_components
        best_bic = float('-inf')

        if len(self.sequences) > 2:

            # KFold splie method will only for for sequences of 2 or more samples

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  

                # determine training set
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)

                # determine cross validation set
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)


                for num_states in range(self.min_n_components,self.max_n_components):

                    try:
                        # train model with training set
                        hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(X_train, lengths_train)
                        logL = hmm_model.score(X_test, lengths_test)
                        N = hmm_model.n_features

                        # now calculate bic
                        bic = -2 * logL + num_states * np.log(N)

                        if bic > best_bic:

                            # new set of best numbers
                            best_num_components, best_bic, best_model = num_states, bic, hmm_model
                            # print('Selected LogL {} , Num of steaes {} '.format(best_logL, best_num_components))
                    except Exception:
                        # if it fails, it will try again with the next set of elements, or simply return an empty model
                        pass

        elif len(self.sequences) == 2:

            # determine training set ... first element
            X_train, lengths_train = combine_sequences([0], self.sequences)

            # determine cross validation set .... second element
            X_test, lengths_test = combine_sequences([1], self.sequences)


            for num_states in range(self.min_n_components,self.max_n_components):

                try:
                    # train model with training set
                    hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(X_train, lengths_train)
                    logL = hmm_model.score(X_test, lengths_test)
                    N = hmm_model.n_features

                    # now calculate bic
                    bic = -2 * logL + num_states * np.log(N)

                    if bic > best_bic:

                        # new set of best numbers
                        best_num_components, best_bic, best_model = num_states, bic, hmm_model
  
                except Exception:
                    # if it fails, will simply return a empty best model
                    pass

        # print('Selected {} '.format(best_num_components))
        return best_model    


        '''    
        best_model = None
        best_num_components = self.min_n_components
        best_bic = float('-inf')

        for num_states in range(self.min_n_components,self.max_n_components):

            try:
                # train model with training set
                hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
                N = hmm_model.n_features

                # now calculate bic
                bic = -2 * logL + num_states * np.log(N)

                if bic > best_bic:
                    # new set of best numbers
                    best_num_components, best_bic, best_model = num_states, bic, hmm_model
                    # print('Selected LogL {} , Num of steaes {} '.format(best_logL, best_num_components))
            except Exception:
                pass

        
        return best_model
        '''

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
        best_model = None
        best_num_components = self.min_n_components
        best_dic = float('-inf')

        for num_states in range(self.min_n_components,self.max_n_components):

            try:
                # train model with training set
                hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
                N = hmm_model.n_features

                # now calculate bic
                dic = -2 * logL + num_states * np.log(N)

                if dic > best_dic:
                    # new set of best numbers
                    best_num_components, best_dic, best_model = num_states, dic, hmm_model
                    # print('Selected LogL {} , Num of steaes {} '.format(best_logL, best_num_components))
            except Exception:
                pass

        
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError

        # print('Called Selector CV for {} '.format(self.this_word))

        split_method = KFold()

        # print('Number of sequences: {} '.format(len(self.sequences)))

        best_model = None
        best_num_components = self.min_n_components
        best_logL = float('-inf')

        if len(self.sequences) > 2:

            # KFold splie method will only for for sequences of 2 or more samples

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  

                # determine training set
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)

                # determine cross validation set
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)


                for num_states in range(self.min_n_components,self.max_n_components):

                    try:
                        # train model with training set
                        hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(X_train, lengths_train)
                        logL = hmm_model.score(X_test, lengths_test)

                        if logL > best_logL:

                            # new set of best numbers
                            best_num_components, best_logL, best_model = num_states, logL, hmm_model
                            # print('Selected LogL {} , Num of steaes {} '.format(best_logL, best_num_components))
                    except Exception:
                        # if it fails, it will try again with the next set of elements, or simply return an empty model
                        pass

        elif len(self.sequences) == 2:

            # determine training set ... first element
            X_train, lengths_train = combine_sequences([0], self.sequences)

            # determine cross validation set .... second element
            X_test, lengths_test = combine_sequences([1], self.sequences)


            for num_states in range(self.min_n_components,self.max_n_components):

                try:
                    # train model with training set
                    hmm_model = GaussianHMM(n_components=num_states, n_iter=1000).fit(X_train, lengths_train)
                    logL = hmm_model.score(X_test, lengths_test)

                    if logL > best_logL:

                        # new set of best numbers
                        best_num_components, best_logL, best_model = num_states, logL, hmm_model
                    # print('Selected LogL {} , Num of steaes {} '.format(best_logL, best_num_components))
                except Exception:
                    # if it fails, will simply return a empty best model
                    pass

        # print('Selected {} '.format(best_num_components))
        return best_model      

            