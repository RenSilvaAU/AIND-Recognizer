import warnings
from asl_data import SinglesData


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

    print('\n\n*** one set')
    print(' length of all sequences: {} '.format(len(test_set.get_all_sequences())))
    print(' length of all Xlenghts: {} '.format(len(test_set.get_all_Xlengths())))
    print(' length of all Word list: {} '.format(len(test_set.wordlist)))

    all_sequences = test_set.get_all_sequences()
    all_Xlenghts = test_set.get_all_Xlengths()

    for i, test_word in zip( range(0,len(all_sequences) -1 ), test_set.wordlist):
        # try to get the word based on GaussianHMM model
        # print('test word is: {} '.format(test_word))   
    
    
        bestLogL = float("-inf")
        bestWord = ''

        myProbs = []
        
        for word in models.keys():

            model = models[word]

            #print('My Sequence {} '.format(all_sequences[i][0]))

            # print('My X Lenght {} '.format(all_Xlenghts[i][1]))

            try: 
              logL = model.score(all_sequences[i][0],all_Xlenghts[i][1] )

              # print('Calculated {} '.format(logL))

              myProbs.append([{ word : logL }])
              if logL > bestLogL:
                  bestLogL = logL
                  bestWord = word

            except Exception:
              pass
 
        # print('Best guess for {} is {} with logL: {} '.format(test_word,bestWord,bestLogL))

        guesses.append([bestWord])
        probabilities.append([myProbs])
        
    '''
    for i in range(0, len(guesses) - 1):
      print('guesses: {} '.format(guesses[i]))
      print('probabilities: {} '.format(probabilities[i]))
    '''
