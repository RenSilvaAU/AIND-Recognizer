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

    all_sequences = test_set.get_all_sequences()
    all_Xlenghts = test_set.get_all_Xlengths()

    print('Started recognizing ...')

    for i, test_word in zip( range(0,len(all_sequences)  ), test_set.wordlist):

        bestLogL = float("-inf")
        bestWord = ''

        myProbs = {}
        
        for word in models.keys():

            model = models[word]

            try: 

              logL = model.score(all_sequences[i][0],all_Xlenghts[i][1] )

              if logL > bestLogL:
                  bestLogL = logL
                  bestWord = word

            except Exception:
              pass
              
            myProbs[word] = logL 
    
 
        guesses.append(bestWord)
        probabilities.append(myProbs)
        
    print('Finished analyzing {} words '.format(len(all_sequences)))

    return probabilities, guesses

def show_errors(guesses, test_set):

  no_of_correct = 0
  no_of_words = 0

  header_printed = False

  print('\n\nHey Ren, this is how good you are ....')

  for guess, test in zip(guesses, test_set.wordlist):

    no_of_words += 1

    if (guess == test):
      no_of_correct += 1
    else:
      if not header_printed:
        
        print('\n*** These were the errors I found:')
        header_printed = True

      print('{} != {}'.format(guess,test))


  print('===========================================')

  wer = round( (no_of_words - no_of_correct) / no_of_words * 100,2)

  comment = ''

  if wer < 20:
    comment = "Bloddy ledgend!"
  elif wer < 40:
    comment = "You rock"
  elif wer < 60:
    comment = "You passed ... just"
  else:
    comment = "You suck!"

  print('Your WER: {}%  You got {} out of {} right. Guess what? {} '.format(wer, no_of_correct,no_of_words,comment))