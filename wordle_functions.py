import numpy as np # for stats
import random # for randomly generating target and start words
import operator # for sorting letter frequency distribution
import time # for #dramaticeffect
import pandas as pd
from nltk.corpus import movie_reviews, treebank, brown, gutenberg, switchboard

english_alphabet = "abcdefghijklmnopqrstuvwxyz"

def get_letter_counts(letters: str, word_list: list, sort: str = "descending"):
    """
    Given a passed str of letters and a list of words, produces a frequency distribution of all letters
    
    ------
    Parameters:
    ------
    `letters`: str
        a string of letters to be counted. String must only be desired letters, with no spaces. Default is local variable containing all letters of the English alphabet
    `word_list`: list
        list of words (str) from which word frequencies will be counted
    `sort`: str
        if either "descending" or "ascending" are passed, returned list of tuples will be sorted accoringly, else returned dictionary will be unsorted

    ------
    Returns:
    ------
    `letters_counts_dict`: dict
        dictionary of {letter : count} pairs for each letter in passed `letters` sequence
    `sorted_counts_dicts`: list of tuples
        list of tuples. Format is ("letter", frequency). Ordered according to `sort` values
    """

    words_counts_dict = {}

    for word in word_list: # real dataset
        word_dict = {}
 
        for letter in word:
            if letter in word_dict:
                word_dict[letter] += 1
            else:
                word_dict[letter] = 1
        words_counts_dict[word] = word_dict

    letters_counts_dict = {}

    for letter in letters:
        letters_counts_dict[letter] = 0

    for word, count_dict in words_counts_dict.items():
        # print (word, count_dict)
        for letter, count in count_dict.items():
            letters_counts_dict[letter] += count

    if sort == "ascending":
        sorted_counts_dict = (sorted(letters_counts_dict.items(), key = operator.itemgetter(1), reverse = False))
        return sorted_counts_dicts

    if sort == "descending":
        sorted_counts_dict = sorted(letters_counts_dict.items(), key = operator.itemgetter(1), reverse = True)
        return sorted_counts_dict
    else:
        return letters_counts_dict
    
### Best first guesses for a given Wordle list

def best_guess_words(word_list: list, show_letters: bool = False):
    """
    Given a passed list of English words of a consistent length, calculates the most statistically optimal first guess words, alongside a rating for each word. 
    
    Rating = sum(frequency of each unique letter in that word) / sum (all unique letter frequencies in word_list) * 100, rounded to 2 decimals.

    ------
    Parameters:
    ------
    `word_list`: list
        list of words (str) of consistent length
    `show_letters`: bool
        if True, also prints set of most optimal letters to guess

    ------
    Returns:
    ------
    `word_ratings`: list
        list of tuples. Format is [(word, rating)], where rating is calculated according to above formula
    `sorted_counts`: list of tuples
        list of tuples. Format is ("letter", frequency). Sorted according to `sort` value; ["descending" or "ascending"] if passed
    """
        
    english_alphabet = "abcdefghijklmnopqrstuvwxyz"

    sorted_counts = get_letter_counts(english_alphabet, word_list, sort = "descending")

    max_len_possible = len(word_list[0])

    ### Get words with the highest letter diversity
    while max_len_possible:

        best_letters = set()
        best_words = []

        for letter, freq in sorted_counts:
            best_letters.add(letter)
            if len(best_letters) == max_len_possible:
                break

        ### Get all words that have one of each of the 5 top most frequent letters
        for word in word_list:
            word_set = set()

            for letter in word:
                word_set.add(letter)

            if best_letters.issubset(word_set):
                best_words.append(word)

        if len(best_words) > 0:
            break
        else:
            max_len_possible -= 1 # only try the top 4 letters, then 3, then 2, ...
        
        if max_len_possible == 0:
            break

    all_letters_count = 0
    for letter, freq in sorted_counts:
        all_letters_count += freq

    word_ratings = []
    for word in best_words:
        ratings_dict = {}
        for letter in word:
            for freq_letter, freq in sorted_counts:
                if letter == freq_letter:
                    ratings_dict[letter] = freq
        
        total_rating = 0
        for letter, rating in ratings_dict.items():
            total_rating += rating
        
        word_ratings.append((word, round(total_rating / all_letters_count * 100, 2)))

    word_ratings = sorted(word_ratings, key = operator.itemgetter(1), reverse = True)

    if show_letters == True:
        return word_ratings, sorted_counts
    else:
        return word_ratings
    
def count_vows_cons(word: str, y_vow = True):
    """
    Given a passed word, calculate the number of non-unique vowels and consonants in the word (duplicates counted more than once).
    
    ------
    Parameters:
    ------
    `word`: str
        a single passed word (str)
    `y_vow`: bool
        if True, "y" is considered a vowel. If False, "y" considered a consonant. Default is True

    ------
    Returns:
    ------
    `counts`: dict
        dictionary, where format is {letter type : count}
    """

    word = word.lower() # for consistency

    if y_vow == True:
        vows = "aeiouy"
        cons = "bcdfghjklmnpqrstvwxz"
    elif y_vow == False:
        vows = "aeiou"
        cons = "bcdfghjklmnpqrstvwxyz"

    counts = {}
    counts["vows"] = 0
    counts["cons"] = 0
    for letter in word:
        if letter in vows:
            counts["vows"] += 1
        if letter in cons:
            counts["cons"] += 1

    return counts

def get_word_entropy(words_to_rate: list, word_list: list, normalized: bool = True, ascending: bool = False):
    """
    Given a word and a word list, calculates entropy each word as a measure of its impact to the next possible guesses in Wordle, ordered according to `reverse` parameter.
    
    ------
    Parameters:
    ------
    `words_to_rate`: list
        list of strings to be rated
    `word_list`: list
        list of all possible words (str) of consistent length, to which each word in `words_to_rate` will be compared
    `normalized`: bool
        if True, normalizes all ratings on a scale of 0-100, with 100 being the rating for the most optimal word, and 0 for the least optimal word
    `ascending`: bool
        if True, returns list ordered ascending. If False, returns list in descending order

    ------
    Returns:
    ------
    `word_ratings`: list
        list of tuples. Format is [(word, rating)], where rating is calculated according to above formula
    `sorted_counts`: list of tuples
        list of tuples. Format is ("letter", frequency). Sorted according to `sort` value; ["descending" or "ascending"] if passed
    """

    if ascending == True:
        sorted_counts = get_letter_counts(english_alphabet, word_list, sort = "ascending")
    else:
        sorted_counts = get_letter_counts(english_alphabet, word_list, sort = "descending")

    all_letters_count = 0
    for letter, freq in sorted_counts:
        all_letters_count += freq

    unnormalized_ratings = []
    for word in words_to_rate:
        word = word.lower()
        ratings_dict = {}
        for letter in word:
            for freq_letter, freq in sorted_counts:
                if letter == freq_letter:
                    ratings_dict[letter] = freq
        
        total_rating = 0
        for letter, rating in ratings_dict.items():
            total_rating += rating

        unnormalized_ratings.append((word, round(total_rating / all_letters_count * 100, 2)))
    
    word_ratings = sorted(unnormalized_ratings, key = operator.itemgetter(1), reverse = True)
    # print (word_ratings)

    if normalized == True:
        if len(word_ratings) > 1:
            new_tests = []

            for tup in word_ratings:
                try:
                    normd = round(((tup[1] - word_ratings[-1][1]) / (word_ratings[0][1] - word_ratings[-1][1])) * 100, 2)
                    new_tests.append((tup[0], normd))
                except:
                    ZeroDivisionError
                    new_tests.append((tup[0], 0.0))    
                
            return new_tests
        else:
            return [(word_ratings[0][0], float(100))]
        
    elif normalized == False:

        return word_ratings
    
### Gets most common words of all words of the dataset

def get_word_distribution(word_list: list, sort: str = "descending"):
    """
    Given a passed str of words and a list of words, produces a frequency distribution of all words
    
    ------
    Parameters:
    ------
    `word_list`: list
        list of words (str) from which word frequencies will be counted
    `sort`: str
        if either "descending" or "ascending" are passed, returned list of tuples will be sorted accoringly, else returned dictionary will be unsorted

    ------
    Returns:
    ------
    `words_counts_dict`: dict
        dictionary of {word : count} pairs for each word in passed `word_list`
    `sorted_counts_dicts`: list of tuples
        list of tuples. Format is ("word", frequency). Ordered according to `sort` values
    """

    words_counts_dict = {}

    for word in word_list: 
        if word in words_counts_dict:
            words_counts_dict[word] += 1
        else:
            words_counts_dict[word] = 1

    if sort == "ascending":
        sorted_counts_dict = (sorted(words_counts_dict.items(), key = operator.itemgetter(1), reverse = False))
        return sorted_counts_dict

    if sort == "descending":
        sorted_counts_dict = sorted(words_counts_dict.items(), key = operator.itemgetter(1), reverse = True)
        return sorted_counts_dict
    
def wordle_wizard(word_list: list, max_guesses: int = None, 
                  guess: str = None, target: str = None, bias: bool = True, 
                  random_guess: bool = False, random_target: bool = False, 
                  verbose: bool = False, drama: float = None, 
                  return_stats: bool = False, record: bool = False):
    """
    Mimicking the popular web game, this function matches a current word to a target word automatically, in the most statistically optimal way possible.

    ------
    Parameters:
    ------
    `word_list`: list
        list of valid words to be considered
    `guess`: str
        a string -- must be the same length as `target_word`
    `target`: str
        a string -- must be the same length as `opening_word`
    `bias`: str ['entropy', 'common', 'rare', None]
        'entropy' biases next word guesses to be the ones with the highest impact on the range of next possible guesses. Entropy values associated with each word are normalized across the list.

        'common' biases next word guesses to be words that are more commonly used

        'rare' biases next word guesses to be words that are more rarely used

        'no_bias' chooses a next guess at random of all available guesses

    `max_guesses`: int
        the maximum number of attempts allowed to solve the Wordle
    `random_guess`: bool
        if True, randomly chooses a starting word from all words within `word_list`. If False, passed starting word must be used instead
    `random_target`: bool
        if True, randomly chooses a target word from all words within `word_list`. If False, passed target word must be used instead
    `verbose`: bool
        if True, prints progress and explanation of how function solves the puzzle. If False, prints only the guessed word at each guess.
    `drama`: float or int
        if int provided, each guess' output is delayed by that number of seconds, else each output is shown as quickly as possible. For ~dRaMaTiC eFfEcT~
    `return_stats`: bool
        if True, prints nothing and returns a dictionary of various statistics about the function's performance trying to solve the puzzle
    `record`: bool
        if True, creates a .txt file with the same information printed according to the indicated verbosity

    ------
    Returns:
    ------
    `stats_dict`: dict
        dictionary containing various statistics about the function's performance trying to solve the puzzle
    """

    sugg_words = []

    for i in range(0, 20):
        ran_int = random.randint(0, len(word_list) - 1)
        word = word_list[ran_int]
        sugg_words.append(word)

    if guess not in word_list:
        print ("Guess word not in passed word list.\nOnly words within the given word list are valid.")
        print (f"Here are some examples of valid words from the passed word list.\n\t{sugg_words[:10]}")
        return None
    
    if target not in word_list:
        print ("Target word not in passed word list.\nOnly words within the given word list are valid.")
        print (f"Here are some examples of valid words from the passed word list.\n\t{sugg_words[-10:]}")
        return None

    if random_guess == True:
        randomint_guess = random.randint(0, len(word_list) - 1)
        guess = word_list[randomint_guess]

    if random_target == True:
        randomint_target = random.randint(0, len(word_list) - 1)
        target = word_list[randomint_target]

    stats_dict = {}
    stats_dict['first_guess'] = guess
    stats_dict['target_word'] = target
    stats_dict['first_guess_vowels'] = float(count_vows_cons(guess, y_vow = True)['vows'])
    stats_dict['first_guess_consonants'] = float(count_vows_cons(guess, y_vow = True)['cons'])
    stats_dict['target_vowels'] = float(count_vows_cons(target, y_vow = True)['vows'])
    stats_dict['target_consonants'] = float(count_vows_cons(target, y_vow = True)['cons'])
    
    # get entropy of the first guess word and target word in the entire word_list
    for tup in get_word_entropy(word_list, word_list, normalized = True):
        if tup[0] == guess:
            stats_dict['first_guess_entropy'] = tup[1]
        if tup[0] == target:
            stats_dict['target_entropy'] = tup[1]

    guess_entropies = []
    guess_entropies.append(stats_dict['first_guess_entropy'])

    # luck_guess_1 = round(1 - ((1 / len(word_list)) * guess_entropies[0] / 100), 2) * 100

    english_alphabet = "abcdefghijklmnopqrstuvwxyz"

    word_list_sorted_counts = get_letter_counts(english_alphabet, word_list, sort = "descending")
    
    wordlen = len(guess)
    letter_positions = set(i for i in range(0, wordlen))

    guess_set = set()
    perfect_dict = {}
    wrong_pos_dict = {}
    wrong_pos_set = set()
    dont_guess_again = set()

    guessed_words = [] # running set of guessed words
    guess_num = 0 # baseline for variable
    dont_guess_words = set()
    incorrect_positions = []
    reduction_per_guess = []

    if max_guesses == None: # if no value is passed, default is len(guess)
        max_guesses = wordlen
    else: # else it is the value passed
        max_guesses = max_guesses

    perfect_letts_per_guess = []
    wrong_pos_per_guess = []
    wrong_letts_per_guess = []

    record_list = []

    while guess: # while there is any guess -- there are conditions to break it at the bottom

        guess_num += 1

        guessed_words.append(guess)

        if drama:
            time.sleep(drama)

        # guess_num += 1 # each time the guess is processed
        if return_stats == False:
            if guess_num == 1:
                print("-----------------------------\n")
                record_list.append("-----------------------------\n")
    
        if return_stats == False:
            print(f"Guess {guess_num}: '{guess}'")
            record_list.append(f"Guess {guess_num}: '{guess}'")

        if guess == target:
            stats_dict['target_guessed'] = True
            if return_stats == False:
                if guess_num == 1:
                    print(f"Congratulations! The Wordle has been solved in {guess_num} guess, that's amazingly lucky!")
                    print(f"The target word was {target}")
                    record_list.append(f"Congratulations! The Wordle has been solved in {guess_num} guess, that's amazingly lucky!")
                    record_list.append(f"The target word was {target}")
                    perfect_letts_per_guess.append(5)
                    wrong_pos_per_guess.append(0)
                    wrong_letts_per_guess.append(0)
            break

        guess_set = set()
        wrong_pos_set = set()

        #### Step 2 -- ALL PERFECT
        for i in letter_positions: # number of letters in each word (current word and target word)
            guess_set.add(guess[i])

            if guess[i] not in perfect_dict:
                perfect_dict[guess[i]] = set()
            if guess[i] not in wrong_pos_dict:
                wrong_pos_dict[guess[i]] = set()

            ### EVALUATE CURRENT GUESS
            if guess[i] == target[i]: # letter == correct and position == correct
                perfect_dict[guess[i]].add(i)

            if (guess[i] != target[i] and  guess[i] in target): # letter == correct and position != correct
                wrong_pos_dict[guess[i]].add(i)
                wrong_pos_set.add(guess[i])

            if guess[i] not in target: # if letter is not relevant at all
                dont_guess_again.add(guess[i])

        #### Step 3 -- ALL PERFECT
        next_letters = set()
        for letter, positions in perfect_dict.items():
            if len(positions) > 0:
                next_letters.add(letter)

        for letter, positions in wrong_pos_dict.items():
            if len(positions) > 0:
                next_letters.add(letter)

        #### List of tuples of correct letter positions in new valid words. Eg: [('e', 2), ('a', 3)]
        perfect_letters = []
        for letter, positions in perfect_dict.items():
            for pos in positions:
                if len(positions) > 0:
                    perfect_letters.append((letter, pos))

        #### all words that have correct letters in same spots
        words_matching_correct_all = []
        for word in word_list:
            word_set = set()
            for letter, pos in perfect_letters:
                if word[pos] == letter:
                    words_matching_correct_all.append(word)

        #### excluding words with letters in known incorrect positions
        for letter, positions in wrong_pos_dict.items():
            for pos in positions:
                if len(positions) > 0:
                    if (letter, pos) not in incorrect_positions:
                        incorrect_positions.append((letter, pos))

        # sorting lists of tuples just to make them look nice in the printout
        incorrect_positions = sorted(incorrect_positions, key = operator.itemgetter(1), reverse = False)
        perfect_letters = sorted(perfect_letters, key = operator.itemgetter(1), reverse = False)

        #### all words that have correct letters in incorrect spots -- so they can be excluded efficiently
        
        # print(incorrect_positions)
        
        for word in word_list:
            word_set = set()
            for letter, pos in incorrect_positions:
                if word[pos] == letter:
                    dont_guess_words.add(word)
        for word in word_list:
            word_set = set()
            for letter, pos in incorrect_positions:
                if word[pos] == letter:
                    dont_guess_words.add(word)

        for bad_letter in dont_guess_again:
            for word in word_list:
                if (bad_letter in word and word not in dont_guess_words):
                    dont_guess_words.add(word)

        if return_stats == False:
            if verbose == True:
                print(f"Letters in correct positions:\n\t{perfect_letters}\n")
                print(f"Letters in incorrect positions:\n\t{incorrect_positions}\n")
                print (f"Letters to guess again:\n\t{sorted(list(next_letters), reverse = False)}\n")
                print(f"Letters to not guess again:\n\t{sorted(list(dont_guess_again), reverse = False)}\n") # works
                record_list.append(f"Letters in correct positions:\n\t{perfect_letters}\n")
                record_list.append(f"Letters in incorrect positions:\n\t{incorrect_positions}\n")
                record_list.append(f"Letters to guess again:\n\t{sorted(list(next_letters), reverse = False)}\n")
                record_list.append(f"Letters to not guess again:\n\t{sorted(list(dont_guess_again), reverse = False)}\n") # works

        # Returns True
        # print(A.issubset(B)) # "if everything in A is in B", returns Bool

        perfect_letts_per_guess.append(len(perfect_letters))
        wrong_pos_per_guess.append(len(incorrect_positions))
        wrong_letts_per_guess.append(len(dont_guess_again))

        potential_next_guesses = set()
        middle_set = set()

        if len(perfect_letters) == 0 and len(incorrect_positions) == 0: # if there are NEITHER perfect letters, NOR incorrect positions, ....
            for word in word_list:
                if word not in dont_guess_words:
                    if word not in guessed_words:
                        potential_next_guesses.add(word)
                                        
            # print(f"GUESS {guess_num} : TEST 1-1")

        if len(perfect_letters) == 0 and len(incorrect_positions) != 0: # if there are no perfect letters whatsoever, but there ARE incorrect positions ....
            for word in word_list:
                for incor_letter, incor_pos in incorrect_positions:
                    if word[incor_pos] != incor_letter:
                        if word not in dont_guess_words: # just in case
                            word_set = set()
                            for letter in word:
                                word_set.add(letter)

                                if next_letters.issubset(word_set):
                                    if word not in guessed_words:
                                        if len(dont_guess_again) > 0:
                                            for bad_letter in dont_guess_again:
                                                if bad_letter not in word:
                                                    # potential_next_guesses.append(word)
                                                    potential_next_guesses.add(word)
                                        else:
                                            potential_next_guesses.add(word)
            
            # print(f"GUESS {guess_num} : TEST 2-1")

        else:
            for word in word_list:
                if word not in dont_guess_words: # just in case
                    word_set = set()
                    for letter in word:
                        word_set.add(letter)
                        if next_letters.issubset(word_set):
                            if word not in guessed_words:
                                # print ("TEST 3-2")

                                if len(dont_guess_again) > 0:
                                    for bad_letter in dont_guess_again:
                                        if bad_letter not in word:
                                            middle_set.add(word)
                                else:
                                    middle_set.add(word)
            for word in middle_set:
                dummy_list = []
                for good_lett, good_pos in perfect_letters:
                    if word[good_pos] == good_lett:
                        dummy_list.append(1)
                        if len(dummy_list) == len(perfect_letters):
                            potential_next_guesses.add(word)
            for word in middle_set:
                dummy_list = []
                for bad_lett, bad_pos in incorrect_positions:
                    if word[bad_pos] == bad_lett:
                        dummy_list.append(1)
                        if len(dummy_list) > 0:
                            potential_next_guesses.remove(word)
                                        
            # print(f"GUESS {guess_num} : TEST 3-1")

        if return_stats == False:
            if verbose == True:
                print(f"At this point:")
                print(f"\t{len(word_list) - len(potential_next_guesses)}, {round((len(word_list) - len(potential_next_guesses)) / len(word_list) * 100, 2)}% of total words have been eliminated, and")
                print(f"\t{len(potential_next_guesses)}, {round(len(potential_next_guesses) / len(word_list) * 100, 2)}% of total words remain possible.\n")
                record_list.append(f"At this point:")
                record_list.append(f"\t{len(word_list) - len(potential_next_guesses)}, {round((len(word_list) - len(potential_next_guesses)) / len(word_list) * 100, 2)}% of total words have been eliminated, and")
                record_list.append(f"\t{len(potential_next_guesses)}, {round(len(potential_next_guesses) / len(word_list) * 100, 2)}% of total words remain possible.\n")
        
        reduction_per_guess.append(len(potential_next_guesses))
                
        #### Guessing next word
        if len(potential_next_guesses) == 1:

            if return_stats == False:
                if verbose == True:
                    print(f"The only remaining possible word is:\n\t'{list(potential_next_guesses)[0]}'\n")
                    record_list.append(f"The only remaining possible word is:\n\t'{list(potential_next_guesses)[0]}'\n")
                
            guess = list(potential_next_guesses)[0]
            guess_entropies.append(get_word_entropy([guess], word_list, normalized = True, ascending = False)[0][1])

        else:

            if bias == "entropy":
                
                best_next_guesses = list(potential_next_guesses)                
                # print (best_next_guesses)
                word_ratings = get_word_entropy(best_next_guesses, word_list, normalized = True, ascending = False) # "internal" ratings

                # Get max rated word
                max_rating = -np.inf
                for word, rating in word_ratings:
                    if rating > max_rating:
                        max_rating = rating

                for word, rating in word_ratings:
                    if rating == max_rating:
                        guess = word
                
                guess_entropies.append(get_word_entropy([guess], word_list, normalized = True, ascending = False)[0][1])

                if return_stats == False:
                    if verbose == True:
                        if len(word_ratings) <= 40:
                            print(f"All potential next guesses:\n\t{word_ratings}\n")
                            print(f"Words guessed so far:\n\t{guessed_words}.\n")
                            record_list.append(f"Potential next guesses:\n\t{word_ratings}\n")
                            record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n")
                        else:
                            print(f"The top 40 potential next guesses are:\n\t{word_ratings[:40]}\n")
                            print(f"Words guessed so far:\n\t{guessed_words}.\n")
                            record_list.append(f"The top 40 potential next guesses are::\n\t{word_ratings[:40]}\n")
                            record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n")

            if bias == "no_bias":
                best_next_guesses = set()
                for word in potential_next_guesses:
                    for letter, freq in word_list_sorted_counts:
                        if letter not in dont_guess_again:
                            if len(next_letters) > 0:
                                if letter in next_letters:
                                    if letter in word:
                                        best_next_guesses.add(word)
                                        break
                            else:
                                if letter in word:
                                    best_next_guesses.add(word)
                                    break
                                
                if return_stats == False:
                    if verbose == True:
                        if len(best_next_guesses) <= 40:
                            print(f"Potential next guesses:\n\t{best_next_guesses}\n")
                            print(f"Words guessed so far:\n\t{guessed_words}.\n") 
                            record_list.append(f"Potential next guesses:\n\t{best_next_guesses}\n")
                            record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n") 

            if bias == ("common" or "rare"):
                found_words = []
                for word in word_list:
                    if word in nltk_counts.keys():
                        found_words.append(word)

                found_words_sorted = sorted(found_words, key = operator.itemgetter(1), reverse = True) # sorted descending

                rated_words = []
                for word in potential_next_guesses:
                    for tup in found_words_sorted:
                        if tup[0] == word:
                            rated_words.append(tup)

                rated_words = sorted(rated_words, key = operator.itemgetter(1), reverse = True) # sorted descending
                
                if bias == "common":
                    guess = rated_words[0][0] # word in first position // most frequent word
                    
                    if return_stats == False:
                        if verbose == True:
                            if len(potential_next_guesses) <= 40:
                                print(f"Potential next guesses:\n\t{rated_words}\n")
                                print(f"Words guessed so far:\n\t{guessed_words}.\n") 
                                record_list.append(f"Potential next guesses:\n\t{potential_next_guesses}\n")
                                record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n") 
                
                if bias == "rare":
                    guess = rated_words[-1][0] # word in last position // least frequent word
                
                    if return_stats == False:
                        if verbose == True:
                            if len(potential_next_guesses) <= 40:
                                print(f"Potential next guesses:\n\t{rated_words}\n")
                                print(f"Words guessed so far:\n\t{guessed_words}.\n") 
                                record_list.append(f"Potential next guesses:\n\t{potential_next_guesses}\n")
                                record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n") 
                    
                # guess = list(best_next_guesses)[0]
                guess_entropies.append(get_word_entropy([guess], word_list, normalized = True, ascending = False)[0][1])

        #### Guess has now been made -- what to do next
        if guess_num == max_guesses: # if at max guesses allowed
            guessed_words.append(guess)
            stats_dict['target_guessed'] = False
            if return_stats == False:
                if verbose == True:
                    # print("-----------------------------\n")
                    print(f"Unfortunately, the Wordle could not be solved in {max_guesses} guesses.\n")
                    print(f"The target word was '{target}'. Better luck next time!\n")
                    print("-----------------------------\n")
                    record_list.append(f"Unfortunately, the Wordle could not be solved in {max_guesses} guesses.\n")
                    record_list.append(f"The target word was '{target}'. Better luck next time!\n")
                    record_list.append("-----------------------------\n")
                else:
                    print(f"\nUnfortunately, the Wordle could not be solved in {max_guesses} guesses.")
                    print(f"The target word was '{target}'. Better luck next time!\n")
                    record_list.append(f"\nUnfortunately, the Wordle could not be solved in {max_guesses} guesses.")
                    record_list.append(f"The target word was '{target}'. Better luck next time!\n")
            break
        else: # if not at max guesses yet allowed
            # stats_dict['target_guessed'] = False
            if return_stats == False:
                if verbose == True:
                    print(f"Next guess:\n\t'{guess}'")
                    print("\n-----------------------------\n")
                    record_list.append(f"Next guess:\n\t'{guess}'")
                    record_list.append("\n-----------------------------\n")

        if guess == target:
            guess_num += 1
            guessed_words.append(guess)
            stats_dict['target_guessed'] = True

            if return_stats == False:
                print(f"Guess {guess_num}: '{guess}'\n")
                print(f"Congratulations! The Wordle has been solved in {guess_num} guesses!")
                record_list.append(f"Guess {guess_num}: '{guess}'\n")
                record_list.append(f"Congratulations! The Wordle has been solved in {guess_num} guesses!")

                if max_guesses - guess_num == 0:
                    print(f"Lucky! It was the last guess.")
                    record_list.append(f"Lucky! It was the last guess.")
                else:
                    print(f"There were still {max_guesses - guess_num} guesses remaining.")
                    record_list.append(f"There were still {max_guesses - guess_num} guesses remaining.")

            if return_stats == False:   
                # stats_dict['target_guessed'] = True                 
                print(f"\nThe target word was '{target}'.")
                print("\n-----------------------------")
                record_list.append(f"\nThe target word was '{target}'.")
                record_list.append("\n-----------------------------")
            break

    #### STATS STUFF    
    mid_guesses_vows = 0
    mid_guesses_cons = 0
    avg_perf_letters = 0
    avg_wrong_pos_letters = 0
    avg_wrong_letters = 0

    for i, word in enumerate(guessed_words):
        mid_guesses_vows += count_vows_cons(word, y_vow = True)['vows']
        mid_guesses_cons += count_vows_cons(word, y_vow = True)['cons']
        
    for i in range(0, len(guessed_words) - 1):
        avg_perf_letters += perfect_letts_per_guess[i]
        avg_wrong_pos_letters += wrong_pos_per_guess[i]
        avg_wrong_letters += wrong_letts_per_guess[i]

    stats_dict['mid_guesses_avg_vows'] = float(round(mid_guesses_vows / len(guessed_words), 2))
    stats_dict['mid_guesses_avg_cons'] = float(round(mid_guesses_cons / len(guessed_words), 2))

    stats_dict['avg_perf_letters'] = float(round(np.mean(avg_perf_letters), 2))
    stats_dict['avg_wrong_pos_letters'] = float(round(np.mean(avg_wrong_pos_letters), 2))
    stats_dict['avg_wrong_letters'] = float(round(np.mean(avg_wrong_letters), 2))
    
    # average number of words remaining after each guess -- the higher this is, the luckier the person got (the lower, the more guesses it took)
    stats_dict['avg_remaining'] = float(round(np.mean(reduction_per_guess), 2))

    # avg entropy of each guessed word relative to all other words possible at that moment -- this should consistently be 100 for the algorithm, but will be different for user
    if len(guess_entropies) > 1: # in case of guessing it correctly on the first try
        sum_entropies = 0
        for entropy in guess_entropies:
            sum_entropies += entropy

        average_entropy = float(round(sum_entropies / len(guess_entropies), 2))
        stats_dict['avg_intermediate_guess_entropy'] = average_entropy
    else:
        stats_dict['avg_intermediate_guess_entropy'] = float(100)

    expected_guesses = 3.85

    # guess_num = 3
    # average_entropy = 95
    luck = round(1 - ((((guess_num / expected_guesses) * (stats_dict['avg_intermediate_guess_entropy'] / 100)) / max_guesses) * 5), 2)
    stats_dict['luck'] = luck
    
    stats_dict['bias'] = bias


    if record == True:
        if verbose == True:
            with open(f"solutions/{guessed_words[0]}_{target}_wizard_detailed.txt", "w") as fout:
                for line in record_list:
                    fout.write(line + "\n") # write each line of list of printed text to .txt file
        else:
            with open(f"solutions/{guessed_words[0]}_{target}_wizard_summary.txt", "w") as fout:
                for line in record_list:
                    fout.write(line + "\n") # write


    # if guess_num <= len(guess):
    if guess_num <= 6:
        stats_dict['valid_success'] = True
    else:
        stats_dict['valid_success'] = False

    stats_dict['num_guesses'] = float(guess_num)

    if return_stats == True:
        return stats_dict

def compare_wordle(word_list: list, max_guesses: int = None, guess_list: list = None,
                  player: str = None, target: str = None,
                  verbose: bool = False,
                  return_stats: bool = False, record: bool = False):
    """
    Mimicking the popular web game, this function matches a current word to a target word automatically, in the most statistically optimal way possible.

    ------
    Parameters:
    ------
    `word_list`: list
        list of valid words to be considered
    `target`: str
        a string -- must be the same length as `opening_word`
    `max_guesses`: int
        the maximum number of attempts allowed to solve the Wordle
    `verbose`: bool
        if True, prints progress and explanation of how function solves the puzzle. If False, prints only the guessed word at each guess.
    `return_stats`: bool
        if True, prints nothing and returns a dictionary of various statistics about the function's performance trying to solve the puzzle
    `record`: bool
        if True, creates a .txt file with the same information printed according to the indicated verbosity

    ------
    Returns:
    ------
    `stats_dict`: dict
        dictionary containing various statistics about the function's performance trying to solve the puzzle
    """
    
    stats_dict = {}
    
    # official_words list seems to not be 100% the same as the real game, so this adds new words to it
    for word in guess_list:
        if word not in word_list:
            word_list.append(word)

    guess = guess_list[0]
    first_guess = guess_list[0]

    stats_dict['first_guess'] = guess
    stats_dict['target_word'] = target
    stats_dict['first_guess_vowels'] = float(count_vows_cons(guess, y_vow = True)['vows'])
    stats_dict['first_guess_consonants'] = float(count_vows_cons(guess, y_vow = True)['cons'])
    stats_dict['target_vowels'] = float(count_vows_cons(target, y_vow = True)['vows'])
    stats_dict['target_consonants'] = float(count_vows_cons(target, y_vow = True)['cons'])
    
    # get entropy of the first guess word and target word in the entire word_list
    for tup in get_word_entropy(word_list, word_list, normalized = True):
        if tup[0] == guess:
            stats_dict['first_guess_entropy'] = tup[1]
        if tup[0] == target:
            stats_dict['target_entropy'] = tup[1]

    guess_entropies = []
    guess_entropies.append(stats_dict['first_guess_entropy'])

    english_alphabet = "abcdefghijklmnopqrstuvwxyz"

    word_list_sorted_counts = get_letter_counts(english_alphabet, word_list, sort = "descending")
    
    wordlen = len(guess)
    letter_positions = set(i for i in range(0, wordlen))

    guess_set = set()
    perfect_dict = {}
    wrong_pos_dict = {}
    wrong_pos_set = set()
    dont_guess_again = set()

    guessed_words = [] # running set of guessed words
    guess_num = 0 # baseline for variable
    dont_guess_words = set()
    incorrect_positions = []
    reduction_per_guess = []

    if max_guesses == None: # if no value is passed, default is len(guess)
        max_guesses = wordlen
    else: # else it is the value passed
        max_guesses = max_guesses

    perfect_letts_per_guess = []
    wrong_pos_per_guess = []
    wrong_letts_per_guess = []

    record_list = []

    while guess: # while there is any guess -- there are conditions to break it at the bottom

        guess_num += 1

        guessed_words.append(guess)

        # if drama:
        #     time.sleep(drama)

        # guess_num += 1 # each time the guess is processed
        if return_stats == False:
            if guess_num == 1:
                print("-----------------------------\n")
                record_list.append("-----------------------------\n")
    
        if return_stats == False:
            print(f"Guess {guess_num}: '{guess}'")
            record_list.append(f"Guess {guess_num}: '{guess}'")

        if guess == target:
            stats_dict['target_guessed'] = True
            if return_stats == False:
                if guess_num == 1:
                    print(f"Congratulations! The Wordle has been solved in {guess_num} guess, that's amazingly lucky!")
                    print(f"The target word was {target}")
                    record_list.append(f"Congratulations! The Wordle has been solved in {guess_num} guess, that's amazingly lucky!")
                    record_list.append(f"The target word was {target}")
                    perfect_letts_per_guess.append(5)
                    wrong_pos_per_guess.append(0)
                    wrong_letts_per_guess.append(0)
            break

        guess_set = set()
        wrong_pos_set = set()

        #### Step 2 -- ALL PERFECT
        for i in letter_positions: # number of letters in each word (current word and target word)
            guess_set.add(guess[i])

            if guess[i] not in perfect_dict:
                perfect_dict[guess[i]] = set()
            if guess[i] not in wrong_pos_dict:
                wrong_pos_dict[guess[i]] = set()

            ### EVALUATE CURRENT GUESS
            if guess[i] == target[i]: # letter == correct and position == correct
                perfect_dict[guess[i]].add(i)

            if (guess[i] != target[i] and  guess[i] in target): # letter == correct and position != correct
                wrong_pos_dict[guess[i]].add(i)
                wrong_pos_set.add(guess[i])

            if guess[i] not in target: # if letter is not relevant at all
                dont_guess_again.add(guess[i])

        #### Step 3 -- ALL PERFECT
        next_letters = set()
        for letter, positions in perfect_dict.items():
            if len(positions) > 0:
                next_letters.add(letter)

        for letter, positions in wrong_pos_dict.items():
            if len(positions) > 0:
                next_letters.add(letter)

        #### List of tuples of correct letter positions in new valid words. Eg: [('e', 2), ('a', 3)]
        perfect_letters = []
        for letter, positions in perfect_dict.items():
            for pos in positions:
                if len(positions) > 0:
                    perfect_letters.append((letter, pos))

        #### all words that have correct letters in same spots
        words_matching_correct_all = []
        for word in word_list:
            word_set = set()
            for letter, pos in perfect_letters:
                if word[pos] == letter:
                    words_matching_correct_all.append(word)

        #### excluding words with letters in known incorrect positions
        for letter, positions in wrong_pos_dict.items():
            for pos in positions:
                if len(positions) > 0:
                    if (letter, pos) not in incorrect_positions:
                        incorrect_positions.append((letter, pos))

        # sorting lists of tuples just to make them look nice in the printout
        incorrect_positions = sorted(incorrect_positions, key = operator.itemgetter(1), reverse = False)
        perfect_letters = sorted(perfect_letters, key = operator.itemgetter(1), reverse = False)

        #### all words that have correct letters in incorrect spots -- so they can be excluded efficiently
        
        # print(incorrect_positions)
        
        for word in word_list:
            word_set = set()
            for letter, pos in incorrect_positions:
                if word[pos] == letter:
                    dont_guess_words.add(word)
        for word in word_list:
            word_set = set()
            for letter, pos in incorrect_positions:
                if word[pos] == letter:
                    dont_guess_words.add(word)

        for bad_letter in dont_guess_again:
            for word in word_list:
                if (bad_letter in word and word not in dont_guess_words):
                    dont_guess_words.add(word)

        if return_stats == False:
            if verbose == True:
                print(f"Letters in correct positions:\n\t{perfect_letters}\n")
                print(f"Letters in incorrect positions:\n\t{incorrect_positions}\n")
                print (f"Letters to guess again:\n\t{sorted(list(next_letters), reverse = False)}\n")
                print(f"Letters to not guess again:\n\t{sorted(list(dont_guess_again), reverse = False)}\n") # works
                record_list.append(f"Letters in correct positions:\n\t{perfect_letters}\n")
                record_list.append(f"Letters in incorrect positions:\n\t{incorrect_positions}\n")
                record_list.append(f"Letters to guess again:\n\t{sorted(list(next_letters), reverse = False)}\n")
                record_list.append(f"Letters to not guess again:\n\t{sorted(list(dont_guess_again), reverse = False)}\n") # works

        # Returns True
        # print(A.issubset(B)) # "if everything in A is in B", returns Bool

        perfect_letts_per_guess.append(len(perfect_letters))
        wrong_pos_per_guess.append(len(incorrect_positions))
        wrong_letts_per_guess.append(len(dont_guess_again))

        potential_next_guesses = set()
        middle_set = set()

        if len(perfect_letters) == 0 and len(incorrect_positions) == 0: # if there are NEITHER perfect letters, NOR incorrect positions, ....
            for word in word_list:
                if word not in dont_guess_words:
                    if word not in guessed_words:
                        potential_next_guesses.add(word)
                                        
            # print(f"GUESS {guess_num} : TEST 1-1")

        if len(perfect_letters) == 0 and len(incorrect_positions) != 0: # if there are no perfect letters whatsoever, but there ARE incorrect positions ....
            for word in word_list:
                for incor_letter, incor_pos in incorrect_positions:
                    if word[incor_pos] != incor_letter:
                        if word not in dont_guess_words: # just in case
                            word_set = set()
                            for letter in word:
                                word_set.add(letter)

                                if next_letters.issubset(word_set):
                                    if word not in guessed_words:
                                        if len(dont_guess_again) > 0:
                                            for bad_letter in dont_guess_again:
                                                if bad_letter not in word:
                                                    # potential_next_guesses.append(word)
                                                    potential_next_guesses.add(word)
                                        else:
                                            potential_next_guesses.add(word)
            
            # print(f"GUESS {guess_num} : TEST 2-1")

        else:
            for word in word_list:
                if word not in dont_guess_words: # just in case
                    word_set = set()
                    for letter in word:
                        word_set.add(letter)
                        if next_letters.issubset(word_set):
                            if word not in guessed_words:
                                # print ("TEST 3-2")

                                if len(dont_guess_again) > 0:
                                    for bad_letter in dont_guess_again:
                                        if bad_letter not in word:
                                            middle_set.add(word)
                                else:
                                    middle_set.add(word)
            for word in middle_set:
                dummy_list = []
                for good_lett, good_pos in perfect_letters:
                    if word[good_pos] == good_lett:
                        dummy_list.append(1)
                        if len(dummy_list) == len(perfect_letters):
                            potential_next_guesses.add(word)
            for word in middle_set:
                dummy_list = []
                for bad_lett, bad_pos in incorrect_positions:
                    if word[bad_pos] == bad_lett:
                        dummy_list.append(1)
                        if len(dummy_list) > 0:
                            potential_next_guesses.remove(word)
                                        
            # print(f"GUESS {guess_num} : TEST 3-1")

        if return_stats == False:
            if verbose == True:
                print(f"At this point:")
                print(f"\t{len(word_list) - len(potential_next_guesses)}, {round((len(word_list) - len(potential_next_guesses)) / len(word_list) * 100, 2)}% of total words have been eliminated, and")
                print(f"\t{len(potential_next_guesses)}, {round(len(potential_next_guesses) / len(word_list) * 100, 2)}% of total words remain possible.\n")
                record_list.append(f"At this point:")
                record_list.append(f"\t{len(word_list) - len(potential_next_guesses)}, {round((len(word_list) - len(potential_next_guesses)) / len(word_list) * 100, 2)}% of total words have been eliminated, and")
                record_list.append(f"\t{len(potential_next_guesses)}, {round(len(potential_next_guesses) / len(word_list) * 100, 2)}% of total words remain possible.\n")
        
        reduction_per_guess.append(len(potential_next_guesses))
                
        #### Guessing next word
        if len(potential_next_guesses) == 1:

            if return_stats == False:
                if verbose == True:
                    print(f"The only remaining possible word is:\n\t'{list(potential_next_guesses)[0]}'\n")
                    record_list.append(f"The only remaining possible word is:\n\t'{list(potential_next_guesses)[0]}'\n")
                
            # guess = list(potential_next_guesses)[0]
            del guess_list[0]
            # print (guess_list)
            guess = guess_list[0]
            guess_entropies.append(get_word_entropy([guess], word_list, normalized = True, ascending = False)[0][1])

        else:
            
            best_next_guesses = list(potential_next_guesses)                
            word_ratings = get_word_entropy(best_next_guesses, word_list, normalized = True, ascending = False) # "internal" ratings

            del guess_list[0]
            # print (guess_list)
            guess = guess_list[0]

            guess_entropies.append(get_word_entropy([guess], word_list, normalized = True, ascending = False)[0][1])

            if return_stats == False:
                if verbose == True:
                    if len(word_ratings) <= 40:
                        print(f"All potential next guesses:\n\t{word_ratings}\n")
                        print(f"Words guessed so far:\n\t{guessed_words}.\n")
                        record_list.append(f"Potential next guesses:\n\t{word_ratings}\n")
                        record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n")
                    else:
                        print(f"The top 40 potential next guesses are:\n\t{word_ratings[:40]}\n")
                        print(f"Words guessed so far:\n\t{guessed_words}.\n")
                        record_list.append(f"The top 40 potential next guesses are::\n\t{word_ratings[:40]}\n")
                        record_list.append(f"Words guessed so far:\n\t{guessed_words}.\n")

        #### Guess has now been made -- what to do next
        if guess_num == max_guesses: # if at max guesses allowed
            guessed_words.append(guess)
            stats_dict['target_guessed'] = False
            if return_stats == False:
                if verbose == True:
                    # print("-----------------------------\n")
                    print(f"Unfortunately, the Wordle could not be solved in {max_guesses} guesses.\n")
                    print(f"The target word was '{target}'. Better luck next time!\n")
                    print("-----------------------------\n")
                    record_list.append(f"Unfortunately, the Wordle could not be solved in {max_guesses} guesses.\n")
                    record_list.append(f"The target word was '{target}'. Better luck next time!\n")
                    record_list.append("-----------------------------\n")
                else:
                    print(f"\nUnfortunately, the Wordle could not be solved in {max_guesses} guesses.")
                    print(f"The target word was '{target}'. Better luck next time!\n")
                    record_list.append(f"\nUnfortunately, the Wordle could not be solved in {max_guesses} guesses.")
                    record_list.append(f"The target word was '{target}'. Better luck next time!\n")
            break
        else: # if not at max guesses yet allowed
            # stats_dict['target_guessed'] = False
            if return_stats == False:
                if verbose == True:
                    print(f"Next guess:\n\t'{guess}'")
                    print("\n-----------------------------\n")
                    record_list.append(f"Next guess:\n\t'{guess}'")
                    record_list.append("\n-----------------------------\n")

        if guess == target:
            guess_num += 1
            guessed_words.append(guess)
            stats_dict['target_guessed'] = True

            if return_stats == False:
                print(f"Guess {guess_num}: '{guess}'\n")
                print(f"Congratulations! The Wordle has been solved in {guess_num} guesses!")
                record_list.append(f"Guess {guess_num}: '{guess}'\n")
                record_list.append(f"Congratulations! The Wordle has been solved in {guess_num} guesses!")

                if max_guesses - guess_num == 0:
                    print(f"Lucky! It was the last guess.")
                    record_list.append(f"Lucky! It was the last guess.")
                else:
                    print(f"There were still {max_guesses - guess_num} guesses remaining.")
                    record_list.append(f"There were still {max_guesses - guess_num} guesses remaining.")

            if return_stats == False:   
                # stats_dict['target_guessed'] = True                 
                print(f"\nThe target word was '{target}'.")
                print("\n-----------------------------")
                record_list.append(f"\nThe target word was '{target}'.")
                record_list.append("\n-----------------------------")
            break

    #### STATS STUFF    
    mid_guesses_vows = 0
    mid_guesses_cons = 0
    avg_perf_letters = 0
    avg_wrong_pos_letters = 0
    avg_wrong_letters = 0

    for i, word in enumerate(guessed_words):
        mid_guesses_vows += count_vows_cons(word, y_vow = True)['vows']
        mid_guesses_cons += count_vows_cons(word, y_vow = True)['cons']
        
    for i in range(0, len(guessed_words) - 1):
        avg_perf_letters += perfect_letts_per_guess[i]
        avg_wrong_pos_letters += wrong_pos_per_guess[i]
        avg_wrong_letters += wrong_letts_per_guess[i]

    stats_dict['mid_guesses_avg_vows'] = float(round(mid_guesses_vows / len(guessed_words), 2))
    stats_dict['mid_guesses_avg_cons'] = float(round(mid_guesses_cons / len(guessed_words), 2))

    stats_dict['avg_perf_letters'] = float(round(np.mean(avg_perf_letters), 2))
    stats_dict['avg_wrong_pos_letters'] = float(round(np.mean(avg_wrong_pos_letters), 2))
    stats_dict['avg_wrong_letters'] = float(round(np.mean(avg_wrong_letters), 2))
    
    # average number of words remaining after each guess -- the higher this is, the luckier the person got (the lower, the more guesses it took)
    stats_dict['avg_remaining'] = float(round(np.mean(reduction_per_guess), 2))

    # avg entropy of each guessed word relative to all other words possible at that moment -- this should consistently be 100 for the algorithm, but will be different for user
    if len(guess_entropies) > 1: # in case of guessing it correctly on the first try
        sum_entropies = 0
        for entropy in guess_entropies:
            sum_entropies += entropy

        average_entropy = float(round(sum_entropies / len(guess_entropies), 2))
        stats_dict['avg_intermediate_guess_entropy'] = average_entropy
    else:
        stats_dict['avg_intermediate_guess_entropy'] = float(100)
    
    # stats_dict['bias'] = bias

    if record == True:
        if verbose == True:
            with open(f"solutions/{guessed_words[0]}_{target}_wizard_detailed.txt", "w") as fout:
                for line in record_list:
                    fout.write(line + "\n") # write each line of list of printed text to .txt file
        else:
            with open(f"solutions/{guessed_words[0]}_{target}_wizard_summary.txt", "w") as fout:
                for line in record_list:
                    fout.write(line + "\n") # write

    # if guess_num <= len(guess):
    if guess_num <= 6:
        stats_dict['valid_success'] = True
    else:
        stats_dict['valid_success'] = False

    stats_dict['player'] = player

    stats_dict['num_guesses'] = float(guess_num)

    wizard_dict = wordle_wizard(word_list = word_list, max_guesses = max_guesses, 
        guess = first_guess, target = target, bias = 'entropy', 
        random_guess = False, random_target = False, 
        verbose = False, drama = 0, return_stats = return_stats, record = False)
    
    wizard_dict['player'] = "wizard"
    del wizard_dict['bias'] # leftover from the wordle_wizard() output stats_dict, but isn't relevant anymore
    wizard_dict['luck'] = 0
    
    wizard_dict['expected_guesses'] = wizard_dict['num_guesses']
    stats_dict['expected_guesses'] = wizard_dict['num_guesses']

    expected_guesses = wizard_dict['num_guesses']
    # stats_dict['luck'] = round((1 - (guess_num / expected_guesses)) * (stats_dict['avg_intermediate_guess_entropy'] / 100), 2)
    stats_dict['luck'] = round((1 - ((guess_num / expected_guesses)) * (stats_dict['avg_intermediate_guess_entropy'] / 100)), 2)

    stats_master = {}
    for metric, result in stats_dict.items():
        if metric in stats_master:
            stats_master[metric].append(result)
        else:
            stats_master[metric] = []
            stats_master[metric].append(result)

    for metric, result in wizard_dict.items():
        stats_master[metric].append(result)

    if return_stats == True:
        return stats_master
    
def convert_row(df, row):
    """
    Converts row of passed pandas dataFrame object into usable inputs for `compare_wordle()` function
    
    ------
    Parameters:
    ------
    `df`: pandas df object
        pandas dataFrame object
    `row`: int
        row number of pd df object

    ------
    Returns:
    ------
    3-tuple containing:
    `guess_list`: list
        list of words guessed in this playthrough of passed iteration of the puzzle
    `target`: str
        target word of passed iteration of the puzzle
    `player`: str
        name of player of passed iteration of the puzzle
    """

    list_1 = df.loc[row, :].tolist()
    
    player = list_1[-1]
    del list_1[-1]
    target = list_1[-1]
    del list_1[-1]

    to_delete = []
    for i, word in enumerate(list_1):
        if word == "none":
            to_delete.append(i)

    to_delete = sorted(to_delete, reverse = True) # this has to be done or else index will be eventually be out of range

    for pos in to_delete:
        del list_1[pos]

    guess_list = list_1

    return (guess_list, target, player)

