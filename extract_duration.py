def extract_word_duration(sen):
    """This functions extracts and calculates
    the duration of the words in the TextGrid file 
    using slicing. It takes a list of strings as 
    an argument. Returns a list of tuples with 
    words at index 0 and durations at index 1."""
    sen_lst = []
    for i in range(len(sen)):
        if 'text' in sen[i] and 'text = ""' not in sen[i]:
            # Cleaningand and extracting the word string at idx [2]
            word = sen[i].replace('"', '').lower().split()[2]
            # Extracts the strings at i-1 and i-2, split them, and exctracts the duration at idx[2]
            duration_xmax = float(sen[i-1].split()[2])
            duration_xmin = float(sen[i-2].split()[2])
            duration = duration_xmax - duration_xmin
            # Converts to ms and appends to list
            sen_lst.append((word, round((duration*1000), 3)))
    return sen_lst


if __name__ == '__main__':
    test = ['intervals [8]', 'xmin = 2.100000',
            'xmax = 2.220000 ', 'text = "FOOT"']
    cf = extract_word_duration(test)
    print(cf)

   