def accord_index(actual, expected):
    '''
    calculate the accord score of two sets of trailer
    titles. These sets are assumed to be lowercase and
    stripped of "special" characters 
    '''
    intersection = 0
    for word in actual:
        if word in expected:
            intersection += 1

    union = len(actual) + len(expected) - intersection

    return 1.0 * intersection / union

if __name__ =="__main__":
    print(accord_index(["incredible", "hulk", "official", "trailer"],
                 ["the", "incredible", "hulk", "official", "trailer"]))
        
    
