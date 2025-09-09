def which_slot(tup):
    if tup[0] >0 and tup[0] < 50:
        return 1
    elif tup[0] > 50 and tup[0] < 100:
        return 2
    elif tup[0] > 100 and tup[0] < 150:
        return 3
    elif tup[0] > 150 and tup[0] < 200:
        return 4
    else:
        return None  # Explicitly return None for coordinates outside valid ranges