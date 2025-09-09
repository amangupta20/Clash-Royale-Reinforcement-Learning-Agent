def which_slot(tup):
    if tup[0] >0 and tup[0] < 40:
        return 1
    elif tup[0] > 40 and tup[0] < 80:
        return 2
    elif tup[0] > 80 and tup[0] < 120:
        return 3
    elif tup[0] > 120 and tup[0] < 160:
        return 4
    else:
        return None  # Explicitly return None for coordinates outside valid ranges