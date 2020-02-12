def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x: x[1])[0]

def plot_ccs(S):
    pass