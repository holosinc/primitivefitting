def map_range(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

def argmin(lst, f):
    min_score = None
    min_elem = None
    for elem in lst:
        score = f(elem)
        if min_score is None or score < min_score:
            min_score = score
            min_elem = elem
    return min_elem

def argmax(lst, f):
    max_score = None
    max_elem = None
    for elem in lst:
        score = f(elem)
        if max_score is None or score > max_score:
            max_score = score
            max_elem = elem
    return max_elem

# Partial function application
def partial(f, *args, **kwargs):
    def ret(*more_args, **morekwargs):
        kwargs.update(morekwargs)
        return f(*(args + more_args), **kwargs)
    return ret