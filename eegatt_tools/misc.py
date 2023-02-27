from collections import Iterable


def flatten(lis):
    """
    This function flattens a nested list
    :param lis:
    :return: A generator, use list to make a list out of it.
    """
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

