import numpy


def high_bit_pos(i):
    """Converts a positive integer to base 2 and returns the position of the high order bit.

    Keyword arguments:
    i -- Integer to find high order bit within
    """
    if i < 0:
        raise RuntimeError("Supplied value {0} was not positive".format(i))

    i = numpy.floor(i)
    bit = 0
    while i > 0:
        bit += 1
        i //= 2
    return bit


def low_bit_pos(i):
    """Converts a positive integer to base 2 and returns the position of the low order bit.

    Keyword arguments:
    i -- Integer to find high order bit within
    """
    if i < 0:
        raise RuntimeError("Supplied value {0} was not positive".format(i))

    i = numpy.floor(i)
    bit = 1
    while i != 2 * (i // 2):
        bit += 1
        i //= 2
    return bit
