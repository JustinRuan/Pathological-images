import numpy
from .utility import high_bit_pos, low_bit_pos


class SobolGenerator(object):
    """Handles all aspects of Sobol' sequence generation."""

    n_max = 40
    n_min = 1
    log_max = 30
    int_type_string = "i4"
    int_type = numpy.int32

    def __init__(
        self,
        n,
        seed = None,
        leap = 0
    ):
        self.n = n

        if seed is None:
            self.seed = SobolGenerator.good_seeds[self.n]
        else:
            self.seed = SobolGenerator.int_type(seed)

        self.leap = leap

        self.last_q = numpy.zeros(self.n, SobolGenerator.int_type_string)

        for i in range(self.seed):
            self.last_q = self.nextQ(self.last_q, i)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value < SobolGenerator.n_min or value > SobolGenerator.n_max:
            raise RuntimeError(
                "Number of dimensions specified outside of range [{0}, {1}]".format(
                    SobolGenerator.n_min, SobolGenerator.n_max
                )
            )

        self._n = value
        self.v = numpy.copy(SobolGenerator.base_v)

        for i in range(1, self.n):
            j = SobolGenerator.poly[i]
            m = 0
            j //= 2
            while j > 0:
                j //= 2
                m += 1
            j = SobolGenerator.poly[i]
            include = numpy.full(m, False, bool)

            for k in reversed(range(m)):
                j2 = j // 2
                include[k] = (j != 2 * j2)
                j = j2

            for j in range(m, SobolGenerator.max_col):
                new_v = self.v[i, j - m]
                l = 1
                for k in range(m):
                    l *= 2
                    if include[k]:
                        new_v = numpy.bitwise_xor(
                            new_v, l * self.v[i, j - k - 1]
                        )
                self.v[i, j] = new_v

        l = 1
        for j in reversed(range(0, SobolGenerator.max_col - 1)):
            l *= 2
            self.v[:, j] *= l

        self.recip_d = 1.0 / float(2 * l)

    def generate(self, N, seed = None, leap = None):
        """Generates a Sobol' sequence.

        Keyword arguments:
        N -- How many elements to generate
        seed -- The seed to start from (defaults to last seed used, or 0)
        leap -- The number of elements to ignore for every element taken (defaults
            to value set in object, or 0)
        """

        if seed is None:
            seed = self.seed
        if leap is None:
            leap = self.leap

        matrix = numpy.full((N, self.n), numpy.nan)
        next_output = seed
        number_output = 0

        while number_output < N:
            element = self.element(seed)

            if seed == next_output:
                matrix[number_output, :] = element
                next_output += leap + 1
                number_output += 1

            seed += 1

        return matrix

    def element(self, seed):
        """Generates a single element of a Sobol' sequence.

        Keyword arguments:
        seed - The seed to use to generate the element
        """

        if seed == 0:
            self.seed = 0
            self.last_q = numpy.zeros(self.n, SobolGenerator.int_type_string)
        elif seed > self.seed:
            for _ in range(self.seed, seed):
                self.last_q = self.nextQ(self.last_q, self.seed)
                self.seed += 1
        elif seed < self.seed:
            self.last_q = numpy.zeros(self.n, SobolGenerator.int_type_string)
            self.seed = 0
            for _ in range(0, seed):
                self.last_q = self.nextQ(self.last_q, self.seed)
                self.seed += 1

        vec = self.last_q * self.recip_d

        self.last_q = self.nextQ(self.last_q, self.seed)
        self.seed += 1

        return vec

    def nextQ(self, last_q, seed):
        """Generates the next value of Q in the sequence

        Keyword arguments:
        last_q -- The last Q in the sequence
        seed -- The seed used to generate the last Q
        """

        if seed == 0:
            l = 1
        else:
            l = low_bit_pos(seed)

        if l >= SobolGenerator.max_col:
            raise RuntimeError(
                "Requested element too far into sequence. Element {0} requested but \
                element {1} is maximum".format(
                    seed + 1, SobolGenerator.at_most
                )
            )

        return numpy.bitwise_xor(last_q, self.v[0:self.n, l - 1])


SobolGenerator.at_most = 2 ** SobolGenerator.log_max - 1
SobolGenerator.max_col = high_bit_pos(SobolGenerator.at_most)
SobolGenerator.good_seeds = numpy.array(
    [0, 0, 1, 3, 5, 8, 11, 15, 19, 23, 27, 31, 35],
    SobolGenerator.int_type_string
)
SobolGenerator.base_v = numpy.zeros(
    (SobolGenerator.n_max, SobolGenerator.log_max),
    SobolGenerator.int_type_string
)
SobolGenerator.base_v[:, 0] = 1
SobolGenerator.base_v[2:SobolGenerator.n_max, 1] = numpy.transpose([
          1, 3, 1, 3, 1, 3, 3, 1,
    3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
    1, 3, 1, 3, 3, 1, 1, 1, 3, 1,
    3, 1, 3, 3, 1, 3, 1, 1, 1, 3
])
SobolGenerator.base_v[3:SobolGenerator.n_max, 2] = numpy.transpose([
             7, 5, 1, 3, 3, 7, 5,
    5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
    5, 3, 7, 1, 7, 5, 1, 3, 7, 7,
    1, 1, 1, 5, 7, 7, 5, 1, 3, 3
])
SobolGenerator.base_v[5:SobolGenerator.n_max, 3] = numpy.transpose([
                        1, 7,  9,  13, 11,
    1, 3,  7,  9,  5,  13, 13, 11, 3,  15,
    5, 3,  15, 7,  9,  13, 9,  1,  11, 7,
    5, 15, 1,  15, 11, 5,  11,  1,  7,  9
])
SobolGenerator.base_v[7:SobolGenerator.n_max, 4] = numpy.transpose([
                                9,  3,  27,
    15, 29, 21, 23, 19, 11, 25, 7,  13, 17,
    1,  25, 29, 3,  31, 11, 5,  23, 27, 19,
    21, 5,  1,  17, 13, 7,  15, 9,  31, 25
])
SobolGenerator.base_v[13:SobolGenerator.n_max, 5] = numpy.transpose([
                37, 33, 7,  5,  11, 39, 63,
    59, 17, 15, 23, 29, 3,  21, 13, 31, 25,
    9,  49, 33, 19, 29, 11, 19, 27, 15, 25
])
SobolGenerator.base_v[19:SobolGenerator.n_max, 6] = numpy.transpose([
                                           13,
    33, 115, 41, 79, 17, 29,  119, 75, 73, 105,
    7,  59,  65, 21, 3,  113, 61,  89, 45, 107
])
SobolGenerator.base_v[37:SobolGenerator.n_max, 7] = numpy.transpose([
    7, 23, 39
])
SobolGenerator.base_v[0, :] = 1
SobolGenerator.poly = numpy.array(
    [
        1, 3, 7, 11, 13, 19, 25, 37, 59, 47,
        61, 55, 41, 67, 97, 91, 109, 103, 115, 131,
        193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
        213, 191, 253, 203, 211, 239, 247, 285, 369, 299
    ],
    SobolGenerator.int_type_string
)
