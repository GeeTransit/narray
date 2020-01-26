# narray.py
#
# Array functions

from collections import deque, namedtuple
from copy import deepcopy
from functools import reduce
from itertools import product
from operator import getitem, __add__, __sub__, __mul__, __truediv__


def shape_of(array, *, strict=False):
    """
    Return the shape of array. (sizes of each dimension)
    """
    shape = []
    layer = array

    while True:
        if not isinstance(layer, (tuple, list)):
            break
        size = len(layer)
        shape.append(size)
        if not size:
            break
        layer = layer[0]

    if strict:
        layers = deque(
            (str(i), sub)
            for i, sub in enumerate(array)
        )

        for size in shape[1:]:
            for _ in range(len(layers)):
                indices, layer = layers.popleft()
                if not isinstance(layer, (tuple, list)):
                    raise ValueError(
                        f"array is not uniform: "
                        f"not isinstance(array[{indices}], (tuple, list)) ({layer})"
                    )
                if len(layer) != size:
                    raise ValueError(
                        f"array is not uniform: "
                        f"len(array[{indices}]) ({layer}) != {size}"
                    )
                layers.extend(
                    (indices + f", {i}", sub)
                    for i, sub in enumerate(layer)
                )

        for _ in range(len(layers)):
            indices, layer = layers.popleft()
            if isinstance(layer, (tuple, list)):
                raise ValueError(
                    f"array is not uniform: "
                    f"isinstance(array[{indices}], (tuple, list)) ({layer})"
                )

    return tuple(shape)


def getitems(array, values):
    """
    Equivalent to array[*values]
    """
    return reduce(getitem, values, array)

def setitems(array, values, item):
    """
    Equivalent to array[*values] = item
    """
    layer = getitems(array, values[:-1])
    layer[values[-1] if values else 0] = item


def empty(shape, num=0):
    """
    Return an array of shape filled with num.
    """
    data = [num]*(shape[-1] if shape else 1)
    for dim in shape[-2::-1]:
        data = [deepcopy(data) for _ in range(dim)]
    return data

def zeros(shape):
    """
    Return an zero-filled array of shape.
    """
    return empty(shape, 0)

def eye(size, dims=2):
    """
    Return an identity array with size and dims.
    """
    array = zeros([size]*dims)
    for i in range(size):
        setitems(array, [i]*dims, 1)
    return array


def trp(array):
    """
    Return array transposed.
    """
    sp = shape_of(array)
    if len(sp) != 2:
        raise ValueError(f"array doesn't have two dimensions: {sp}")
    return [*map(list, zip(*array))]

def isq(array):
    """
    Return if array has equal sized dimensions.
    """
    ap = shape_of(array)
    return all(i == ap[0] for i in ap)


# TODO: change into iterative (remove recursion)
def minor(array, exclude):
    """
    Return the minor of array using exclude.
    """
    if len(exclude) > len(shape_of(array)):
        raise ValueError(f"exclude has too many indices: {len(exclude)} > {len(shape_of(array))}")
    if not exclude:
        return array
    if len(exclude) == 1:
        if exclude[0] is None:
            return array
        return array[:exclude[0]] + array[exclude[0] + 1:]

    first, *index = exclude
    return [*map(lambda row: minor(row, index), minor(array, [first]))]


def rre(array):
    """
    Return the reduced row echelon form of array.
    """
    sp = shape_of(array)
    if len(sp) != 2:
        raise ValueError(f"array doesn't have two dimensions: {sp}")

    copy = deepcopy(array)
    lead = 0

    for r in range(sp[0]):
        if lead >= sp[1]:
            break

        i = r
        while not copy[i][lead]:
            i += 1
            if i != sp[0]:
                continue
            i = r
            lead += 1
            if sp[1] == lead:
                break

        copy[i], copy[r] = copy[r], copy[i]

        lv = copy[r][lead]
        for n, mrx in enumerate(copy[r]):
            copy[r][n] = mrx/lv

        for i in range(sp[0]):
            if i != r:
                lv = copy[i][lead]
                for n, (rv, iv) in enumerate(zip(copy[r], copy[i])):
                    copy[i][n] = iv - lv*rv

        lead += 1

    return copy


gauss_return = namedtuple("gauss_return", "det a b")
def gauss(a, b):
    """
    Return the determinant of a and a matrix x that when matrix
    multiplied with a gives b.

    If b is an identity matrix, a will be its inverse.
    """
    ap = shape_of(a)
    bp = shape_of(b)

    if len(ap) != 2:
        raise ValueError(f"a doesn't have two dimensions: {ap}")
    if len(bp) != 2:
        raise ValueError(f"b doesn't have two dimensions: {bp}")

    a = deepcopy(a)
    b = deepcopy(b)
    n = ap[0]
    p = bp[1]
    det = 1

    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j

        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det

        for j in range(i + 1, n):
            t = a[j][i] / a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= t*a[i][k]
            for k in range(p):
                b[j][k] -= t*b[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            t = a[i][j]
            for k in range(p):
                b[i][k] -= t*b[j][k]

        t = 1 / a[i][i]
        det *= a[i][i]
        for j in range(p):
            b[i][j] *= t

    return gauss_return(det, a, b)

def inv(array):
    """
    Return the inverse of array.
    """
    sp = shape_of(array)
    return gauss(array, eye(sp[0])).b

def det(array):
    """
    Return the determinant of array.
    """
    sp = shape_of(array)
    return gauss(array, zeros(sp)).det


def join(a, b, *more, key):
    """
    Return an array where each element is the result of calling key on
    an element of a and b (and more if specified).
    """
    if more:
        r = join(a, b, key=key)
        for c in more:
            r = join(r, c, key=key)
        return r

    ap = shape_of(a)
    bp = shape_of(b)

    if not ap and not bp:
        return key(a, b)
    elif not ap or not bp:
        s, sp = (b, ap) if ap else (a, bp)
        c = empty(sp, s)
        return join(a, c, key=key) if ap else join(c, b, key=key)

    if ap != bp:
        raise ValueError(f"shapes {ap} and {bp} not aligned: {ap} != {bp}")

    result = zeros(ap)
    for i in product(*map(range, ap)):
        j, k = getitems(a, i), getitems(b, i)
        setitems(result, i, key(j, k))
    return result

def add(a, b, *more):
    """
    Return a+b (+more if specified).
    """
    return join(a, b, *more, key=__add__)

def sub(a, b, *more):
    """
    Return a-b (-more if specified).
    """
    return join(a, b, *more, key=__sub__)

def mul(a, b, *more):
    """
    Return a*b (*more if specified).
    """
    return join(a, b, *more, key=__mul__)

def div(a, b, *more):
    """
    Return a/b (/more if specified).
    """
    return join(a, b, *more, key=__truediv__)


def dot(a, b, *more):
    """
    Return a@b (@more if specified).
    """
    if more:
        r = dot(a, b)
        for c in more:
            r = dot(r, c)
        return r

    ap = shape_of(a)
    bp = shape_of(b)

    if not ap or not bp:
        return multiply(a, b)
    if ap[-1] != bp[0]:
        raise ValueError(f"shapes {ap} and {bp} not aligned: {ap[-1]} (dim {len(ap) - 1}) != {bp[0]} (dim 0)")

    ap, n, bp = ap[:-1], bp[0], bp[1:]
    shape = (*ap, *bp)
    if not shape:  # faster case
        return sum(map(__mul__, a, b))

    result = zeros(shape)
    for i in product(*map(range, ap)):
        for j in product(*map(range, bp)):
            setitems(result, [*i, *j], sum(
                getitems(a, [*i, k]) * getitems(b, [k, *j])
                for k in range(n)
            ))
    return result


transposed = trp
is_quadratic = isq
rrechelon = rre
inverse = inv
determinant = det
subtract = sub
multiply = mul
divide = div


if __name__ == "__main__":
    a = eye(3)
    assert isq(a)
    assert inv(a) == a
    assert det(a) == 1
    assert rre(a) == a

    b = [[1, 2, 3]]
    assert dot(b, a) == b
    assert dot(b, a, a) == b

    c = [[2],
         [4],
         [6]]
    assert dot(a, c) == c
    assert dot(a, a, c) == c

    d = [[2, 4, 6]]
    assert trp(c) == d
    assert minor(d, [None, None]) == d

    assert add(b, b) == d
    assert sub(b, mul(b, -1)) == d
    assert mul(b, 2) == d
    assert div(b, 1/2) == d
