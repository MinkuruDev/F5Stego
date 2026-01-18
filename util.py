import random
import copy

import numpy as np

def parity_matrix(bits):
    """
    Create a parity-check matrix for a given number of bits.
    Parameters:
    - bits: int, the number of bits in the codeword.
    Returns:
    - np.ndarray, the parity-check matrix.
    """
    W = (1 << bits) - 1
    H = np.array([[0]*W for _ in range(bits)])
    for j in range(1, W+1):
        for i in range(bits):
            H[i][j-1] = (j >> i) & 1
    return H

def bit_array_to_int(bit_array: np.ndarray) -> int:
    """
    Convert a bit array (1D numpy array) to an integer.
    Parameters:
    - bit_array: np.ndarray, 1D array of bits (0s and 1s) or 2D array with shape (n, 1).
    Returns:
    - int, the integer representation of the bit array.
    """
    result = 0
    bit_array = bit_array.flatten() # Ensure it's 1D
    for i in range(len(bit_array)):
        bit = bit_array[i]
        if isinstance(bit, np.ndarray):
            bit = bit[0]
        result |= (bit & 1) << i
    return result

def int_to_bit_array(value: int, length: int) -> np.ndarray:
    """
    Convert an integer to a bit array (1D numpy array) of a specified length.
    Parameters:
    - value: int, the integer to convert.
    - length: int, the desired length of the bit array.
    Returns:
    - np.ndarray, 1D array of bits (0s and 1s).
    """
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bits[i] = (value >> i) & 1
    return bits

def create_array(default_value=None, *dims, dtype=None, copy_default=False):
    """
    Create a NumPy array with shape given by dims, filled with default_value.

    - If no dims are provided, returns a 0-d numpy array: np.array(default_value, dtype=dtype).
    - dims may contain zeros (resulting array will have size 0 along that axis).
    - If copy_default is True and dtype is object (or inferred to object), each element
      will receive a deep copy of default_value so mutable defaults don't share the same instance.
    """
    # Validate dims
    if any(d < 0 for d in dims):
        raise ValueError("dimensions must be non-negative")

    if not dims:
        # Return 0-d array (scalar array)
        return np.array(default_value, dtype=dtype)

    shape = tuple(int(d) for d in dims)

    # Fast path for numeric / immutable defaults: np.full is efficient.
    if not copy_default:
        # Let numpy infer dtype if dtype is None
        try:
            return np.full(shape, default_value, dtype=dtype)
        except Exception:
            # In case np.full fails (e.g. for certain object defaults), fall back to object-array fill
            pass

    # If we need independent mutable copies or np.full cannot be used,
    # create an object array and fill with deep copies.
    arr = np.empty(shape, dtype=object)
    if copy_default:
        for idx in np.ndindex(shape):
            arr[idx] = copy.deepcopy(default_value)
    else:
        # Fill with same object reference (matches original function semantics)
        arr.fill(default_value)
    # If user requested a specific dtype other than object, try to cast
    if dtype is not None and dtype is not object:
        try:
            return arr.astype(dtype)
        except Exception:
            # If casting fails, keep object array
            pass
    return arr

def string_to_bit_array(s: str) -> np.ndarray:
    """
    Convert a string into a numpy array of bits (0s and 1s).
    Each character is represented by its ASCII value in 8 bits.
    Parameters:
    - s: str, the input string.
    Returns:
    - np.ndarray, 1D array of bits representing the string.
    """
    bit_list = []
    for char in s:
        ascii_val = ord(char)
        for i in range(7, -1, -1):
            bit = (ascii_val >> i) & 1
            bit_list.append(bit)
    return np.array(bit_list, dtype=np.uint8)

def bit_array_to_string(bit_array: np.ndarray) -> str:
    """
    Convert a numpy array of bits (0s and 1s) back into a string.
    Each group of 8 bits is converted back to its corresponding ASCII character.
    Parameters:
    - bit_array: np.ndarray, 1D array of bits.
    Returns:
    - str, the reconstructed string.
    """
    if len(bit_array) % 8 != 0:
        raise ValueError("Length of bit_array must be a multiple of 8.")
    
    chars = []
    for i in range(0, len(bit_array), 8):
        byte = 0
        for j in range(8):
            bit = bit_array[i + j]
            byte |= (bit & 1) << (7 - j)
        chars.append(chr(byte))
    return ''.join(chars)

class BreakException(Exception):
    def __init__(self):
        super(BreakException, self).__init__('break to outside loop')

class EmbedData(object):
    def __init__(self, data):
        self._data = data
        self.now = 0
        self.len = len(data)

    def __len__(self):
        return self.len

    def read(self):
        self.now += 1
        if self.now > self.len:
            return 0
        #print(self._data[self.now - 1])
        #return ord(self._data[self.now - 1])
        return self._data[self.now - 1]

    def available(self):
        return self.len - self.now

class Permutation(object):
    def __init__(self, size, f5random):
        self.shuffled = list(range(size))
        max_random = size
        for i in range(size):
            random_index = f5random.get_next_value(max_random)
            max_random -= 1
            tmp = self.shuffled[random_index]
            self.shuffled[random_index] = self.shuffled[max_random]
            self.shuffled[max_random] = tmp

    def get_shuffled(self, i):
        return self.shuffled[i]

class FilteredCollection(object):
    def __init__(self, l, filter_func):
        self.iterator = l
        self.filter_func = filter_func
        self.now = 0

    def offer(self, count=1):
        result = []
        while count:
            while self.now < len(self.iterator) and not self.filter_func(self.iterator[self.now]):
                self.now += 1
            if self.now < len(self.iterator):
                count -= 1
                result.append(self.iterator[self.now])
                self.now += 1
            else:
                raise FilteredCollection.ListNotEnough(count)
        return result

    def reset(self):
        self.now = 0

    class ListNotEnough(Exception):
        def __init__(self, count):
            super(FilteredCollection.ListNotEnough, self).__init__('sorry list is not enough to provide %d elements' % count)

class F5Random(object):
    def get_next_byte(self):
        raise Exception('not implemented')

    def get_next_value(self, max_value):
        ret_val = self.get_next_byte() | self.get_next_byte() << 8 | \
                self.get_next_byte() << 16 | self.get_next_byte() << 24
        ret_val %= max_value
        if ret_val < 0: ret_val += max_value
        return ret_val
    
class PythonF5Random(F5Random):
    def __init__(self, password):
        random.seed(password)

    def get_next_byte(self):
        return random.randint(-128, 127)
