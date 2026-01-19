from typing import Tuple
import util
import jpeglib
import hashlib
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("f5")

def convert_to_jpeg(input_path, output_path):
    """
    Convert an image to JPEG format. 

    Parameters:
    - input_path: str, path to the input image file.
    - output_path: str, path to save the converted JPEG image.
    """
    # Open the input image
    img = Image.open(input_path)

    # Convert image to RGB if it's in a different mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if not output_path:
        output_path = input_path.rsplit('.', 1)[0] + '.jpg'

    # Save the image in JPEG format
    img.save(output_path, 'JPEG')

def flatten_dct(jpeg_path: str) -> Tuple[np.ndarray, dict, jpeglib.DCTJPEG]:
    """
    Flatten the DCT coefficients from a JPEG image into a 1D numpy array.
    Parameters:
    - jpeg_path: str, path to the JPEG image file.
    Returns:
    - flat: np.ndarray, 1D array of flattened DCT coefficients.
    - meta: dict, metadata including shapes of Y and C components.
    - jpeg: jpeglib.JPEGImage, the JPEG image object.
    """

    jpeg = jpeglib.read_dct(jpeg_path)
    flat = []
    hy, wy, _, _ = jpeg.Y.shape
    hc, wc, _, _ = jpeg.Cb.shape

    for i in range(hy):
        for j in range(wy):
            block = jpeg.Y[i, j]
            flat.extend(block.flatten().tolist())

    for i in range(hc):
        for j in range(wc):
            block = jpeg.Cb[i, j]
            flat.extend(block.flatten().tolist())

    for i in range(hc):
        for j in range(wc):
            block = jpeg.Cr[i, j]
            flat.extend(block.flatten().tolist())

    meta = {
        'Y_shape': jpeg.Y.shape,
        'C_shape': jpeg.Cb.shape,
    }
    return np.array(flat, dtype=np.int16), meta, jpeg

def reconstruct_dct(flat: np.ndarray, meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the Y, Cb, Cr DCT coefficient blocks from a flattened array.
    Parameters:
    - flat: np.ndarray, 1D array of flattened DCT coefficients.
    - meta: dict, metadata including shapes of Y and C components.
    Returns:
    - Y: np.ndarray, reconstructed Y component blocks.
    - Cb: np.ndarray, reconstructed Cb component blocks.
    - Cr: np.ndarray, reconstructed Cr component blocks.
    """

    hy, wy, _, _ = meta['Y_shape']
    hc, wc, _, _ = meta['C_shape']

    index = 0
    Y_blocks = []
    for i in range(hy):
        row_blocks = []
        for j in range(wy):
            block = flat[index:index+64].reshape((8, 8))
            row_blocks.append(block)
            index += 64
        Y_blocks.append(row_blocks)
    Y = np.array(Y_blocks)

    Cb_blocks = []
    for i in range(hc):
        row_blocks = []
        for j in range(wc):
            block = flat[index:index+64].reshape((8, 8))
            row_blocks.append(block)
            index += 64
        Cb_blocks.append(row_blocks)
    Cb = np.array(Cb_blocks)

    Cr_blocks = []
    for i in range(hc):
        row_blocks = []
        for j in range(wc):
            block = flat[index:index+64].reshape((8, 8))
            row_blocks.append(block)
            index += 64
        Cr_blocks.append(row_blocks)
    Cr = np.array(Cr_blocks)

    return Y, Cb, Cr

def read_n_bits(block: np.ndarray, n: int) -> int:
    """
    Read n bits from the 2^n - 1 coefficients in the block.
    Parameters:
    - block: np.ndarray, the block of JPEG coefficients.
    - n: int, number of bits to read.

    Returns:
    - int, the integer value represented by the n bits.
    """
    if n == 1 and len(block) == 1:
        coeff = block[0]
        lsb = (1-coeff if coeff < 0 else coeff) % 2
        return lsb
    W = 2**n - 1
    # W = (1 << n) - 1
    # W = len(block)
    if len(block) != W:
        print(len(block), W, n)
        raise ValueError("Block size does not match expected size.")

    value = util.create_array(0, W, 1, dtype=int)
    for i in range(W):
        coeff = block[i]
        lsb = (1-coeff if coeff < 0 else coeff) % 2
        value[i][0] = lsb

    represent_matrix = util.parity_matrix(n) @ value
    return util.bit_array_to_int(represent_matrix % 2)

def write_n_bits(coefficients: np.ndarray, indicies: np.ndarray, n: int, value: int) -> int:
    """
    Write n bits into the 2^n - 1 coefficients at the given indices.
    Parameters:
    - coefficients: np.ndarray, the flat array of JPEG coefficients.
    - indicies: np.ndarray, the indices where bits will be written.
    - n: int, number of bits to write.
    - value: int, the integer value to write as bits.
    Returns:
    - int, index of the modified coefficient, or -1 if no modification was needed.
    """
    if n == 1 and len(indicies) == 1:
        target_index = indicies[0]
        coeff = coefficients[target_index]
        lsb = (1-coeff if coeff < 0 else coeff) % 2
        if lsb != (value & 1):
            coefficients[target_index] += 1 if coeff < 0 else -1
            return target_index
        else:
            return -1  # No modification needed

    W = (1 << n) - 1
    if len(indicies) != W:
        raise ValueError("Number of indices does not match expected size.")

    parity = util.parity_matrix(n)
    current_bits = util.create_array(0, W, 1, dtype=int)
    for i in range(W):
        coeff = coefficients[indicies[i]]
        lsb = (1-coeff if coeff < 0 else coeff) % 2
        current_bits[i][0] = lsb

    current_value = (parity @ current_bits) % 2
    current_value = util.bit_array_to_int(current_value)
    diff = current_value ^ value
    if diff == 0:
        return -1  # No modification needed
    
    diff_index = diff - 1
    target_index = indicies[diff_index]
    coefficients[target_index] += 1 if coefficients[target_index] < 0 else -1
    return target_index

def embed_message(coefficients: np.ndarray, message: str, key: str, w: int = 0) -> np.ndarray:
    """
    Embed a secret message into JPEG coefficients using F5 algorithm.
    Parameters:
    - coefficients: np.ndarray, the flat array of JPEG quantized DCT coefficients.
    - message: str, the secret message to embed.
    - key: str, the key used for permutation and embedding.
    - w: the number of bits to embed at a time. If 0, automatically determine based on capacity.
    Returns:
    - np.ndarray, the modified coefficients with the embedded message.
    """
    coeff_count = len(coefficients)
    logger.info(f"Total coefficients: {coeff_count}")
    _one, _zero = 0, 0
    for coeff in coefficients:
        if coeff == 1 or coeff == -1:
            _one += 1
        elif coeff == 0:
            _zero += 1
    expected_capacity = coeff_count - coeff_count//64 - _zero - np.floor(0.51 * _one)
    logger.info(f"Expected capacity: {expected_capacity} bits")
    embed_length = len(message) * 8 + 32  # 32 bits for message length header
    if expected_capacity < embed_length:
        raise ValueError("Not enough capacity to embed the message.")

    random = util.PythonF5Random(hashlib.md5(key.encode()).hexdigest())
    permutation = util.Permutation(coeff_count, random)
    filter_func = lambda index: index % 64 != 0 and coefficients[index] != 0
    filtered_collection = util.FilteredCollection(permutation.shuffled, filter_func)

    if w == 0:
        for bits in [1,2,4,8]:
            W = (1 << bits) - 1
            if expected_capacity / W >= embed_length / bits:
                w = bits
        logger.info(f"Automatically selected w: {w}")

    logger.info(f"Embedding message length to first 24 bits, using 1 coefficient per bit.")
    message_length = len(message) * 8
    for i in range(24):
        bit = (message_length >> i) & 1
        indices = filtered_collection.offer(1)
        while True:
            index = write_n_bits(coefficients, indices, 1, bit)
            if index != -1 and coefficients[index] == 0:
                # shrikage happened, need to get a new index
                indices = filtered_collection.offer(1)
            else:
                break

    logger.info(f"Embedding w={w} to next 8 bits, using 1 coefficient per bit.")
    for i in range(8):
        bit = (w >> i) & 1
        indices = filtered_collection.offer(1)
        while True:
            index = write_n_bits(coefficients, indices, 1, bit)
            if index != -1 and coefficients[index] == 0:
                # shrikage happened, need to get a new index
                indices = filtered_collection.offer(1)
            else:
                break

    logger.info(f"Embedding message using w={w} bits per group.")
    bits = util.string_to_bit_array(message)
    for i in range(len(bits)):
        xor_value = random.get_next_value(2)
        bits[i] ^= xor_value
    total_bits = len(bits)
    bit_index = 0
    W = (1 << w) - 1
    while total_bits > 0:
        bits_to_write = min(w, total_bits)
        value = util.bit_array_to_int(bits[bit_index:bit_index + bits_to_write])
        # print("Writing value:", value)
        indices = filtered_collection.offer(W)
        while True:
            index = write_n_bits(coefficients, indices, bits_to_write, value)
            if index != -1 and coefficients[index] == 0:
                # shrikage happened, need to get a new index
                indices.remove(index)
                indices.append(filtered_collection.offer(1)[0])
            else:
                break
        bit_index += bits_to_write
        total_bits -= bits_to_write

    return coefficients

def extract_message(coefficients: np.ndarray, key: str) -> str:
    """
    Extract a secret message from JPEG coefficients using F5 algorithm.
    Parameters:
    - coefficients: np.ndarray, the flat array of JPEG quantized DCT coefficients.
    - key: str, the key used for permutation and embedding.
    Returns:
    - str, the extracted secret message.
    """
    coeff_count = len(coefficients)
    random = util.PythonF5Random(hashlib.md5(key.encode()).hexdigest())
    permutation = util.Permutation(coeff_count, random)
    filter_func = lambda index: index % 64 != 0 and coefficients[index] != 0
    filtered_collection = util.FilteredCollection(permutation.shuffled, filter_func)

    # Extract message length from first 24 bits
    message_length = 0
    for i in range(24):
        indices = filtered_collection.offer(1)
        bit = read_n_bits(np.array([coefficients[indices[0]]]), 1)
        message_length |= (bit << i)

    # Extract w from next 8 bits
    w = 0
    for i in range(8):
        indices = filtered_collection.offer(1)
        bit = read_n_bits(np.array([coefficients[indices[0]]]), 1)
        w |= (bit << i)

    logger.info(f"Extracted message length: {message_length} bits, w: {w}")

    bits = []
    total_bits = message_length
    W = (1 << w) - 1
    while total_bits > 0:
        bits_to_read = min(w, total_bits)
        indices = filtered_collection.offer(W)
        value = read_n_bits(np.array([coefficients[i] for i in indices]), bits_to_read)
        # print("Read value:", value)
        for b in util.int_to_bit_array(value, bits_to_read):
            bits.append(b)
        total_bits -= bits_to_read

    for i in range(len(bits)):
        xor_value = random.get_next_value(2)
        bits[i] ^= xor_value
    return util.bit_array_to_string(np.array(bits))

if __name__ == "__main__":
    image_path = 'AE.jpg'
