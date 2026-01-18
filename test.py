import util
import jpeg
import numpy as np

def test_huffman_decoder():
    image_path = 'AE.jpg'
    coefficients, meta, jpeg_image = jpeg.flatten_dct(image_path)
    print(len(coefficients))

def test_permutation():
    size = 15
    f5random = util.PythonF5Random(36)
    permutation = util.Permutation(size, f5random)
    for i in range(size):
        print(permutation.get_shuffled(i))

def test_filtered_collection():
    image_path = 'AE.jpg'
    coefficients, meta, jpeg_image = jpeg.flatten_dct(image_path)
    f5ranom = util.PythonF5Random(36)
    permutation = util.Permutation(len(coefficients), f5ranom)
    print(len(coefficients))
    filter_func = lambda index: index % 64 != 0 and coefficients[index] != 0
    filtered_collection = util.FilteredCollection(permutation.shuffled, filter_func)
    try:
        selected_indices = filtered_collection.offer(15)
        print(selected_indices)
        values = [coefficients[i] for i in selected_indices]
        print(values)
    except util.FilteredCollection.ListNotEnough as e:
        print(f"Not enough elements: {e}")

def test_read_write():
    image_path = 'AE.jpg'
    coefficients, meta, jpeg_image = jpeg.flatten_dct(image_path)
    f5ranom = util.PythonF5Random(36)
    permutation = util.Permutation(len(coefficients), f5ranom)
    filter_func = lambda index: index % 64 != 0 and coefficients[index] != 0
    filtered_collection = util.FilteredCollection(permutation.shuffled, filter_func)
    try:
        selected_indices = filtered_collection.offer(15)
        print("Selected indices:", selected_indices)
        block = np.array([coefficients[i] for i in selected_indices])
        print("Block coefficients:", block)
        n = 4
        value = jpeg.read_n_bits(block, n)
        print(f"Read value: {value}")
        write_value = 0b1001
        modified_index = jpeg.write_n_bits(coefficients, selected_indices, n, write_value)
        if modified_index != -1:
            print(f"Modified coefficient at index: {modified_index}, new value: {coefficients[modified_index]}")
        else:
            print("No modification needed.")

        block_after = np.array([coefficients[i] for i in selected_indices])
        print("Block coefficients after writing:", block_after)
        value_after = jpeg.read_n_bits(block_after, n)
        print(f"Read value after writing: {value_after}")
        
    except util.FilteredCollection.ListNotEnough as e:
        print(f"Not enough elements: {e}")

def test_embed_message():
    image_path = 'AE.jpg'
    coefficients, meta, jpeg_image = jpeg.flatten_dct(image_path)
    message = "Hello, World!"
    key = "secret_key"
    modified_coefficients = jpeg.embed_message(coefficients, message, key)
    jpeg_image.Y, jpeg_image.Cb, jpeg_image.Cr = \
        jpeg.reconstruct_dct(modified_coefficients, meta)
    output_path = 'AE_stego.jpg'
    jpeg_image.write_dct(output_path)
    print("Message embedded successfully.")
    
    coefficients_extracted, _, _ = jpeg.flatten_dct(output_path)
    extracted_message = jpeg.extract_message(coefficients_extracted, key)
    print("Extracted message:", extracted_message)

if __name__ == "__main__":
    # test_permutation()
    # test_filtered_collection()
    # test_read_write()
    test_embed_message()