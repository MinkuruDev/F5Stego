import os
import sys
import argparse
import jpeg
import logging
from pathlib import Path

logger = logging.getLogger("f5")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def embed_message(input_path: Path, message: str, password: str, output_path: Path, keep_temp: bool):
    """
    Embed a message into a JPEG image using F5 steganography.

    Args:
        input_path (Path): Path to the input JPEG image.
        message (str): Message to embed.
        password (str): Password for embedding.
        output_path (Path): Path to save the output JPEG image.
    """
    temp_path = Path(f"temp/{input_path.stem}_temp.jpg")
    jpeg.convert_to_jpeg(str(input_path), str(temp_path))
    coefficients, meta, jpeg_image = jpeg.flatten_dct(str(temp_path))
    modified_coefficients = jpeg.embed_message(coefficients, message, password)
    jpeg_image.Y, jpeg_image.Cb, jpeg_image.Cr = \
        jpeg.reconstruct_dct(modified_coefficients, meta)
    jpeg_image.write_dct(str(output_path))
    if not keep_temp:
        os.remove(temp_path)

def extract_message(input_path: Path, password: str) -> str:
    """
    Extract a message from a JPEG image using F5 steganography.

    Args:
        input_path (Path): Path to the input JPEG image.
        password (str): Password for extraction.

    Returns:
        str: The extracted message.
    """

    coefficients, _, _ = jpeg.flatten_dct(str(input_path))
    extracted_message = jpeg.extract_message(coefficients, password)
    return extracted_message

argparser = argparse.ArgumentParser(description="F5 Steganography Tool")
argparser.add_argument('command', choices=['embed', 'extract', 'e', 'x'], help='Command to execute')
argparser.add_argument('-i', '--input', type=Path, help='Input JPEG image path', required=True)
argparser.add_argument('-o', '--output', type=Path, help='Output JPEG image path (for embed only)')
argparser.add_argument('-m', '--message', type=str, help='Message to embed into the image (for embed only)')
argparser.add_argument('-p', '--password', type=str, help='Password for embedding/extracting', default='')
argparser.add_argument('-k', '--keep', action='store_true', help='Keep temporary files')

if __name__ == "__main__":
    args = argparser.parse_args()
    if args.command == 'embed' or args.command == 'e':
        embed_message(args.input, args.message, args.password, args.output, args.keep)
        logger.info(f"Message embedded successfully into {args.output}")
    elif args.command == 'extract' or args.command == 'x':
        message = extract_message(args.input, args.password)
        print(f"Extracted message: {message}")
    else:
        argparser.print_help()
