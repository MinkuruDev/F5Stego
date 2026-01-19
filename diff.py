import argparse
import sys
from PIL import Image, ImageChops

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    parser = argparse.ArgumentParser(
        description="Compare two images of the same size using subtraction"
    )
    parser.add_argument("image1", help="First image path")
    parser.add_argument("image2", help="Second image path")
    parser.add_argument("-m", "--maximize", action="store_true", help="Maximize the difference output")
    parser.add_argument(
        "-o", "--output",
        help="Path to save the subtracted image (optional)"
    )

    args = parser.parse_args()

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    if img1.size != img2.size:
        print("Error: Images must have the same dimensions", file=sys.stderr)
        sys.exit(1)

    diff = ImageChops.difference(img1, img2)
    if args.maximize:
        diff = diff.point(lambda p: 255 if p > 0 else 0)

    if args.output:
        diff.save(args.output)
        print(f"Subtracted image saved to: {args.output}")

    diff.show()

if __name__ == "__main__":
    main()
