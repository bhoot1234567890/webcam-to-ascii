from PIL import Image, ImageDraw, ImageFont
import string

def get_ink_usage(char, font_path="arial.ttf", font_size=100):
    # Create an image with white background
    img = Image.new('RGB', (font_size, font_size), color='white')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    
    # Draw the character
    d.text((0, 0), char, font=font, fill='black')
    
    # Convert image to grayscale
    img = img.convert("L")
    
    # Count the number of non-white pixels
    non_white_pixels = sum(1 for x in range(img.width) for y in range(img.height) if img.getpixel((x, y)) < 255)
    
    return non_white_pixels

# Get ink usage for all ASCII characters
# Include all printable ASCII characters (from space to tilde)
ascii_characters = ''.join(chr(i) for i in range(32, 127))
ink_usage = {char: get_ink_usage(char) for char in ascii_characters}

# Sort characters by ink usage in descending order
sorted_ink_usage = sorted(ink_usage.items(), key=lambda item: item[1], reverse=True)

# Print the sorted characters
for char, usage in sorted_ink_usage:
    print(f"'{char}': {usage} pixels")

