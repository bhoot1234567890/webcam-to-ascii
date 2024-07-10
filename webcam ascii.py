import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian

# Constants
MAX_WIDTH_CHARS = 150
MAX_HEIGHT_LINES = 200

# Ink usage data
ink_usage = [
    ("@", 3261),
    ("M", 2647),
    ("W", 2614),
    ("Q", 2238),
    ("B", 2226),
    ("N", 2196),
    ("R", 2170),
    ("&", 2115),
    ("%", 2085),
    ("G", 2083),
    ("D", 2070),
    ("O", 1999),
    ("$", 1984),
    ("S", 1925),
    ("E", 1917),
    ("m", 1852),
    ("#", 1843),
    ("H", 1836),
    ("8", 1827),
    ("K", 1816),
    ("6", 1766),
    ("P", 1757),
    ("U", 1757),
    ("w", 1755),
    ("9", 1754),
    ("X", 1712),
    ("A", 1701),
    ("Z", 1699),
    ("C", 1682),
    ("0", 1614),
    ("5", 1600),
    ("d", 1572),
    ("b", 1570),
    ("2", 1509),
    ("g", 1501),
    ("3", 1484),
    ("a", 1481),
    ("e", 1461),
    ("p", 1452),
    ("V", 1445),
    ("q", 1445),
    ("4", 1422),
    ("k", 1397),
    ("h", 1388),
    ("F", 1377),
    ("o", 1299),
    ("Y", 1298),
    ("s", 1278),
    ("n", 1218),
    ("T", 1215),
    ("u", 1213),
    ("z", 1175),
    ("x", 1135),
    ("7", 1108),
    ("J", 1103),
    ("c", 1093),
    ("y", 1063),
    ("?", 1046),
    ("L", 1044),
    ("v", 969),
    ("f", 946),
    ("1", 915),
    ("[", 898),
    ("]", 898),
    ("{", 891),
    ("<", 887),
    (">", 886),
    ("}", 867),
    ("=", 864),
    ("t", 811),
    (")", 781),
    ("(", 780),
    ("+", 774),
    ("I", 720),
    ("l", 720),
    ("j", 710),
    ("r", 706),
    ("^", 694),
    ("!", 666),
    ("|", 656),
    ("i", 620),
    ("_", 610),
    ("/", 609),
    ("\\", 609),
    ("~", 548),
    ("*", 514),
    ('"', 491),
    (";", 279),
    ("-", 252),
    ("'", 247),
    (":", 220),
    (",", 169),
    ("`", 151),
    (".", 110),
    (" ", 0),
]

# Extract characters and ink values
characters = [char for char, _ in ink_usage]
ink_values = np.array([ink for _, ink in ink_usage])

# Normalize ink values
min_ink = ink_values.min()
max_ink = ink_values.max()
normalized_ink_values = 255 * (ink_values - min_ink) / (max_ink - min_ink)


# Sobel edge detection with angle
def sobel_edge_detection_with_angle(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude_8bit = np.uint8(gradient_magnitude)
    angle = np.arctan2(grad_y, grad_x)
    angle_degrees = np.degrees(angle)
    return gradient_magnitude_8bit, angle, angle_degrees


# Difference of Gaussians (DoG)
def apply_dog(image, sigma1=1, sigma2=2):
    if image.ndim == 3:
        blur1 = gaussian(image, sigma=sigma1, channel_axis=-1)
        blur2 = gaussian(image, sigma=sigma2, channel_axis=-1)
    else:
        blur1 = gaussian(image, sigma=sigma1)
        blur2 = gaussian(image, sigma=sigma2)
    dog_image = blur1 - blur2
    dog_image = np.clip(dog_image, 0, 1)
    dog_image = np.uint8(dog_image * 255)
    return dog_image


def sobel_angle_to_char(degree):
    mapping = {
        -135: "/",
        -90: "-",
        -45: "\\",
        45: "/",
        90: "-",
        135: "\\",
        0: " ",
        180: "|",
    }
    degree = degree % 360  # Ensure the degree is within the 0-359 range
    closest = min(mapping, key=lambda x: min(abs(degree - x), 360 - abs(degree - x)))
    return mapping[closest]


# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if frame_pil.width > frame_pil.height:
        new_width = MAX_WIDTH_CHARS
        aspect_ratio = frame_pil.height / frame_pil.width
        new_height = int(new_width * aspect_ratio)
        new_width *= 2
    else:
        new_height = MAX_HEIGHT_LINES
        aspect_ratio = frame_pil.width / frame_pil.height
        new_width = int(new_height * aspect_ratio)
        new_width *= 2
    resized_image = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to grayscale
    gray_image = resized_image.convert("L")
    gray_image_np = np.array(gray_image)

    # Apply DoG
    dog_image = apply_dog(gray_image_np)

    # Apply Sobel edge detection
    sobel_image, gradient_angle, gradient_angle_degrees = (
        sobel_edge_detection_with_angle(dog_image)
    )
    sobel_pil_image = Image.fromarray(sobel_image)

    # Resize Sobel image and gradient angles
    sobel_resized_image = sobel_pil_image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )
    sobel_pixels = np.array(sobel_resized_image)
    resized_gradient_angles = cv2.resize(
        gradient_angle_degrees, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    # Create ASCII art from Sobel edge detection results
    sobel_ascii_art_list = []
    for i, row in enumerate(sobel_pixels):  # Use enumerate to correctly update 'i'
        sobel_ascii_row = []
        for j, pixel in enumerate(row):
            idx = (np.abs(normalized_ink_values - pixel)).argmin()
            if idx < 90:
                sobel_ascii_row.append(
                    sobel_angle_to_char(resized_gradient_angles[i, j])
                )
            else:
                sobel_ascii_row.append(" ")
        sobel_ascii_art_list.append(sobel_ascii_row)

    # Convert grayscale image to ASCII art
    ascii_art_list = []
    for row in gray_image_np:
        ascii_row = []
        for pixel in row:
            idx = (np.abs(normalized_ink_values - pixel)).argmin()
            ascii_row.append(characters[idx])
        ascii_art_list.append(ascii_row)

    # Create final combined ASCII art
    combined_ascii_art_list = []
    for i in range(len(ascii_art_list)):
        combined_row = [
            (
                ascii_art_list[i][j]
                if sobel_ascii_art_list[i][j] == " "
                else sobel_ascii_art_list[i][j]
            )
            for j in range(len(ascii_art_list[i]))
        ]
        combined_ascii_art_list.append(combined_row)

    # Print the combined ASCII art
    for row in combined_ascii_art_list:
        print("".join(row))

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
