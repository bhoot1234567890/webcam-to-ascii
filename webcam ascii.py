import cv2
from PIL import Image

from ascii_core import render_image

MAX_WIDTH_CHARS = 150
MAX_HEIGHT_LINES = 200


def main():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = render_image(
                image,
                max_width_chars=MAX_WIDTH_CHARS,
                max_height_lines=MAX_HEIGHT_LINES,
            )
            print(result["text"])
            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
