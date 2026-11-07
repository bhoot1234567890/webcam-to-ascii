# Python terminal viewer

The original clack surface: live ASCII frames printed to the terminal, with the raw feed in an OpenCV window. It shares the measured ink-usage ramp and the numerical oracle (`ascii_core.py`) with the browser app.

**Prerequisites:** Python 3.9+, a working webcam, and (on macOS) Terminal granted camera access in *System Settings → Privacy & Security → Camera*.

Clone the repository:

```bash
git clone <your-repo-url>/webcam-to-ascii.git
cd webcam-to-ascii
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> **Note on `skimage`:** `requirements.txt` lists `skimage>=0.0`, the legacy alias package for **scikit-image**. If pip cannot resolve it, install the real package directly: `pip install scikit-image`.

The full dependency set is OpenCV (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`), NumPy, Pillow, scikit-image, and Matplotlib.

## 🚀 Usage

Run the main viewer (note the space in the filename — quotes are required):

```bash
python "webcam ascii.py"
```

The terminal fills with the live ASCII render; a separate *Webcam Feed* window shows the source frame. **Press `q`** (with the preview window focused) to stop and release the camera.

Using a different camera? Find its index first, then point the viewer at it:

```bash
python "list webcams.py"
```

```python
# In "webcam ascii.py", change the device index on this line:
cap = cv2.VideoCapture(0)   # ← 0 is the default webcam
```

### What you'll see

Each frame is printed as a complete block of glyphs. Landscape sources use twice the width setting; portrait and square sources use the height setting and derive doubled columns from their aspect ratio. Flat regions range from space (dark) toward `@` (bright). The terminal scrolls continuously.
