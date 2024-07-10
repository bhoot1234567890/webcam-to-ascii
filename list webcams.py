import cv2


def list_available_cameras(max_test_index=10):
    available_cameras = []
    for i in range(max_test_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use CAP_DSHOW for DirectShow
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        else:
            break  # Stop if an index is not available
    return available_cameras


available_cameras = list_available_cameras()
print("Available Cameras:", available_cameras)
