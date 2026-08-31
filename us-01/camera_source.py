import cv2


def open_camera(camera_index=0):
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    return camera


def read_frame(camera):
    success, frame = camera.read()

    if not success:
        return None

    return frame


def release_camera(camera):
    camera.release()


if __name__ == "__main__":
    camera = open_camera()

    while True:
        frame = read_frame(camera)

        if frame is None:
            print("Could not read frame")
            break

        cv2.imshow("Live Camera Feed", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(camera)
    cv2.destroyAllWindows()