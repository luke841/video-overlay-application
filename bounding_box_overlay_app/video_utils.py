import cv2


def video_iterator(cap):
    while True:
        rval, frame = cap.read()
        if rval is False:
            return None
        yield frame


def frame_to_binary(frame):
    # TODO: Don't write to file
    cv2.imwrite('pic.jpg', frame)
    return (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n'
    )