from flask import Flask, render_template, Response
import cv2
import time
import socket
import io
from bounding_box_overlay_app import frame_to_binary, video_iterator


app = Flask(__name__)
cap = cv2.VideoCapture(0)

# start = time.time()
#
# finish = time.time()
# print(1 / (finish - start))
# start = finish


def process_frame(cap):
    start = time.time()
    for frame in video_iterator(cap):

        ##### Start of decorator #####
        # TODO: Convert this into a decorator that manages frame processing
        frame = cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0))

        finish = time.time()
        fps = 1 / (finish - start)
        start = finish

        # Write FPS to frame
        frame = cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        ##### End of decorator #####
        binary_out = frame_to_binary(frame)
        yield binary_out


@app.route('/')
def index():
   """Video streaming ."""
   return render_template('index.html')


@app.route('/video_feed')
def video_feed():
   """Video streaming route. Put this in the src attribute of an img tag."""
   return Response(
       process_frame(cap),
       mimetype='multipart/x-mixed-replace; boundary=frame'
   )


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
