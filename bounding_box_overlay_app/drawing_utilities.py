import numpy as np
import cv2


def draw_bounding_box_on_image(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    class_name: str,
    confidence: float,
    colour: tuple[int, int, int],
) -> np.ndarray:
    """
    Draws a bounding box on the image with the defined confidence and colour

    Parameters
    ----------
    image:
        Image to draw the boxes on
    x0:
        Top left x coordinate
    y0:
        Top left y coordinate
    x1:
        bottom right x coordinate
    y1:
        bottom right y coordinate
    class_name:
        The name of the class, used to draw on the box
    confidence:
        The confidence of the detection, will be drawn on the box
    colour:
        The box colour to draw on the frame
    """
    thickness = 4
    top_left = (x0, y0)
    bottom_right = (x1, y1)

    # Determine top-left and bottom-right corners
    cv2.rectangle(image, top_left, bottom_right, color=colour, thickness=thickness)

    # Draw label background
    top_left = (int(x0 - thickness / 2), max(y0 - 30, 0))
    bottom_right = (x0 + 80, y0)
    cv2.rectangle(image, top_left, bottom_right, color=colour, thickness=-1)

    # draw label name
    origin = (x0 + 4, max(y0 - 10, 0))
    cv2.putText(
        image,
        text=f"{class_name} {confidence:.0%}",
        org=origin,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.3,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return image


def draw_yolo_predictions(frame, predictions, colour: list[int, int, int] = [0, 0, 255]):
    for x0, y0, x1, y1, score, class_id in predictions.boxes.data:
        frame = draw_bounding_box_on_image(
            image=frame,
            x0=int(x0),
            y0=int(y0),
            x1=int(x1),
            y1=int(y1),
            class_name=class_id,
            confidence=score,
            colour=colour,
        )

    return frame