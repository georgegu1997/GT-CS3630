import cozmo
import numpy as np
import os
from PIL import ImageDraw, ImageFont

def annotate_markers(image, markers, scale=1):
    '''
    Draw marker annotations directly on top of the camera image. This modifies the image.

    Inputs: 
        - image: a PIL Image
        - markers: list of marker detections as provided by `detect.detect_markers(...)`
        - scale: resolution scale of the annotation
    '''

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'Arial.ttf'), 14)

    for marker in markers:
        corners = np.array(marker['corner_coords']) * scale

        # Draw the marker boundary as lines
        for i, (x,y) in enumerate(corners):
            ni = 0 if len(corners) == i + 1 else i + 1
            nx, ny = corners[ni]
            draw.line([x, y, nx, ny], fill='lightgreen', width=3)

        # Draw the marker corners as points
        for x, y in corners:
            r = 5
            draw.ellipse([x-r, y-r, x+r, y+r], fill='yellow')

        # Show the location information
        x, y, h = marker['xyh']

        draw.text(
            (corners[0][0] + 5, corners[0][1] + 5), 
            'x: {:0.2f}mm\ny: {:0.2f}mm\nh: {:0.2f}*'.format(x, y, h), 
            fill='yellow',
            font=font
        )


# annotator for the camera stream
class MarkerAnnotator(cozmo.annotate.Annotator):
    def __init__(self, img_annotator):
        super().__init__(img_annotator)
        self.markers = []

    def apply(self, image, scale):
        if not self.markers:
            return

        annotate_markers(image, self.markers, scale)

        