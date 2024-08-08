
import base64

from io import BytesIO
from PIL import Image


def image2base64(image: Image.Image):
    buff = BytesIO()
    image = image.convert('RGB')
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str


def base642image(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


