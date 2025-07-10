from typing import NotRequired, TypedDict, Union

from jaxtyping import Float
from numpy import ndarray
from PIL.Image import Image
from torch import Tensor


class Example(TypedDict, total=True):
    # NOTE this Image can include a mask as alpha channel (LA or RGBA mode)
    image: NotRequired[Union[Image, Tensor]]
    label: NotRequired[int | dict[str, int] | Tensor]  # Allow tensor labels for embeddings
    path: NotRequired[str]
