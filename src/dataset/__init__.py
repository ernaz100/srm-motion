from typing import Type, Union

from .dataset import Dataset
from .dataset_ffhq import DatasetFFHQ, DatasetFFHQCfg
from .dataset_grid import DatasetGrid   # noqa
from .dataset_mnist import DatasetMnist, DatasetMnistCfg
from .dataset_mnist_sudoku_3x3_eager import DatasetMnistSudoku3x3Eager, DatasetMnistSudoku3x3EagerCfg
from .dataset_mnist_sudoku_9x9_eager import DatasetMnistSudoku9x9Eager, DatasetMnistSudoku9x9EagerCfg
from .dataset_mnist_sudoku_9x9_lazy import DatasetMnistSudoku9x9Lazy, DatasetMnistSudoku9x9LazyCfg
from .dataset_mnist_sudoku_explicit_indices import DatasetMnistSudokuExplicitIndices, DatasetMnistSudokuExplicitIndicesCfg
from .dataset_counting_polygons import DatasetCountingPolygonsBlank, DatasetCountingPolygonsBlankCfg
from .dataset_counting_polygons import DatasetCountingPolygonsFFHQ, DatasetCountingPolygonsFFHQCfg
from .dataset_even_pixels import DatasetEvenPixels, DatasetEvenPixelsCfg
from .dataset_humanml3d import DatasetHumanML3D, DatasetHumanML3DCfg

from src.type_extensions import ConditioningCfg, Stage


DATASETS: dict[str, Dataset] = {
    "ffhq": DatasetFFHQ,
    "mnist": DatasetMnist,
    "mnist_grid": DatasetMnistSudoku3x3Eager,
    "mnist_sudoku": DatasetMnistSudoku9x9Eager,
    "mnist_sudoku_lazy": DatasetMnistSudoku9x9Lazy,
    "mnist_sudoku_explicit_indices": DatasetMnistSudokuExplicitIndices,
    "counting_polygons_blank": DatasetCountingPolygonsBlank,
    "counting_polygons_blank_explicit_conditional": DatasetCountingPolygonsBlank,
    "counting_polygons_blank_ambiguous_conditional": DatasetCountingPolygonsBlank,
    "counting_polygons_ffhq": DatasetCountingPolygonsFFHQ,
    "counting_polygons_ffhq_explicit_conditional": DatasetCountingPolygonsFFHQ,
    "counting_polygons_ffhq_ambiguous_conditional": DatasetCountingPolygonsFFHQ,
    "even_pixels": DatasetEvenPixels,
    "humanml3d": DatasetHumanML3D,
}


DatasetCfg = Union[
    DatasetFFHQCfg,
    DatasetMnistCfg,
    DatasetMnistSudoku3x3EagerCfg,
    DatasetMnistSudoku9x9EagerCfg,
    DatasetMnistSudoku9x9LazyCfg,
    DatasetMnistSudokuExplicitIndicesCfg,
    DatasetCountingPolygonsBlankCfg,
    DatasetCountingPolygonsFFHQCfg,
    DatasetEvenPixelsCfg,
    DatasetHumanML3DCfg,
]


def get_dataset_class(
    cfg: DatasetCfg
) -> Type[Dataset]:
    return DATASETS[cfg.name]


def get_dataset(
    cfg: DatasetCfg,
    conditioning_cfg: ConditioningCfg,
    stage: Stage,
) -> Dataset:
    return DATASETS[cfg.name](cfg, conditioning_cfg, stage)
