import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import sleap_io as sio

from movement.io.validators import DeepLabCutPosesFile

# get logger
logger = logging.getLogger(__name__)


def from_dlc(file_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a DeepLabCut (DLC) files.
    Files must be in .h5 format or .csv format.

    Parameters
    ----------
    file_path : pathlib Path or str
        Path to the file containing the DLC poses.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC poses

    Examples
    --------
    >>> from movement.io import load_poses
    >>> poses = load_poses.from_dlc("path/to/file.h5")
    """

    # Validate the input file path
    dlc_poses_file = DeepLabCutPosesFile(file_path=file_path)  # type: ignore
    file_suffix = dlc_poses_file.file_path.suffix

    # Load the DLC poses
    try:
        if file_suffix == ".csv":
            df = _parse_dlc_csv_to_dataframe(dlc_poses_file.file_path)
        else:  # file can only be .h5 at this point
            df = pd.read_hdf(dlc_poses_file.file_path)
            # above line does not necessarily return a DataFrame
            df = pd.DataFrame(df)
    except (OSError, TypeError, ValueError) as e:
        error_msg = (
            f"Could not load poses from {file_path}. "
            "Please check that the file is valid and readable."
        )
        logger.error(error_msg)
        raise OSError from e
    logger.info(f"Loaded poses from {file_path}")
    return df


def _parse_dlc_csv_to_dataframe(file_path: Path) -> pd.DataFrame:
    """If poses are loaded from a DeepLabCut.csv file, the resulting DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the csv file to a DataFrame with multi-index columns.

    Parameters
    ----------
    file_path : pathlib Path
        Path to the file containing the DLC poses, in .csv format.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC poses, with multi-index columns.
    """

    possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
    with open(file_path, "r") as f:
        # if line starts with a possible level name, split it into a list
        # of strings, and add it to the list of header lines
        header_lines = [
            line.strip().split(",")
            for line in f.readlines()
            if line.split(",")[0] in possible_level_names
        ]

    # Form multi-index column names from the header lines
    level_names = [line[0] for line in header_lines]
    column_tuples = list(zip(*[line[1:] for line in header_lines]))
    columns = pd.MultiIndex.from_tuples(column_tuples, names=level_names)

    # Import the DLC poses as a DataFrame
    df = pd.read_csv(
        file_path, skiprows=len(header_lines), index_col=0, names=columns
    )
    df.columns.rename(level_names, inplace=True)
    return df


def convert_dlc_dataframe_to_sleap_labels(df: pd.DataFrame) -> sio.Labels:
    """Converts a DataFrame containing DLC poses to a SLEAP Labels object.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the DLC poses.

    Returns
    -------
    sleap Labels
        SLEAP Labels object containing labeled frames.
    """

    # TODO: create video object from image frames

    # assume 1 scorer
    scorer = df.columns.get_level_values("scorer").unique()[0]
    individuals = (
        df.columns.get_level_values("individuals").unique().to_list()
        if "individuals" in df.columns.names
        else []
    )
    bodyparts = df.columns.get_level_values("bodyparts").unique().to_list()

    # create Tracks and Skeleton
    tracks = [sio.Track(name=track) for track in individuals]
    skeleton = sio.Skeleton(nodes=bodyparts)

    lfs = []

    # create a PredictedInstance for each row and each track
    for i, row in df.iterrows():
        instances = []
        if tracks:
            # multianimal dataset
            for track in tracks:
                any_not_missing = False
                instance_points = {}
                for node in skeleton.node_names:
                    x, y, score = (
                        row[(scorer, track.name, node, "x")],
                        row[(scorer, track.name, node, "y")],
                        row[(scorer, track.name, node, "likelihood")],
                    )
                    instance_points[node] = sio.PredictedPoint(
                        x, y, score=score
                    )
                    if ~(np.isnan(x) and np.isnan(y)):
                        any_not_missing = True

                # skip instance creation if all points are missing
                if any_not_missing:
                    instances.append(
                        sio.PredictedInstance(
                            points=instance_points,
                            track=track,
                            skeleton=skeleton,
                        )
                    )
        else:
            any_not_missing = False
            instance_points = {}
            for node in skeleton.node_names:
                x, y, score = (
                    row[(scorer, node, "x")],
                    row[(scorer, node, "y")],
                    row[(scorer, node, "likelihood")],
                )
                instance_points[node] = sio.PredictedPoint(x, y, score=score)
                if ~(np.isnan(x) and np.isnan(y)):
                    any_not_missing = True
            # skip instance creation if all points are missing
            if any_not_missing:
                instances.append(
                    sio.PredictedInstance(
                        points=instance_points,
                        skeleton=skeleton,
                    )
                )
        if instances:
            lfs.append(
                sio.LabeledFrame(
                    video=sio.Video(""), instances=instances, frame_idx=i
                )
            )

    return sio.Labels(labeled_frames=lfs)
