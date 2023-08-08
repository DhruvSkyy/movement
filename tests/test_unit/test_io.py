import os

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.datasets import fetch_pose_data_path
from movement.io import PosesAccessor, load_poses, save_poses
from movement.io.validators import (
    ValidFile,
    ValidHDF5,
    ValidPosesCSV,
    ValidPoseTracks,
)


class TestPosesIO:
    """Test the IO functionalities of the PoseTracks class."""

    @pytest.fixture
    def valid_tracks_array(self):
        """Return a valid tracks array."""
        return np.zeros((10, 2, 2, 2))

    @pytest.fixture
    def valid_pose_dataset(self, valid_tracks_array):
        """Return a valid pose tracks dataset."""
        dim_names = PosesAccessor.dim_names
        return xr.Dataset(
            data_vars={
                "pose_tracks": xr.DataArray(
                    valid_tracks_array, dims=dim_names
                ),
                "confidence": xr.DataArray(
                    valid_tracks_array[..., 0], dims=dim_names[:-1]
                ),
            },
            coords={
                "time": np.arange(valid_tracks_array.shape[0]),
                "individuals": ["ind1", "ind2"],
                "keypoints": ["key1", "key2"],
                "space": ["x", "y"],
            },
            attrs={
                "fps": None,
                "time_unit": "frames",
                "source_software": "SLEAP",
                "source_file": "test.h5",
            },
        )

    @pytest.fixture
    def invalid_pose_datasets(self, valid_pose_dataset):
        """Return a list of invalid pose tracks datasets."""
        return {
            "not_a_dataset": [1, 2, 3],
            "empty_dataset": xr.Dataset(),
            "missing_var": valid_pose_dataset.drop_vars("pose_tracks"),
            "missing_dim": valid_pose_dataset.drop_dims("time"),
        }

    @pytest.fixture
    def dlc_file_h5_single(self):
        """Return the path to a valid DLC h5 file containing pose data
        for a single animal."""
        return fetch_pose_data_path("DLC_single-wasp.predictions.h5")

    @pytest.fixture
    def dlc_file_csv_single(self):
        """Return the path to a valid DLC .csv file containing pose data
        for a single animal. The underlying data is the same as in the
        `dlc_file_h5_single` fixture."""
        return fetch_pose_data_path("DLC_single-wasp.predictions.csv")

    @pytest.fixture
    def dlc_file_csv_multi(self):
        """Return the path to a valid DLC .csv file containing pose data
        for multiple animals."""
        return fetch_pose_data_path("DLC_two-mice.predictions.csv")

    @pytest.fixture
    def sleap_file_h5_single(self):
        """Return the path to a valid SLEAP "analysis" .h5 file containing
        pose data for a single animal."""
        return fetch_pose_data_path("SLEAP_single-mouse_EPM.analysis.h5")

    @pytest.fixture
    def sleap_file_slp_single(self):
        """Return the path to a valid SLEAP .slp file containing
        predicted poses (labels) for a single animal."""
        return fetch_pose_data_path("SLEAP_single-mouse_EPM.predictions.slp")

    @pytest.fixture
    def sleap_file_h5_multi(self):
        """Return the path to a valid SLEAP "analysis" .h5 file containing
        pose data for multiple animals."""
        return fetch_pose_data_path(
            "SLEAP_three-mice_Aeon_proofread.analysis.h5"
        )

    @pytest.fixture
    def sleap_file_slp_multi(self):
        """Return the path to a valid SLEAP .slp file containing
        predicted poses (labels) for multiple animals."""
        return fetch_pose_data_path(
            "SLEAP_three-mice_Aeon_proofread.predictions.slp"
        )

    @pytest.fixture
    def valid_files(
        self,
        dlc_file_h5_single,
        dlc_file_csv_single,
        dlc_file_csv_multi,
        sleap_file_h5_single,
        sleap_file_slp_single,
        sleap_file_h5_multi,
        sleap_file_slp_multi,
    ):
        """Aggregate all valid files in a dictionary, for convenience."""
        return {
            "DLC_h5_single": dlc_file_h5_single,
            "DLC_csv_single": dlc_file_csv_single,
            "DLC_csv_multi": dlc_file_csv_multi,
            "SLEAP_h5_single": sleap_file_h5_single,
            "SLEAP_slp_single": sleap_file_slp_single,
            "SLEAP_h5_multi": sleap_file_h5_multi,
            "SLEAP_slp_multi": sleap_file_slp_multi,
        }

    @pytest.fixture
    def invalid_files(self, tmp_path):
        unreadable_file = tmp_path / "unreadable.h5"
        with open(unreadable_file, "w") as f:
            f.write("unreadable data")
            os.chmod(f.name, 0o000)

        wrong_ext_file = tmp_path / "wrong_extension.txt"
        with open(wrong_ext_file, "w") as f:
            f.write("")

        h5_file_no_dataframe = tmp_path / "no_dataframe.h5"
        with h5py.File(h5_file_no_dataframe, "w") as f:
            f.create_dataset("data_in_list", data=[1, 2, 3])

        nonexistent_file = tmp_path / "nonexistent.h5"

        directory = tmp_path / "directory"
        directory.mkdir()

        fake_h5_file = tmp_path / "fake.h5"
        with open(fake_h5_file, "w") as f:
            f.write("")

        fake_csv_file = tmp_path / "fake.csv"
        with open(fake_csv_file, "w") as f:
            f.write("some,columns\n")
            f.write("1,2")

        return {
            "unreadable": unreadable_file,
            "wrong_ext": wrong_ext_file,
            "no_dataframe": h5_file_no_dataframe,
            "nonexistent": nonexistent_file,
            "directory": directory,
            "fake_h5": fake_h5_file,
            "fake_csv": fake_csv_file,
        }

    @pytest.fixture
    def dlc_style_df(self, dlc_file_h5_single):
        """Return a valid DLC-style DataFrame."""
        df = pd.read_hdf(dlc_file_h5_single)
        return df

    def test_load_from_valid_files(self, valid_files):
        """Test that loading pose tracks from a wide variety of valid files
        returns a proper Dataset."""
        abbrev_expand = {"DLC": "DeepLabCut", "SLEAP": "SLEAP"}

        for file_type, file_path in valid_files.items():
            if file_type.startswith("DLC"):
                ds = load_poses.from_dlc_file(file_path)
            elif file_type.startswith("SLEAP"):
                ds = load_poses.from_sleap_file(file_path)

            assert isinstance(ds, xr.Dataset)
            # Expected variables are present and of right shape/type
            for var in ["pose_tracks", "confidence"]:
                assert var in ds.data_vars
                assert isinstance(ds[var], xr.DataArray)
            assert ds.pose_tracks.ndim == 4
            assert ds.confidence.shape == ds.pose_tracks.shape[:-1]
            # Check the dims and coords
            DIM_NAMES = PosesAccessor.dim_names
            assert all([i in ds.dims for i in DIM_NAMES])
            for d, dim in enumerate(DIM_NAMES[1:]):
                assert ds.dims[dim] == ds.pose_tracks.shape[d + 1]
                assert all([isinstance(s, str) for s in ds.coords[dim].values])
            assert all([i in ds.coords["space"] for i in ["x", "y"]])
            # Check the metadata attributes
            assert ds.source_software == abbrev_expand[file_type.split("_")[0]]
            assert ds.source_file == file_path.as_posix()
            assert ds.fps is None

    def test_load_from_invalid_files(self, invalid_files):
        """Test that loading pose tracks from a wide variety of invalid files
        raises the appropriate errors."""
        for file_path in invalid_files.values():
            with pytest.raises((OSError, ValueError)):
                load_poses.from_dlc_file(file_path)
            with pytest.raises((OSError, ValueError)):
                load_poses.from_sleap_file(file_path)

    @pytest.mark.parametrize("file_path", [1, 1.0, True, None, [], {}])
    def test_load_with_incorrect_file_path_types(self, file_path):
        """Test loading poses from a file_path with an incorrect type."""
        with pytest.raises(TypeError):
            load_poses.from_dlc_file(file_path)
        with pytest.raises(TypeError):
            load_poses.from_sleap_file(file_path)

    def test_file_validator(self, invalid_files):
        """Test that the file validator class raises the right errors."""
        for file_type, file_path in invalid_files.items():
            if file_type == "unreadable":
                with pytest.raises(PermissionError):
                    ValidFile(path=file_path, expected_permission="r")
            elif file_type == "wrong_ext":
                with pytest.raises(ValueError):
                    ValidFile(
                        path=file_path,
                        expected_permission="r",
                        expected_suffix=["h5", "csv"],
                    )
            elif file_type == "nonexistent":
                with pytest.raises(FileNotFoundError):
                    ValidFile(path=file_path, expected_permission="r")
            elif file_type == "directory":
                with pytest.raises(IsADirectoryError):
                    ValidFile(path=file_path, expected_permission="r")
            elif file_type in ["fake_h5", "no_dataframe"]:
                with pytest.raises(ValueError):
                    ValidHDF5(path=file_path, expected_datasets=["dataframe"])
            elif file_type == "fake_csv":
                with pytest.raises(ValueError):
                    ValidPosesCSV(path=file_path)

    def test_load_and_save_to_dlc_df(self, dlc_style_df):
        """Test that loading pose tracks from a DLC-style DataFrame and
        converting back to a DataFrame returns the same data values."""
        ds = load_poses.from_dlc_df(dlc_style_df)
        df = save_poses.to_dlc_df(ds)
        assert np.allclose(df.values, dlc_style_df.values)

    def test_save_and_load_dlc_file(self, valid_pose_dataset, tmp_path):
        """Test that saving pose tracks to DLC .h5 and .csv files and then
        loading them back in returns the same Dataset."""
        save_poses.to_dlc_file(valid_pose_dataset, tmp_path / "dlc.h5")
        save_poses.to_dlc_file(valid_pose_dataset, tmp_path / "dlc.csv")
        ds_from_h5 = load_poses.from_dlc_file(tmp_path / "dlc.h5")
        ds_from_csv = load_poses.from_dlc_file(tmp_path / "dlc.csv")
        xr.testing.assert_allclose(ds_from_h5, valid_pose_dataset)
        xr.testing.assert_allclose(ds_from_csv, valid_pose_dataset)

    def test_save_valid_dataset_to_invalid_file_paths(
        self, valid_pose_dataset, invalid_files, tmp_path
    ):
        with pytest.raises(FileExistsError):
            save_poses.to_dlc_file(
                valid_pose_dataset, invalid_files["fake_h5"]
            )
        with pytest.raises(ValueError):
            save_poses.to_dlc_file(valid_pose_dataset, tmp_path / "dlc.txt")
        with pytest.raises(IsADirectoryError):
            save_poses.to_dlc_file(
                valid_pose_dataset, invalid_files["directory"]
            )

    def test_load_from_dlc_file_csv_or_h5_file_returns_same(
        self, dlc_file_h5_single, dlc_file_csv_single
    ):
        """Test that loading pose tracks from DLC .csv and .h5 files
        return the same Dataset."""
        ds_from_h5 = load_poses.from_dlc_file(dlc_file_h5_single)
        ds_from_csv = load_poses.from_dlc_file(dlc_file_csv_single)
        xr.testing.assert_allclose(ds_from_h5, ds_from_csv)

    @pytest.mark.parametrize("fps", [None, -5, 0, 30, 60.0])
    def test_fps_and_time_coords(self, sleap_file_h5_multi, fps):
        """Test that time coordinates are set according to the fps."""
        ds = load_poses.from_sleap_file(sleap_file_h5_multi, fps=fps)
        if (fps is None) or (fps <= 0):
            assert ds.fps is None
            assert ds.time_unit == "frames"
        else:
            assert ds.fps == fps
            assert ds.time_unit == "seconds"
            np.allclose(
                ds.coords["time"].data,
                np.arange(ds.dims["time"], dtype=int) / ds.attrs["fps"],
            )

    def test_load_from_str_path(self, sleap_file_h5_single):
        """Test that file paths provided as strings are accepted as input."""
        xr.testing.assert_allclose(
            load_poses.from_sleap_file(sleap_file_h5_single),
            load_poses.from_sleap_file(sleap_file_h5_single.as_posix()),
        )

    def test_save_invalid_pose_datasets(self, invalid_pose_datasets, tmp_path):
        """Test that saving invalid pose datasets raises ValueError."""
        for ds in invalid_pose_datasets.values():
            with pytest.raises(ValueError):
                save_poses.to_dlc_file(ds, tmp_path / "test.h5")

    @pytest.mark.parametrize(
        "tracks_array",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # not an ndarray
            np.zeros((10, 2, 3)),  # not 4d
            np.zeros((10, 2, 3, 4)),  # last dim not 2 or 3
        ],
    )
    def test_tracks_array_validation(self, tracks_array):
        """Test that invalid tracks arrays raise the appropriate errors."""
        with pytest.raises(ValueError):
            ValidPoseTracks(tracks_array=tracks_array)

    @pytest.mark.parametrize(
        "scores_array",
        [
            None,  # valid, should default to array of NaNs
            np.ones((10, 3, 2)),  # will not match tracks_array shape
            [1, 2, 3],  # not an ndarray, should raise ValueError
        ],
    )
    def test_scores_array_validation(self, valid_tracks_array, scores_array):
        """Test that invalid scores arrays raise the appropriate errors."""
        if scores_array is None:
            poses = ValidPoseTracks(tracks_array=valid_tracks_array)
            assert np.all(np.isnan(poses.scores_array))
        else:
            with pytest.raises(ValueError):
                ValidPoseTracks(
                    tracks_array=valid_tracks_array, scores_array=scores_array
                )

    @pytest.mark.parametrize(
        "individual_names",
        [
            None,  # generate default names
            ["ind1", "ind2"],  # valid input
            ("ind1", "ind2"),  # valid input
            [1, 2],  # will be converted to ["1", "2"]
            "ind1",  # will be converted to ["ind1"]
            5,  # invalid, should raise ValueError
        ],
    )
    def test_individual_names_validation(
        self, valid_tracks_array, individual_names
    ):
        if individual_names is None:
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array,
                individual_names=individual_names,
            )
            assert poses.individual_names == ["individual_0", "individual_1"]
        elif isinstance(individual_names, (list, tuple)):
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array,
                individual_names=individual_names,
            )
            assert poses.individual_names == [str(i) for i in individual_names]
        elif isinstance(individual_names, str):
            poses = ValidPoseTracks(
                tracks_array=np.zeros((10, 1, 2, 2)),
                individual_names=individual_names,
            )
            assert poses.individual_names == [individual_names]
            # raises error if not 1 individual
            with pytest.raises(ValueError):
                ValidPoseTracks(
                    tracks_array=valid_tracks_array,
                    individual_names=individual_names,
                )
        else:
            with pytest.raises(ValueError):
                ValidPoseTracks(
                    tracks_array=valid_tracks_array,
                    individual_names=individual_names,
                )

    @pytest.mark.parametrize(
        "keypoint_names",
        [
            None,  # generate default names
            ["key1", "key2"],  # valid input
            ("key", "key2"),  # valid input
            [1, 2],  # will be converted to ["1", "2"]
            "key1",  # will be converted to ["ind1"]
            5,  # invalid, should raise ValueError
        ],
    )
    def test_keypoint_names_validation(
        self, valid_tracks_array, keypoint_names
    ):
        if keypoint_names is None:
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array, keypoint_names=keypoint_names
            )
            assert poses.keypoint_names == ["keypoint_0", "keypoint_1"]
        elif isinstance(keypoint_names, (list, tuple)):
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array, keypoint_names=keypoint_names
            )
            assert poses.keypoint_names == [str(i) for i in keypoint_names]
        elif isinstance(keypoint_names, str):
            poses = ValidPoseTracks(
                tracks_array=np.zeros((10, 2, 1, 2)),
                keypoint_names=keypoint_names,
            )
            assert poses.keypoint_names == [keypoint_names]
            # raises error if not 1 keypoint
            with pytest.raises(ValueError):
                ValidPoseTracks(
                    tracks_array=valid_tracks_array,
                    keypoint_names=keypoint_names,
                )
        else:
            with pytest.raises(ValueError):
                ValidPoseTracks(
                    tracks_array=valid_tracks_array,
                    keypoint_names=keypoint_names,
                )
