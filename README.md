# Ping pong ball tracker (offline)

This is part of the project for ECE-6258 (Digital image processing) at Georgia Tech.

Not much documentation is provided currently. It will be updated later.

## Usage

`<python_file>.py -h` to get the usage.
Generally use `q` key to get out of GUIs.

## Notes

Ensure `opencv_ffmpeg*.dll` file is present in the current directory or in `PATH`. This is required for reading video.

Generally quitting (with `q`) on `get_*.py` will save important parameters to `new_*.pickle` in root folder. Copy over the _good_ parameters to config folder.

Generally the GUIs are not updated with seek positions. That is, when video moves forward, the seek bar is not updated, but the other way works -- when seek bar is moved, video position changes.

Convert all video inputs to 30 fps or 60 fps to work across all platforms. There were video decoder problems with 80 fps videos on Windows.

Don't use `calib_find_instrinsics_video.py`. It is old, and uses frames from video without much intelligence. Use `calib_find_intrinsics_images.py` with manually selected images instead for reliability.

Keyboard shortcuts for `calib_find_intrinsics_images.py`:

- `s` - save current image
- `b` - seek back 1s
- `n` - go ahead 1s

`calib_find_intrinsics_images.py` expects the folder `calib_images` to be present in the root directory. It also accepts the normal `--video` argument that other scripts expect as video file name is used as prefix for finding calibration images. If calibration video not available, fool this script with a dummy video (file name).

Tested on **GoPro Hero3+ Silver** and **GoPro Hero5 Black**.
