# Ping pong ball tracker (offline)

This is part of the project for ECE-6258 (Digital image processing) at Georgia Tech.

Not much documentation is provided currently. It will be updated later.

## Usage

`<python_file>.py -h` to get the usage.
Generally use `q` key to get out of GUIs.

## Notes

Ensure `opencv_ffmpeg*.dll` file is present in the current directory or in `PATH`. This is required for reading video.

Generally quitting (with `q`) on `get_*.py` will save important parameters to `new_*.pickle` in root folder. Copy over the _good_ parameters to config folder.