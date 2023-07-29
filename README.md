# Deep-SEES

This project, Deep-SEES, leverages the PyTorch library and is designed to be executed in the Google Colaboratory environment.

## Data Preprocessing

The data preprocessing phase occurs in Matlab and is facilitated by the scripts found in the `preprocessing` folder.

### Step 1.1: Data Transformation

The `transform_data(name, xy, polar(optional))` function is used to save the xy and polar data.

- `xy` should be configured as tracks (Time * Dimension * num), and `polar` as the additional information (Time * num).
- Ensure that the lengths of the xy and polar files are identical.
- To use the function, add the necessary paths as shown below:

matlab

Copy

```
addpath('./MyUtils/MyTrajectoryPara/');
addpath('./MyUtils/');
```

### Step 1.2: Data Script Execution

Execute the `get_data_2d/3d/2d_adding_information` scripts.

- These scripts require the input file location `path`, the output file save location `save_path`, the file name `name_set`, and a Boolean value determining whether to eliminate spatial correlation.

### Step 1.3: File Relocation

After running the scripts, copy the two output files to the './data/' directory of the Deep-SEES project.

## Project Execution

Finally, you can execute the project by running the `main_DeepSEES.py` in python.

For your convenience, here's the [Google Colab Link](https://colab.research.google.com/drive/1kSlaNIUMoxP6sY9th2_R_mUwTncpQgVq?usp=sharing) to the project.
