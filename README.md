# Deep-SEES
This project is based on PyTorch, and colab.

Matlab Data Preprocessing (processing) (only need to be processed once):

1.1 Use the transform_data (name, xy, polar(optional)) function to save the xy and polar:
- Make sure the length of the xy and polar files are equal
- addpath('./MyUtils/MyTrajectoryPara/'); addpath('./MyUtils/'); 

1.2 Use the get_data_2d/3d/2d_adding_information scripts
- Input file location path, output file save location save_path, file name name_set, and whether to eliminate spatial correlation.

Finally run main_DeepSEES.ipynb


