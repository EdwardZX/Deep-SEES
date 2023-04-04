# Deep-SEES
This project is based on PyTorch, and colab.

Matlab Data Preprocessing (preprocessing Folder):

1.1 Use the transform_data (name, xy, polar(optional)) function to save the xy and polar:
- xy is the tracks (Time x Dimension x num), polar is the additional information (Time x num)
- Make sure the length of the xy and polar files are equal
- addpath('./MyUtils/MyTrajectoryPara/'); addpath('./MyUtils/'); 

1.2 Use the get_data_2d/3d/2d_adding_information scripts
- Input file location path, output file save location save_path, file name name_set, and whether to eliminate spatial correlation.

1.3 Copy two ouput files to './data/' directory of Deep-SEES.

Finally run main_DeepSEES.ipynb.
