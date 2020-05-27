# Primitive Fitting by Minimizing Volumetric Loss

![Fitted deer](images/deer.png?raw=true)

This project implements primitive fitting by gradient descent. The loss functions being optimized are based on either volume ratios or the volumetric Jaccard index.

## Dependencies
* Python 3
* PyTorch (version 1.5 was used in the writing of this program)
* matplotlib

## How to use

Execute the following command to run the sample program:

```
python3 run.py
```

and follow the prompts to configure the run.

To use a custom .obj file model, download and build the following C++ voxelizer program: https://github.com/karimnaaji/voxelizer/tree/master/example

Use the program to output a text document of points, and take note of the resolution parameter (voxel size) that you pass to to the program. When running run.py, you will be prompted to enter the model that you would like to fit. Choose "Custom", insert the path to the text document of points, and input the resolution (voxel size) you used.

The primary function for use of this project as a library is fit.fit_points. This takes in a voxel model (potentially hollow) as a point cloud, the voxel size, and configuration parameters as arguments and returns a list of fitted models.

## License

This project is distributed under the MIT License except for the deer, monkey, and tram models which are under the CC-BY License. These models were created by Google Poly.

## Images

![Fitted bunny](images/bunny.png?raw=true)
![Fitted monkey](images/monkey.png?raw=true)
![Fitted tram](images/tram.png?raw=true)
![Fitted hornet](images/hornet.png?raw=true)