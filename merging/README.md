# Welcome to mask merging!

Each of the merge_dataset.py files contain the correct categories (labels, hex codes, and RGB arrays) for conversion as well as all the functions required to do so.

To convert more images from either Fortin2018, Frederickson, TaggartHodge Fluvial, or TaggartHodge Regional you can simply use the existing code in merge_class, just make sure you are importing the right .py file into the notebook under the "Convert Masks" heading. Note: TaggartFluvial regenerating area needs to be manually corrected from #FF0000 to #FF0004 before running through merge_class.ipynb. [Quick guide to correcting masks in Affinity Photo](https://docs.google.com/document/d/1k6aSSUPxIlGXiu5I-dNbBTYAGT0lE0akCzrE0JGMY-Y/edit?usp=share_link). __All images must be reviewed to ensure classifications match merged scheme.__ Ability to reach merged schema is based on the [Consensus Cateogorization](https://www.google.com/url?q=https://docs.google.com/spreadsheets/d/1hoETdLEM5VlAI0T7wLzS9dZVotSBIjezb6bnjZ6-5f8/edit%23gid%3D1971148706&sa=D&source=docs&ust=1688071877295900&usg=AOvVaw1huZf8O3mKstBpZ4rxRhPt) table. Currently the Higgs et al. collection also requires manual correction.

## Steps

If the repository has not been cloned start at step 1, if the repository has been cloned start at step 3.
1) Download notebook scripy as .ipynb by right clicking on "raw" button.
2) Download the category conversion file specific to your dataset as .py by right clicking on "raw" button. Also download scriptexample.py if you want instructions on how to update these scripts.
3) Check the requirements.txt file and download the necessary packages into your python environment using either conda install example-package or pip install example-package ex. pip install numpy. Note that cv2 is installed as opencv-python. To install a specific version number use conda install numpy==1.24.3, but the most recent versions should be fine. If you do not know what a python environment is or how to set one up see additional steps at the end of these instructions. You can download all the packages at once using pip install -r requirements.txt
4) Open merge_class.ipynb in Jupyter Notebook
5) To use merge_class.ipynb you will need to make a few small changes to the notebook: First update the root variable to match the folder path you are trying to access on your machine (where the images are located).Then update all of the file paths in the notebook to match the location of the images/masks within your own directory. This includes the variables root, mydir, and img_path and mask_path for testing the merge function.
6) In the merge and save masks section, update mydir to match the location of the image/mask pairs.
8) Image and mask pairs MUST follow the naming convention Dataset_Location_H.tif (or .jpg) and Dataset_Location_H.png. Filenames are sorted into masks and images based on file extension. Saved masks will end up in a subfolder in that directory called 'Merge' with the correct mask filename (Dataset_Location_H.png) and the original files will be updated to be Dataset_Location_H_originalmask.png.

## Additional Information

If updating the merge_dataset.py files -- work from the scriptexample.py and follow the instructions to create a new merge_dataset.py file. If adding a class to an existing categorization scheme/dataset update the corrrepsonding merge_dataset.py file. Intructions can be found in scriptexample.py.

## Setting up a Python environment (to do before Step 3):

Open your terminal:
Using python: (note myenv is an example name, you can call it anything you want)
 
    python -m venv myenv

to activate your environment:

    source myenv/bin/activate 

OR using anaconda

    conda create -n myenv
    conda activate myenv

then install appropriate python modules into the environment you created

to add your environment to jupyter notebook:

    pip install ipykernel
    
    python -m ipykernel install --user --name=myenv
    
now you can select this kernel in Jupyter notebook
    
    
