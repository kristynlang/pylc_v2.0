Welcome to mask merging!

Each of the mask_dataset.py files contain the correct categories (labels, hex codes, and RGB arrays) for conversion as well as all the functions required to do so.

To convert more images from either Fortin2018, Frederickson, TaggartHodge Fluvial, TaggartHodge Regional, or Higgs et al you can simply use the existing code in merge_class, just make sure you are importing the right .py file into the notebook under the "Convert Masks" heading. You will need to make a few small changes:

First update the root variable to match the folder path you are trying to access on your machine (where the images are located).

Update all of the file paths in the notebook to match the location of the images/masks within your own directory. This includes the variables root, mydir, and img_path and mask_path for testing the merge function. 

Merge and save masks section:

Update mydir to match the location of the image/mask pairs

Image and mask pairs MUST follow the naming convention Img_Dataset_Location_H.tif (or .jpg) and Mask_Dataset_Location_H.png. Saved masks will end up in a subfolder in that directory with the same filename. 

If updating the merge_dataset.py files -- work from the scriptexample.py and follow the instructions to create a new merge_dataset.py file. If adding a class to an existing categorization scheme/dataset update the corrrepsonding merge_dataset.py file.