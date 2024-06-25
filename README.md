Notebook and functions used to try various classification tasks on processed multimodal vocal data.

1. Start a virtual environment in python 3.11
conda create -n data_class_env python=3.11
conda activate data_class_env

2. Install requirements: 
pip install -r requirements.txt

3. Running the code:
Follow the step by step process outlines in the python notebook. If interested in the internal workings of the
functions used, look at the other python scripts containing them.

4. Data
It is recommneded that you work with the data within the data folder. Do not commit your data csvs. If you rename them from 
the naming conventioned used by the repo, add them to the gitignore file. 

5. RUNNING THE SCRIPTS
If modules are not recognised, make sure that your project is in your path, 
export PYTHONPATH="path_to/feature_based_classifiers:$PYTHONPATH"
e.g. I would run the following:
export PYTHONPATH="/Users/Suvi/Documents/UPF/THESIS/feature_based_classifiers:$PYTHONPATH"

