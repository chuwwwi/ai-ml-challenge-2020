Hi! This is the sparse folder containing everything needed for BROKE's deployment in
an actual, functional scenario. Since I opted for an entirely offline program to
ensure maximum security and customizability, I thought it only made sense for me to
include the actual solution seen in the demo video.

If you believe the necessary files may not all be in this folder, please run
edit_config.py. The only directory you must have is data/models/broke.SavedModel, which
must be a TensorFlow model that takes in an input of shape (512) and outputs shape (1).
If this model is not in this folder, it can instead be found in the Compiled Methods
folder in the GitHub submission.

To get started, run edit_config.py to configure your settings. You can then place your
files into _INPUTS and expect to see them in _OUTPUTS. If you choose to retrain the model,
you can relabel them sequentially via the command line.

Here are the necessary modules to run BROKE:
	os              - handling files and directories
	re              - searching strings for invalid characters
	json            - handling dictionaries and stringed lists
	numpy           - handling arrays and making code cleaner
	pandas          - creating tables and editing CSV files
	PyPDF2          - reading and parsing PDF files
	(shutil)        - should be included with python, but used to move stuff around
	datetime        - timing is used to uniquify some files names in retraining
	tensorflow      - deep learning framework for loading and using models
	python-docx     - reading and parsing .docx (Word) files
	transformers    - used to access BERT's tokenizer

If using pip and some packages are not showing up but you believe you have them installed,
you can run [python -c "import PACKAGE; print('Success!')"] (without the brackets, and
substituting in your package name for PACKAGE) to check whether you have the package or not.
Please note that all packages are imported with their names above except for python-docx, which
is imported as docx.