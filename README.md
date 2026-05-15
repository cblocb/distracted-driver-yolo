# distracted-driver-yolo

this was a computer vision class project on distracted-driver classification using the state farm in-cabin dataset

i simplified the original 10-class setup into a binary classification problem:

- `focused`
- `distracted`

the repo includes:

- `project2train.py` for dataset splitting + training
- `project2validate.py` for validation metrics / confusion matrix output
- `project2_report.pdf` for the submitted report
- a few report figures

what i was trying to do:

- create a persistent train/validation split by subject
- train a small yolov11 classification model on the binary task
- evaluate it with accuracy, precision, recall, f1, and qualitative review of validation examples

final result from the report:

- about `95.1%` validation accuracy
- weighted f1 around `0.948`

the more important point than raw accuracy was that the binary task worked pretty well, but the model was stronger on the distracted class than the focused class because the data was pretty imbalanced.

the dataset itself and model weights are not included here, so if you want to run this end to end you would need to point the paths in the scripts at your local copy of the dataset.
