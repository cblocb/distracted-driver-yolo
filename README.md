# distracted-driver-yolo

distracted driver classification project from computer vision

i took the state farm in-cabin distracted-driver dataset and collapsed the original 10 classes into a binary problem:

- focused
- distracted

the main point was to build the whole pipeline myself instead of just training a model once and calling it done

so this has:

- the dataset split / setup logic
- the yolov11 training script
- the validation / confusion-matrix script
- the final submitted report pdf

it ended up working pretty well on the binary version of the task

from the report it got about 95.1% validation accuracy with weighted f1 around 0.948

the class imbalance mattered a lot though, so accuracy by itself was kind of flattering and the focused class was definitely weaker than the distracted class

the dataset and weights arent in this repo, so if you want to actually run it youd need to point the paths at your own local copy
