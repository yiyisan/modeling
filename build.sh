rm work.egg-info -rf
jupyter nbconvert --to python work/marvin/binary_classification_evaluation/*.ipynb
jupyter nbconvert --to python work/marvin/binary_classifier_models/*.ipynb
jupyter nbconvert --to python work/marvin/dataPrepareforTraining/*.ipynb
python setup.py sdist
cp dist/work-0.0.1.tar.gz work
