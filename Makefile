.PHONY: init clean lint package all

init:
	conda install flake8

package:
	python setup.py sdist
	conda build purge
	conda build --python 2.7 recipe
	conda build --python 3.5 recipe

clean:
	rm -rf work.egg-info
	find work -name "*.pyc" -exec rm {} \;

lint: 
	jupyter nbconvert --to python work/marvin/binary_classification_evaluation/*.ipynb
	jupyter nbconvert --to python work/marvin/binary_classifier_models/*.ipynb
	jupyter nbconvert --to python work/marvin/dataPrepareforTraining/*.ipynb
	find work -name "*.py" | xargs sed -i "s/get_ipython().magic('matplotlib inline')/matplotlib.use('agg')/"
	flake8 --exclude=lib/,bin/ work

all: init clean lint
