.PHONY: init clean lint build package all

init:
	conda install flake8

build: clean
	jupyter nbconvert --to python work/marvin/binary_classification_evaluation/*.ipynb
	jupyter nbconvert --to python work/marvin/binary_classifier_models/*.ipynb
	jupyter nbconvert --to python work/marvin/dataPrepareforTraining/*.ipynb
	find . -name "*.py" | xargs sed -i "s/get_ipython\(\)\.magic\(\'matplotlib inline\'\)/matplotlib\.use\(\'Agg\'\)/"

package:
	python setup.py sdist
	conda build --python 2.7 recipe
	conda build --python 3.5 recipe

clean:
	rm -rf work.egg-info
	find work -name "*.pyc" -exec rm {} \;

lint: 
	flake8 --exclude=lib/,bin/ .

all: init clean build lint
