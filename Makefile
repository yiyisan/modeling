.PHONY: init clean lint package all

init:
	conda build flake8

package:
	python setup.py sdist
	conda build purge
	conda build --python 2.7 recipe
	conda build --python 3.5 recipe

clean:
	-docker rm marvin_modeling
	rm -rf work.egg-info
	find work -name "__pycache__" -exec rm {} \;
	find work -name "*.pyc" -exec rm {} \;

build:
	jupyter nbconvert --to python work/marvin/binary_classification_evaluation/*.ipynb
	jupyter nbconvert --to python work/marvin/binary_classifier_models/*.ipynb
	jupyter nbconvert --to python work/marvin/dataPrepareforTraining/*.ipynb
	find work -name "*.py" | xargs sed -i "s/get_ipython().magic('matplotlib inline')/matplotlib.use('agg')/"

lint: build
	flake8 --exclude=lib/,bin/ work

test: clean build
	docker run -it --name marvin_modeling -p 28888:8888 -v `pwd`:/home/creditx registry.creditx.com:5000/marvin_modeling:test py.test

install: build package
	 rsync -avhP /opt/anaconda/conda-bld/linux-64/marvin_*.tar.bz2 newreg.creditx.com:/var/lib/docker/pkgs/creditx/linux-64
	 ssh newreg.creditx.com 'bash -s' < index.sh

all: init clean build lint

run: clean
	docker run -it --name marvin_modeling -e PASSWORD=debug -p 28888:8888 -v `pwd`:/home/creditx registry.creditx.com:5000/marvin_modeling:devel
