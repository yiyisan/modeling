.PHONY: init clean lint package all

init:
	conda build flake8

package:
	python setup.py sdist
	rm /opt/anaconda/conda-bld/src_cache/work*.tar.gz -rf
	conda build purge
	conda build --python 2.7 recipe
	conda build --python 3.5 recipe

clean:
	rm -rf work.egg-info
	-sudo find work -name "__pycache__" -exec rm -rf {} \;
	-sudo find work -name "*.pyc" -exec rm {} \;

build:
	jupyter nbconvert --to python work/marvin/binary_classification_evaluation/*.ipynb
	jupyter nbconvert --to python work/marvin/binary_classifier_models/*.ipynb
	jupyter nbconvert --to python work/marvin/dataPrepareforTraining/*.ipynb
	find work -name "*.py" | xargs sed -i "s/get_ipython().magic('matplotlib inline')/matplotlib.use('agg')/"

lint: build
	flake8 --exclude=lib/,bin/ work

test: build
	-docker rm marvin_modeling_test
	docker run -it --name marvin_modeling_test -v `pwd`:/home/creditx registry.creditx.com:5000/marvin_modeling:test py.test

install: build package
	 rsync -avhP /opt/anaconda/conda-bld/linux-64/marvin_*.tar.bz2 newreg.creditx.com:/var/lib/docker/pkgs/creditx/linux-64
	 ssh newreg.creditx.com "bash -c 'cd /var/lib/docker/pkgs/creditx/linux-64; /home/weiyan/miniconda3/bin/conda index'"

run: clean
	-docker kill marvin_modeling
	-docker rm marvin_modeling
	docker run -it -d --name marvin_modeling -e PASSWORD=debug -p 28888:8888 -v `pwd`:/home/creditx registry.creditx.com:5000/marvin_modeling:devel
