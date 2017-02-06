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
	-find work -name "__pycache__" -exec rm -rf {} \;
	-find work -name "*.pyc" -exec rm {} \;
	-find work -name "*.py"  -not -name "__init__.py" -exec rm {} \;

build: 
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	for var in $$(find work -name "*.ipynb" -not -path ".ipynb_checkpoints/*") ; \
	do \
			name=`echo $$var | cut -d'.' -f1`; \
			if [ ! -d "$$name" ]; then \
				jupyter nbconvert --to python --stdout $$name | tac | sed '0,/^$$/d' | tac > $$name.py; \
			fi \
	done
	find work -name "*.py" | xargs sed -i "s/get_ipython().magic('matplotlib inline')/matplotlib.use('agg')/"
	#find . -name "*.py" -print0 | xargs -0 perl -pi -e 's/ +$$//'

lint: build
	flake8 --exclude=lib/,bin/ work

test: build
	-docker rm marvin_modeling_test
	docker run -it --name marvin_modeling_test -v `pwd`:/home/creditx newreg.creditx.com/marvin/marvin_modeling:devel py.test

install: build package
	-docker kill marvin_modeling
	 rsync -avhP /opt/anaconda/conda-bld/linux-64/marvin_*.tar.bz2 newreg.creditx.com:/var/lib/docker/pkgs/creditx/linux-64
	 ssh newreg.creditx.com "bash -c 'cd /var/lib/docker/pkgs/creditx/linux-64; /home/weiyan/miniconda3/bin/conda index'"

run: clean
	-docker kill marvin_modeling
	-docker rm marvin_modeling
	docker run -it -d --name marvin_modeling -e PASSWORD=debug -p 28888:8888 \
		-v `pwd`:/home/creditx newreg.creditx.com/marvin/marvin_modeling:devel
