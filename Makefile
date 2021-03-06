.PHONY: init clean lint package all

init:
	conda build flake8

package:
	python setup.py sdist
	rm /opt/anaconda/conda-bld/src_cache/marvin_modeling*.tar.gz -rf
	conda build purge
	conda build recipe

clean:
	rm -rf work.egg-info
	-find . -name ".ipynb_checkpoints" | xargs rm -rf
	-find work -name ".ipynb" -exec nbstripout {} \;
	-find work -name "__pycache__" -exec rm -rf {} \;
	-find work -name "*.pyc" -exec rm {} \;

build: clean
	-find work/marvin -name "*.py"  -not -name "__init__.py" -exec rm {} \;
	for var in $$(find work/marvin -name "*.ipynb" -not -path ".ipynb_checkpoints/*") ; \
	do \
			name=`echo $$var | cut -d'.' -f1`; \
			if [ ! -d "$$name" ]; then \
				jupyter nbconvert --to python --stdout $$name | tac | sed '0,/^$$/d' | tac > $$name.py; \
			fi \
	done
	find work -name "*.py" | xargs sed -i "s/get_ipython().magic('matplotlib inline')/matplotlib.use('agg')/"

lint: build
	flake8 --exclude=lib/,bin/ work/marvin

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
	docker run -it -d --name marvin_modeling --shm-size 2g -e PASSWORD=debug -p 28888:8888 \
		-v `pwd`/work:/home/creditx/work newreg.creditx.com/marvin/marvin_modeling:devel
