
.venv: 
	(which python3 && echo "Python3 found") || (echo "Please Install python3" && exit 1) &&\
	(python3 -m venv .venv )

.PHONY: deps
deps: 
	. .venv/bin/activate  && pip install pyspark==3.5.0 numpy

.PHONY: prepare-env 
prepare-env: .venv deps

.PHONY: wheel
wheel: prepare-env
	. .venv/bin/activate && python3 setup.py bdist_wheel

.PHONY: tests
tests: prepare-env 
	. .venv/bin/activate && \
	pip install . && \
	cd ./sparkxgb/tests && \
	python3 -m unittest classifier.py 

.PHONY: clean
clean:
	rm -rf dist
	rm -rf build
	pip uninstall -y spark-xgboost
	rm -rf tests/__pycache__
	rm -rf testing/__pycache__

.PHONY: remove-env
remove-env: clean
	rm -rf .venv	
