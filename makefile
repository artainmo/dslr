all:
	@echo "describe | histogram | scatter | pair | train | predict | test | LR | clean | env"

describe:
	python3 describe.py datasets/dataset_train.csv

histogram:
	python3 histogram.py datasets/dataset_train.csv

scatter:
	python3 scatter_plot.py datasets/dataset_train.csv

pair:
	python3 pair_plot.py datasets/dataset_train.csv

train:
	python3 logreg_train.py datasets/dataset_train.csv

predict:
	python3 logreg_predict.py datasets/dataset_test.csv datasets/theta.csv

test:
	cd test; python3 evaluate.py houses.csv; #subprocess goes into the test directory

LR: train predict test

clean:
	rm -rf __pycache__
	rm -rf ./utils/MyLogisticegression/__pycache__
	rm -rf ./utils/MyStats/__pycache__
	rm -rf ./utils/MyStats/stats/__pycache__

#pip3 and python3 should already be installed, tested on Python 3.7.7
env:
	pip3 install pandas
	pip3 install matplotlib
	pip3 install numpy
	pip3 install seaborn
	pip3 install sklearn

.PHONY: all describe histogram scatter pair train predict test LR clean env
