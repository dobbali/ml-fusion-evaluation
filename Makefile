init: clean dependencies

pylint:
	pipenv run pylint fusion_evaluation tests run.py

flake8:
	pipenv run flake8

mypy:
	pipenv run mypy -p fusion_evaluation -p tests --ignore-missing-imports

dependencies:
	pipenv install --dev

lint: flake8 pylint mypy

coverage:
	pipenv run pytest --cov-report term-missing --cov-report xml --cov=sma tests

package: # package the training image
	pipenv lock -r > requirements.txt
	#docker build --pull -f Dockerfile-train -t sma-obfuscator/train .
	#docker tag sma-obfuscator/train sma-obfuscator/train:latest
	rm requirements.txt
	#pipenv run python setup.py bdist_wheel

upload:
	pipenv run twine upload --repository-url https://acm.slicetest.com/artifactory/api/pypi/slice-pypi-local dist/*.whl

test:
	pipenv run pytest
