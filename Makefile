SHELL:=/bin/bash

export FLASK_APP=app
start_backend:
	cd service
	flask run

build:
	pip3 install -r service/requirements.txt
	yarn install ui/

start_frontend:
	cd ui
	yarn start ui
