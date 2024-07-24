SHELL := /bin/bash

.PHONY: all deploy db_init db_migrate db_upgrade

all: requirements.txt
	test -d .venv || python -m venv .venv
	source .venv/Scripts/activate && pip install -r requirements.txt
	touch .venv


db_init:
	source .venv/Scripts/activate && \
	flask db init

db_migrate:
	source .venv/Scripts/activate && \
	flask db migrate -m "Initial migration."

db_upgrade:
	source .venv/Scripts/activate && \
	flask db upgrade

db_setup: db_init db_migrate db_upgrade

db_update:
	source .venv/Scripts/activate && \
	flask db migrate -m "Add new column to database" && \
	flask db upgrade


develop:
	source .venv/Scripts/activate && \
	export FLASK_APP=app.py && \
	export FLASK_ENV=development && \
	export FLASK_DEBUG=1 && \
	flask run

deploy:
	source .venv/Scripts/activate && \
	waitress-serve --listen 0.0.0.0:5000 app:app
