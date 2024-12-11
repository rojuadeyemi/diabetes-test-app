# Specify shell for cross-platform compatibility
SHELL := /bin/bash

.PHONY: all deploy db_init db_migrate db_upgrade db_setup db_update develop

# Detect OS and set Python command
ifeq ($(OS),Windows_NT)
    PYTHON_CMD = py -3.11
    ACTIVATE_CMD = .venv/Scripts/activate
else
    PYTHON_CMD = python3.11
    ACTIVATE_CMD = source .venv/bin/activate
endif

all: requirements.txt
	test -d .venv || $(PYTHON_CMD) -m venv .venv
	$(ACTIVATE_CMD) && pip install -r requirements.txt

# Database setup
db_init:
	$(ACTIVATE_CMD) && flask db init

db_migrate:
	$(ACTIVATE_CMD) && flask db migrate -m "Initial migration."

db_upgrade:
	$(ACTIVATE_CMD) && flask db upgrade

db_setup: db_init db_migrate db_upgrade

db_update:
	$(ACTIVATE_CMD) && \
	flask db migrate -m "Add new column to database" && \
	flask db upgrade

# Development server
develop:
	$(ACTIVATE_CMD) && \
	export FLASK_APP=app.py && \
	export FLASK_ENV=development && \
	export FLASK_DEBUG=1 && \
	flask run

# Deployment server
deploy:
	$(ACTIVATE_CMD) && \
	waitress-serve --listen=0.0.0.0:5000 app:app
