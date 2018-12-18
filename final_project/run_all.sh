#!/bin/sh

for f in nn/*.py; do pipenv run python3 "$f"; done