#!/usr/bin/env bash

set -e

TOKEN=$1

cd ./github-data-fetching/incoder-analysis-java
gradle
gradle shadowJar
cd ./../../
cd ./near-duplicate-code-detector/tokenizers/python
pip install -r requirements.txt
cd ./../../../
mkdir -p ./python-dataset
java -jar ./github-data-fetching/incoder-analysis-java/build/libs/incoder-analysis-1.0-SNAPSHOT-all.jar -l python -e py -o ./python-dataset -t "$1"
python ./near-duplicate-code-detector/tokenizers/python/tokenizefiles.py ./python-dataset/repository-files ./python-dataset/file-tokens
dotnet run --project ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj --dir=./python-dataset/file-tokens ./python-dataset/duplicate-files
python remove_dupes.py ./python-dataset
