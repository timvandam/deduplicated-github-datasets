#!/usr/bin/env bash

TOKEN="$1"
DATASET_FOLDER="$2"

set -e

cd ./github-data-fetching/incoder-analysis-java
gradle
gradle shadowJar
cd ./../../
cd ./near-duplicate-code-detector/tokenizers/javascript
npm i
cd ./../../../
mkdir -p "$DATASET_FOLDER"
java -jar ./github-data-fetching/incoder-analysis-java/build/libs/incoder-analysis-1.0-SNAPSHOT-all.jar -l javascript -e js -o "$DATASET_FOLDER" -s stars -t "$TOKEN"
node ./near-duplicate-code-detector/tokenizers/javascript/parser.js "$DATASET_FOLDER/repository-files" "$DATASET_FOLDER/file-tokens"
dotnet run --project ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj --dir="$DATASET_FOLDER/file-tokens" "$DATASET_FOLDER/duplicate-files"
pip install -r requirements.txt
python remove_dupes.py "$DATASET_FOLDER"
