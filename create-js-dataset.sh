#!/usr/bin/env bash

set -e

TOKEN=$1

cd ./github-data-fetching/incoder-analysis-java
gradle
gradle shadowJar
cd ./../../
cd ./near-duplicate-code-detector/tokenizers/javascript
npm i
cd ./../../../
mkdir -p ./javascript-dataset
java -jar ./github-data-fetching/incoder-analysis-java/build/libs/incoder-analysis-1.0-SNAPSHOT-all.jar -l javascript -e js -o ./javascript-dataset -t "$1"
node ./near-duplicate-code-detector/tokenizers/javascript/parser.js ./javascript-dataset/repository-files ./javascript-dataset/file-tokens
dotnet run --project ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj --dir=./javascript-dataset/file-tokens ./javascript-dataset/duplicate-files
python remove_dupes.py ./javascript-dataset