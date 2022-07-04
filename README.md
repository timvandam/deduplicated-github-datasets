# deduplicated-github-datasets
Repository for fetching datasets from the top-1000 GitHub repositories and deduplicating them

## Cloning this repository
```shell
git clone --recurse-submodules git@github.com:timvandam/deduplicated-github-datasets.git
```

## Prerequisite Runtimes
```shell
# install dotnet
# install java 17
```

## Fetching Repositories
```shell
java ./github-data-fetching/scrape python
java ./github-data-fetching/unzip ./repository-files
python ./near-duplicate-code-detector/tokenizers/python/tokenizepythoncorpus.py ./repository-files ./repository-files-tokens # use a tokenizer suitable for your language
dotnet run ./near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj --dir=./repository-files-tokens duplicate-files
python ./remove_dupes.py
```
