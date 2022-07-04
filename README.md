# deduplicated-github-datasets
Repository for fetching datasets from the top-1000 GitHub repositories and deduplicating them

## Cloning this repository
```shell
git clone --recurse-submodules git@github.com:timvandam/deduplicated-github-datasets.git
```

## Prerequisite Runtimes
[.NET Core 6](https://dotnet.microsoft.com/en-us/download/dotnet/6.0),
[Java 17](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html),
[NodeJS 16](https://nodejs.org/en/download/),
[Python 3.8](https://www.python.org/downloads/)
development kits and/or runtimes are required.

## Creating deduplicated datasets
The `create-js-dataset.sh` and `create-py-dataset.sh` shell scripts can be run to create deduplication JavaScript and Python datasets.
These scripts accept a [GitHub token](https://github.com/settings/tokens) as first positional argument.
This token is **required** to fetch the top-1000 repositories.
Doing it step by step can be done manually - simply have a look at the shell scripts and run what you want to run.

Creating datasets for other languages requires you to create a tokenizer for that language (see [tokenizers](./near-duplicate-code-detector/tokenizers)).
After that you can simply create a shell script similar to the existing ones, but adapter to whichever language you want.
