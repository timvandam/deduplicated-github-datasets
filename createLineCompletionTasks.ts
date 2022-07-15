import { relative, resolve } from 'path';
import { create } from 'random-seed';
import { access, readFile, mkdir } from 'fs/promises';
import { createReadStream, createWriteStream } from 'fs';
import { createInterface } from 'readline';

type CommandLineOption = {
    name: string;
    optional: boolean;
    defaultValue?: string;
    description: string;
}

const DESCRIPTION = 'Create JavaScript or TypeScript line completion tasks';
const OPTIONS: CommandLineOption[] = [
    { name: 'datasetFolder', optional: false, description: 'Folder for a dataset. Must contain ./sets/validation.csv and ./sets/test.csv' },
    { name: 'lineSplitRate', optional: true, defaultValue: '0.20', description: 'The rate of lines which should be used to create line completion tasks' },
    { name: 'lineSplitCap', optional: true, defaultValue: '5', description: 'An upper bound on the amount of lines used to create line completion tasks in a single file' },
    { name: 'seed', optional: true, defaultValue: '42', description: 'An optional seed that is used to deterministically create line completion tasks' },
];

// put all optionals at the end (stable wrt OPTIONS array)
let end = OPTIONS.length;
for (let i = 0; i < end; i++) {
    if (OPTIONS[i].optional) {
        OPTIONS.push(OPTIONS.splice(i, 1)[0]);
        i--;
        end--;
    }
}

const USAGE = `./node_modules/.bin/ts-node ./${relative(process.cwd(), __filename)} ${OPTIONS.map(({ name, optional }) => optional ? `[${name}]` : name).join(' ')}`;
const LONGEST_COMMAND_LENGTH = OPTIONS.map(({ name }) => name.length).reduce((a, b) => Math.max(a, b), 0);
const OPTIONS_DESCRIPTIONS = 'Options:' + '\n' + OPTIONS.map(({ name, description, defaultValue }) => `\t${name.padEnd(LONGEST_COMMAND_LENGTH, ' ')}\t\t${description}` + (defaultValue !== undefined ? ` (default: ${defaultValue})` : '')).join('\n');
const PROGRAM_HELP = DESCRIPTION + '\n\nUsage:\n\t' + USAGE + '\n\n' + OPTIONS_DESCRIPTIONS;

const argv = process.argv.slice(2);

function getOptionDict(): Record<string, string> {
    const options: Record<string, string> = {};

    for (let i = 0; i < OPTIONS.length; i++) {
        const { name, optional, defaultValue } = OPTIONS[i];
        const arg: string | undefined = argv[i];

        if (!optional && arg === undefined) {
            console.error(`You did not provide a value for the mandatory option '${name}'`);
            process.exit(1);
        }

        options[name] = arg ?? defaultValue;
    }

    return options;
}

type Options = {
    datasetFolder: string;
    lineSplitRate: number;
    lineSplitCap: number;
    seed: string;
};

async function getOptions(): Promise<Options> {
    const optionDict = getOptionDict();

    const options: Options = {
        datasetFolder: resolve(process.cwd(), optionDict.datasetFolder),
        lineSplitRate: parseFloat(optionDict.lineSplitRate),
        lineSplitCap: parseInt(optionDict.lineSplitRate),
        seed: optionDict.seed,
    };

    for (const path of [
        options.datasetFolder,
        resolve(options.datasetFolder, './sets'),
        resolve(options.datasetFolder, './sets/validation.csv'),
        resolve(options.datasetFolder, './sets/test.csv'),
        resolve(options.datasetFolder, './repository-files'),
    ]) {
        try {
            await access(path);
        } catch (e) {
            console.error(`Can not access '${path}'. Are you sure it exists? Error: ${e}`);
            process.exit(1);
        }
    }

    return options;
}

type Output = {
    leftContext: string;
    groundTruth: string;
    rightContext: string;
}

if (argv.length === 0) {
    console.log(PROGRAM_HELP);
    process.exit(0);
} else {
    main().then(() => process.exit(0));
}

async function main() {
    const options = await getOptions();

    const random = create(options.seed);

    for (const setName of ['validation', 'test']) {
        const readStream = createReadStream(resolve(options.datasetFolder, `./sets/${setName}.csv`));
        const lineStream = createInterface({ input: readStream, crlfDelay: Infinity });
        await mkdir(resolve(options.datasetFolder, './datasets'), { recursive: true });
        const writeStream = createWriteStream(resolve(options.datasetFolder, `./datasets/${setName}.jsonl`));

        const lineIterator = lineStream[Symbol.asyncIterator]();
        await lineIterator.next();
        for await (const line of lineIterator) {
            if (!line.length) continue;
            const [repository, file] = line.split(',');
            const filePath = resolve(options.datasetFolder, './repository-files', file);
            const fileContent = await readFile(filePath, 'utf8');
            const lines = fileContent.split('\n');
            const nonEmptyLineIndices = Array
                .from({length: lines.length}, (_, i) => i)
                .filter(i => lines[i].trim().length !== 0);

            if (nonEmptyLineIndices.length === 0) {
                continue;
            }

            const selectedLineCount = Math.max(1, Math.min(options.lineSplitCap, Math.floor(options.lineSplitRate * nonEmptyLineIndices.length)));

            const selectedLineIndices: number[] = [];
            for (let i = 0; i < selectedLineCount; i++) {
                const index = random.intBetween(0, nonEmptyLineIndices.length - 1);
                selectedLineIndices.push(nonEmptyLineIndices[index]);
                nonEmptyLineIndices.splice(index, 1);
            }

            const splits: [lineIndex: number, start: number][] = [];
            for (const lineIndex of selectedLineIndices) {
                const possibleStarts = Array
                    .from({length: lines[lineIndex].length}, (_, i) => i)
                    .filter(i => i < lines[lineIndex].length && lines[lineIndex].charAt(i) !== ' ');
                const start = possibleStarts[random.intBetween(0, possibleStarts.length - 1)];
                splits.push([lineIndex, start]);
            }

            for (const [lineIndex, start] of splits) {
                let leftContext = lines.slice(0, lineIndex).join('\n');
                if (lineIndex > 0) leftContext += '\n';
                leftContext += lines[lineIndex].slice(0, start);

                const groundTruth = lines[lineIndex].slice(start).trim();
                const rightContext = lines.slice(lineIndex + 1).join('\n');

                const output: Output = {leftContext, groundTruth, rightContext};
                const json = JSON.stringify(output);
                writeStream.write(json);
                writeStream.write('\n');
            }
        }

        writeStream.end();
    }
}

