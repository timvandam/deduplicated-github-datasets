const {tokenize} = require('esprima');
const {cpus} = require('os');
const {access, readdir, mkdir, readFile} = require('fs/promises');
const {createWriteStream} = require('fs');
const zlib = require('zlib');
const path = require('path');
const cluster = require('cluster');
const {finished} = require('stream/promises');

const {createGzip} = zlib;

const cpuCount = cpus().length;

async function main() {
    if (process.argv.length != 4) {
        console.error(`Usage: ${process.argv[0]} ${process.argv[1]} INPUT_FOLDER OUTPUT_FOLDER`);
    }

    let inputDir = process.argv[2];
    let outputDir = process.argv[3];

    if (!inputDir || !outputDir) {
        console.error('You must provide an input and output directory');
        process.exit(1);
    }

    inputDir = path.resolve(process.cwd(), inputDir);
    outputDir = path.resolve(process.cwd(), outputDir);

    try {
        await access(inputDir);
    } catch (e) {
        console.error(`Input directory ${inputDir} does not exist or is not accessible: ${e.message}`);
        process.exit(1);
    }

    await mkdir(outputDir, {recursive: true});

    for (let i = 0; i < cpuCount; i++) {
        const worker = cluster.fork({
            BATCH_COUNT: cpuCount,
            BATCH_NUMBER: (i + 1).toString(),
            INPUT_DIR: inputDir,
            OUTPUT_DIR: outputDir,
        });

        worker.on('error', (error) => {
            console.error(`[WORKER${i + 1}] Exited with error: ${error.message}`)
        });
    }
}

const compareStrings = (a, b) => a > b ? 1 : b > a ? -1 : 0;

async function handleFiles() {
    const batchNumber = parseInt(process.env.BATCH_NUMBER);
    if (Number.isNaN(batchNumber)) {
        console.error(`Invalid batch number (${process.env.BATCH_NUMBER}). Worker was spawned incorrectly`)
        process.exit(1);
    }

    const batchCount = parseInt(process.env.BATCH_COUNT);
    if (Number.isNaN(batchCount)) {
        console.error(`Invalid batch count (${process.env.BATCH_COUNT}). Worker was spawned incorrectly`)
        process.exit(1);
    }

    const inputDir = process.env.INPUT_DIR;
    const outputDir = process.env.OUTPUT_DIR;

    if (!inputDir || !outputDir) {
        console.error(`Invalid input and/or output dirs: ${inputDir}, ${outputDir}. Worker was spawned incorrectly`)
        process.exit(1);
    }

    const outputFilePath = path.resolve(outputDir, `batch-${batchNumber}.json.gz`)

    const gzip = createGzip({
        chunkSize: 1024,
        flush: zlib.constants.Z_SYNC_FLUSH,
        level: 9,
        memLevel: 8,
        windowBits: 15,
    });

    let write = undefined;
    const writeStream = () => {
        if (write) {
            return write;
        }
        write = gzip.pipe(createWriteStream(outputFilePath, 'utf8'));
        return write;
    }

    let dirents = await readdir(inputDir, {withFileTypes: true});
    dirents.sort((a, b) => compareStrings(a.name, b.name));
    dirents = dirents.filter(dirent => dirent.isFile() && path.extname(dirent.name) === '.js')
    dirents = dirents.filter((_, i) => (i + (batchCount - batchNumber + 1)) % batchCount === 0)

    if (dirents.length === 0) {
        return;
    }

    console.log(`[WORKER${batchNumber}] Processing ${dirents.length} files`)

    for (const dirent of dirents) {
        const filePath = path.resolve(inputDir, dirent.name);
        const fileCode = await readFile(filePath, 'utf8');
        try {
            writeStream();
            const tokens = tokenize(fileCode).map(token => token.value);
            const obj = {filename: path.relative(inputDir, filePath), tokens};
            gzip.write(JSON.stringify(obj))
            gzip.write('\n')
        } catch (e) {
            // If tokenization errors we can just skip the file
        }
    }

    gzip.end();
    await finished(gzip);
    if (write) {
        await finished(write);
    }

    console.log(`[WORKER${batchNumber}] Finished processing ${dirents.length} files`)
}

if (cluster.isMaster) {
    main();
} else {
    handleFiles().then(() => {
        process.exit(0);
    });
}
