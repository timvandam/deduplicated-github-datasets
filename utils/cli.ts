import { relative } from "path";

export type CommandLineOption = {
    name: string;
    optional: boolean;
    defaultValue?: string;
    description: string;
}

/**
 * Returns options provided by the user, or shows the user an overview of how to use the cli.
 * @param description Program description
 * @param options List of possible positional arguments
 * @param fileName The name of the file the CLI is located in (used for displaying how to run it)
 */
export function cli(description: string, options: CommandLineOption[], fileName: string): Record<string, string> {
    // put all optionals at the end (stable w.r.t. options array)
    let end = options.length;
    for (let i = 0; i < end; i++) {
        if (options[i].optional) {
            options.push(options.splice(i, 1)[0]);
            i--;
            end--;
        }
    }

    const positionalArguments = options.map(({ name, optional }) => optional ? `[${name}]` : name).join(' ');
    const usage = `${fileName.endsWith('.ts') ? 'ts-' : ''}node ${relative(process.cwd(), fileName)} ${positionalArguments}`;
    const longestCommandLength = options.reduce((max, { name }) => Math.max(max, name.length), 0);
    const optionDescriptions = 'Options:' + '\n' + options.map(({ name, description, defaultValue }) => `\t${name.padEnd(longestCommandLength, ' ')}\t\t${description}` + (defaultValue !== undefined ? ` (default: ${defaultValue})` : '')).join('\n');
    const help = description + '\n\nUsage:\n\t' + usage + '\n\n' + optionDescriptions;

    const argv = process.argv.slice(2);

    if (argv.length === 0) {
        console.log(help);
        process.exit(0);
    }

    const result: Record<string, string> = {};

    for (let i = 0; i < options.length; i++) {
        const { name, optional, defaultValue } = options[i];
        const arg: string | undefined = argv[i];

        if (!optional && arg === undefined) {
            console.error(`You did not provide a value for the mandatory option '${name}'`);
            process.exit(1);
        }

        result[name] = arg ?? defaultValue;
    }

    return result;
}
