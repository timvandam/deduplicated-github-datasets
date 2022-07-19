import { createReadStream } from 'fs';
import { createInterface } from 'readline';

export async function* createReadLineStream(filePath: string): AsyncGenerator<string> {
  const readStream = createReadStream(filePath, 'utf8');
  const lineStream = createInterface({ input: readStream, crlfDelay: Infinity });
  for await (const line of lineStream) {
    if (line.trim().length > 0) yield line;
  }
}
