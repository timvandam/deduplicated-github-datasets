export async function* drop<T>(amount: number, it: AsyncIterable<T>): AsyncIterable<T> {
    if (amount <= 0) yield* it;

    let i = 0;
    for await (const _ of it) {
        i++;
        if (i >= amount) break;
    }

    yield* it;
}