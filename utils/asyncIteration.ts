export async function* drop<T>(amount: number, it: AsyncIterable<T>): AsyncIterable<T> {
    if (amount <= 0) yield* it;

    let i = 0;
    for await (const _ of it) {
        i++;
        if (i >= amount) break;
    }

    yield* it;
}

export async function* zip<A, B>(a: AsyncIterable<A>, b: AsyncIterable<B>): AsyncIterable<[A, B]> {
    const aIterator = a[Symbol.asyncIterator]();
    const bIterator = b[Symbol.asyncIterator]();

    let aValue = await aIterator.next();
    let bValue = await bIterator.next();

    while (!aValue.done && !bValue.done) {
        yield [aValue.value, bValue.value];
        aValue = await aIterator.next();
        bValue = await bIterator.next();
    }
}

export async function* range(start: number, end: number): AsyncIterable<number> {
    for (let i = start; i < end; i++) {
        yield i;
    }
}

export function enumerate<T>(it: AsyncIterable<T>): AsyncIterable<[number, T]> {
    return zip(range(0, Infinity), it);
}