export async function* drop<T>(amount: number, it: AsyncIterable<T>): AsyncIterable<T> {
    const iterator = it[Symbol.asyncIterator]();
    while (true) {
        if (amount <= 0 || (await iterator.next()).done) break;
        amount--;
    }

    let result: IteratorResult<T>;
    while (!(result = await iterator.next()).done) {
        yield result.value;
    }
}

export async function* zip<A, B>(a: AsyncIterable<A>, b: AsyncIterable<B>): AsyncIterable<[A, B]> {
    const aIterator = a[Symbol.asyncIterator]();
    const bIterator = b[Symbol.asyncIterator]();

    let aValue;
    let bValue;

    while (true) {
        aValue = await aIterator.next();
        bValue = await bIterator.next();
        if (aValue.done || bValue.done) break;
        yield [aValue.value, bValue.value];
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
