function toFixed(value: number, fraction = 2) {
    let label = value.toFixed(fraction)
    const pointIndex = label.indexOf(".")
    let trimIndex = label.length;
    for (; trimIndex > pointIndex; trimIndex--) {
        if (label[trimIndex - 1] !== '0') break;
    }

    return label.slice(0, trimIndex);
}

export async function timeIt(fn: () => any | Promise<any>, label = "time_it", count = 1) {
    const t = performance.now();

    const times = [];
    for (let i = 0; i < count; i++) {
        const localT = performance.now();
        await fn();

        times.push(performance.now() - localT);
    }

    const totalTime = performance.now() - t;

    if (count === 1) {
        console.log(`*** ${label}:`, toFixed(totalTime / 1000, 4));
    } else {
        let mean = 0, max = Number.NEGATIVE_INFINITY, min = Number.POSITIVE_INFINITY;
        for (const t of times) {
            mean += t / count;
            max = Math.max(max, t);
            min = Math.min(min, t);
        }

        const d = ((max - min) / mean) * 50;

        console.log(
            `*** ${label} (x${count})`,
            "Mean:", toFixed(mean / 1000, 4),
            `± ${toFixed(d, 4)}%`,
            "Total:", toFixed(totalTime / 1000, 4),
        );
    }

    return totalTime / 1000;
}