type RGB = [number, number, number];

export function getRGB(hexColor: string): RGB {
    return [
        Number.parseInt(hexColor.substr(1, 2), 16),
        Number.parseInt(hexColor.substr(3, 2), 16),
        Number.parseInt(hexColor.substr(5, 2), 16)
    ] as RGB
}

export function getHexFromBin(bin: number): string {
    const r = bin & 0xff;
    const g = (bin >> 8) & 0xff;
    const b = (bin >> 16) & 0xff;

    return `#${r.toString(16)}${g.toString(16)}${b.toString(16)}`;
}

export function getBinFromHex(hexColor: string) {
    const base = 0xff000000;
    const rgb = getRGB(hexColor);

    return base | (rgb[2] << 16) | (rgb[1] << 8) | (rgb[0]);
}

export function getLinearColorHex(a: string, b: string, weight: number): string {
    return getHexFromBin(getLinearColorBin(getBinFromHex(a), getBinFromHex(b), weight));
}

export function getLinearColorBin(a: number, b: number, weight: number): number {
    weight = Math.max(0, Math.min(1, weight))

    const r1 = (a >> 16) & 0xff;
    const r2 = (b >> 16) & 0xff;
    const g1 = (a >> 8) & 0xff;
    const g2 = (b >> 8) & 0xff;
    const b1 = a & 0xff;
    const b2 = b & 0xff;

    return 0xff000000 | (
        ((r2 - r1) * weight + r1) << 16 |
        ((g2 - g1) * weight + g1) << 8 |
        ((b2 - b1) * weight + b1));
}