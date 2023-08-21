export function promisify(fn, ...args) {
    return new Promise((resolve, reject) => {
        fn(...args, (err, data) => {
            if (err) reject(err)
            else resolve(data);
        })
    })
}

export function getMimeType(filename) {
    const extension = filename.split('.').pop().toLowerCase();

    switch (extension) {
        case 'png':
            return 'image/png';
        case 'jpg':
        case 'jpeg':
            return 'image/jpeg';
        case 'gif':
            return 'image/gif';
        default:
            return 'application/octet-stream';
    }
}