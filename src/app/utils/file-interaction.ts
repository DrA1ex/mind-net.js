export function openFile(contentType: string, multiple: boolean) {
    return new Promise(resolve => {
        let input: HTMLInputElement = document.createElement('input');
        input.type = 'file';
        input.multiple = multiple;
        input.accept = contentType;

        input.onchange = () => {
            const files = Array.from(input.files || []);
            if (multiple) {
                resolve(files);
            } else {
                resolve(files[0]);
            }

            input.remove();
        };

        input.click();
    });
}

export function saveFile(content: any, fileName: string, contentType: string) {
    const a = document.createElement("a");
    const file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();

    setTimeout(() => a.remove(), 0);
}
