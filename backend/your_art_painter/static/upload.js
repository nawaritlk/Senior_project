Dropzone.autoDiscover = false;

cinst myDropzone = new Dropzone("#my-dropzone", {
    url: "create/upload/",
    maxFiles: 1,
    acceptedFiles: '.png, .jpg',
})