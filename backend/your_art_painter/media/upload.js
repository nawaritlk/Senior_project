Dropzone.autoDiscover = false;

const myDropzone = new Dropzone("#my-dropzone", {
    url: "create/upload/",
    maxFiles: 1,
    acceptedFiles: '.png, .jpg',
    complete: function() {
        alert('Your file was successfully uploaded!');
    }
})