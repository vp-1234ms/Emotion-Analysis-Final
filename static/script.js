function checkFileType() {
    var fileType = document.querySelector('input[name="fileType"]:checked').value;
    var fileInput = document.getElementById('fileInput');
    var processBtn = document.getElementById('processBtn');

    var allowedExtensions = {
        'image': ['.jpg', '.jpeg', '.png'],
        'audio': ['.mp3', '.wav'],
        'video': ['.mp4']
    };

    var fileName = fileInput.value.toLowerCase();
    var extension = fileName.substring(fileName.lastIndexOf('.'));

    if (allowedExtensions[fileType].includes(extension)) {
        processBtn.disabled = false;
    } else {
        alert('Invalid file format. Please select a valid file.');
        processBtn.disabled = true;
    }
}
