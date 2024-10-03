// Get references to DOM elements
const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const videoPreview = document.getElementById("videoPreview");
const videoPlayer = document.getElementById("videoPlayer");
const analyzeButton = document.getElementById("analyzeButton");
const loader = document.getElementById("loader");
const resultContainer = document.getElementById("resultContainer");
const analysisResult = document.getElementById("analysisResult");

// Function to handle file selection
function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('video/')) {
            fileInput.files = files;
            displayVideoPreview(file);
            analyzeButton.disabled = false;
        } else {
            alert('Please select a valid video file.');
        }
    }
}

// Function to display video preview
function displayVideoPreview(file) {
    const fileURL = URL.createObjectURL(file);
    videoPlayer.src = fileURL;
    videoPreview.style.display = 'block';
}

// Click event to open file input dialog
dropArea.addEventListener("click", () => fileInput.click());

// Drag and drop events
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = "#f0f0f0";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.backgroundColor = "";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = "";
    handleFiles(e.dataTransfer.files);
});

// File input change event
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Handle video analysis
analyzeButton.addEventListener('click', function() {
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    loader.style.display = 'block';
    analyzeButton.disabled = true;
    resultContainer.style.display = 'none';

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loader.style.display = 'none';
        analyzeButton.disabled = false;
        resultContainer.style.display = 'block';
        analysisResult.textContent = `Result: ${data.Result}, Confidence: ${data.Confidence}`;
    })
    .catch(error => {
        loader.style.display = 'none';
        analyzeButton.disabled = false;
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
    });
});