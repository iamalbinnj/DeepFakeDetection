const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const resultContainer = document.getElementById("resultContainer");
const videoPlayer = document.getElementById("videoPlayer");
const analysisResult = document.getElementById("analysisResult");
const loader = document.getElementById("loader");
const analyzeButton = document.getElementById("analyzeButton");

dropArea.addEventListener("click", () => fileInput.click());

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

// Get references to DOM elements
const uploadForm = document.getElementById('uploadForm');


// Enable or disable the analyze button based on file input
fileInput.addEventListener('change', () => {
    analyzeButton.disabled = !fileInput.files.length;
});

// Handle video upload and analysis
analyzeButton.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    loader.style.display = 'block';
    analyzeButton.disabled = true;

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData,
        });

        // Check if response is ok
        if (!response.ok) {
            const errorText = await response.text(); // Get error text for debugging
            console.error(`Error response from server: ${errorText}`);
            throw new Error(`Error: ${response.statusText}`);
        }

        const result = await response.json();
        
        // Log result for debugging
        console.log('Received result:', result);

        // Ensure we have a valid result before displaying it
        if (result && result.Result && typeof result.Confidence === 'number') {
            // Display video and analysis result
            videoPlayer.src = URL.createObjectURL(file);
            analysisResult.textContent = `Result: ${result.Result}, Confidence: ${result.Confidence.toFixed(2)}`;
            
            // Show result container
            resultContainer.style.display = 'block';
        } else {
            console.error('Unexpected response format:', result);
            analysisResult.textContent = 'Unexpected response format.';
        }

    } catch (error) {
        console.error('Error during video analysis:', error);
        analysisResult.textContent = 'An error occurred during analysis. Please try again.';
    } finally {
        loader.style.display = 'none';
        analyzeButton.disabled = false;
    }
});