const fileInput = document.getElementById('fileInput');
const textInput = document.getElementById('textInput');
const speakerSelect = document.getElementById('speakerSelect');
const temperatureInput = document.getElementById('temperatureInput');
const penaltyInput = document.getElementById('penaltyInput');
const maxLengthInput = document.getElementById('maxLengthInput');
const generateBtn = document.getElementById('generateBtn');
const statusDiv = document.getElementById('status');
const outputCard = document.getElementById('outputCard');
const audioPlayer = document.getElementById('audioPlayer');

let currentAudioUrl = null;
const ttsWorker = new Worker('tts_worker_thread.js', { type: 'module' });

function cleanToEnglishNative(text) {
    // Standardizes whitespaces, strips decorative markdown and multi-line breaks for continuous processing
    return text.trim().normalize("NFD").replace(/\p{M}/gu, "").replace(/\s+/g, " ");
}

ttsWorker.onmessage = function(e) {
    const { status, data, audio, text, progress, ttsDurationSec, ttsDurationPerSentenceSec, ttsDurationPerWordSec, ttsDurationPerTokenSec, tokenCount, sentences, device} = e.data;

    if (status === "processing-unit-found") {
        statusDiv.innerText = device.toUpperCase() + " is detected. Loading OuteTTS AI Model components ...";
    } else if (status === "processing-unit-missing" || status === "gpu-or-model-init-error" || status === "speech-generation-error") {
        statusDiv.style.background = "pink";
        statusDiv.style.color = "red";
        statusDiv.innerText = data;
        generateBtn.disabled = true;
    } else if (status === "model-ready") {
        statusDiv.style.background = "lightblue";
        statusDiv.style.color = "black";
        statusDiv.innerText = "Model is loaded successfully into" +  " " + device.toUpperCase()  +  "!" +  " " + "Ready to generate speech.";
        generateBtn.disabled = false;
    } else if (status === "sentence-processing") {
        // Keeps user aware of progression down in the worker stack
        statusDiv.innerText = progress;
    } else if (status === "speech-generation-complete") {
        statusDiv.innerText = "Audio is successfully generated!";
        console.log("Number of sentences processed: ", sentences.length);
        console.log("Number of words processed: ", text.split(/\s+/).length);
        console.log("Number of tokens processed: ", tokenCount); 
        console.log("TTS generation duration in seconds: ", ttsDurationSec + " sec.");
        console.log("TTS generation duration per sentence in seconds: ", ttsDurationPerSentenceSec + " sec.");
         console.log("TTS generation duration per word in seconds: ", ttsDurationPerWordSec + " sec.");
        console.log("TTS generation duration per token in seconds: ", ttsDurationPerTokenSec + " sec.");

        // Revoke the old URL to prevent application memory leaks
        if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl);
        }
        
        currentAudioUrl = URL.createObjectURL(audio);
        audioPlayer.src = currentAudioUrl;
        outputCard.style.display = "block";
        generateBtn.disabled = false;
        generateBtn.innerText = "Generate Speech via WebGPU";
    }
};

// Handle Document File Parsing (.txt, .docx, .pdf)
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const fileExtension = file.name.split('.').pop().toLowerCase();

    try {
        if (fileExtension === 'txt') {
            const reader = new FileReader();
            reader.onload = function(e) { textInput.value = e.target.result; };
            reader.readAsText(file);
        } 
        else if (fileExtension === 'docx') {
            const arrayBuffer = await file.arrayBuffer();
            const result = await mammoth.extractRawText({ arrayBuffer: arrayBuffer });
            textInput.value = result.value;
        } 
        else if (fileExtension === 'pdf') {
            const arrayBuffer = await file.arrayBuffer();
    
            const pdf = await pdfjsLib.getDocument({data: arrayBuffer}).promise;
    
            let compiledText = "";
            
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map(item => item.str).join(" ");
                compiledText += pageText + "\n";
            }
            textInput.value = compiledText;
        }
        statusDiv.innerText = "File is parsed successfully. Ready for voice generation.";
    } catch (file_reading_error) {
        console.error(file_reading_error);
        alert("File reading error: " + file_reading_error);
        statusDiv.innerText = "Error when parsing file.";
    }
});

generateBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (!text) {
        alert("Please enter text or upload a txt, text-based pdf, or text-based word (docx/doc) file first.");
        return;
    }

    generateBtn.disabled = true;
    generateBtn.innerText = "Processing Audio Synthesis ...";


    ttsWorker.postMessage({
        text: cleanToEnglishNative(text),
        speaker_id: speakerSelect.value,
        temperature: parseFloat(temperatureInput.value),
        repetition_penalty: parseFloat(penaltyInput.value),
        max_length: parseInt(maxLengthInput.value) 
    });
});
