## Name
Human-Centric and WebGPU-Powered Text-To-Speech Inference Interface Project 

## Description
This project enables users to interact with the TTS inference pipeline, and generate voices for the user-written or user-uploaded 
content. A user can either type/copy-paste text content in the textbox area, or upload a text-based PDF, Word, or txt file. It is especially useful for the elderly and/or deaf people. 

## Installation
Firstly, we need to install the outetts library from the NPM package manager using the command below: 
npm i outetts

"outetts" is the library used in the text-to-speech process. 

## How to activate WebGPU API in Google Chrome?

1. Open Google Chrome
2. Type 'chrome://flags/' in the Google Chrome Search Bar, and press 'Enter'.
3. Type "WebGPU" to the search bar where we search for the flags

4. Enable the Features: Find 'Unsafe WebGPU Support' and 'Web Developer Features'. Select 'Enabled' from the respective drop-down menu. 

5. Restart Google Chrome 
("Type 'chrome://restart' to the Google Chrome search bar and press 'Enter'" or "Restart the whole computer and reopen the Google Chrome")

## How to Run? 

In order to run the project, we need to execute the following commands: 

1. cd TTS_Inference_Interface_Project
2. Then, we need to create a virtual environment. In order to do this, we can use venv or conda. 

Each command below will create a virtual environment named 'tts_env'.

venv: python (python_python_version) -m venv tts_env 

conda: conda create --name tts_env

In order to activate the virtual environment created, we can use one of the following commands: 

venv: source tts_env/bin/activate (Macos or Linux) or .\tts_env\bin\activate (Windows)

conda: conda activate tts_env 

Now, we need to install all library dependencies from the requirements.txt file. 

In order to install them, we can use one of the following commands depending on the package manager used. 

venv: pip install -r requirements.txt 

conda: conda install --file requirements.txt

In order to deactivate the virtual environment created, we can use one of the following commands: 

venv: deactivate (Macos, Linux, or Windows) 

conda: conda deactivate 

3. python app.py

## Results: 
As expected, the WebGPU-powered version of this project highly exceeds the WebAssembly-powered one with 
less inference time in total, per word, and per token (approximately 1/8'th of the other). 

<img width="1016" height="641" alt="Ekran Resmi 2026-07-19 12 57 38" src="https://github.com/user-attachments/assets/f93897f5-8fbc-41fb-b03c-4967c72f8213" />
<img width="1029" height="648" alt="Ekran Resmi 2026-07-19 12 58 04" src="https://github.com/user-attachments/assets/74de379d-fbb9-4848-b6a1-61ef5118fcc9" />

## Contributing
This project is open to the further contributions, improving the UI appeal, increasing the set of functionalities, or integrating into a bigger-scale software. 
