/**
 * @file OuteTTS WebGPU-Accelerated Text-to-Speech Web Worker
 * * This worker manages an asynchronous background thread that loads an ONNX-based
 * Text-to-Speech (TTS) model via the OuteTTS library. It offloads intensive text splitting,
 * audio gain scaling, counter tracking, and benchmark metrics execution straight to 
 * the underlying hardware GPU using WebGPU Compute Shaders Language (WGSL).
 */

import { HFModelConfig_v1, InterfaceHF } from "https://esm.sh/outetts";
import { mergeWavBuffers } from './wav_utils.js';
import { getWebGpuDevice, splitIntoSentencesWGSL, processVoiceBufferWGSL, processMetricsWGSL, processWordAndTokenCountWGSL, encodeWavHeader} from './shader_functions.js';

// Global singletons for maintaining state across thread executions
let tts_interface = null;                    // Holds the instantiated OuteTTS inference engine
const model_name = "onnx-community/OuteTTS-0.2-500M"; // Remote Hugging Face model hub identifier
const language_name = "en";                  // Language configuration profile
const device_name = "webgpu";                // Hardware execution device for the underlying ONNX runtime
const data_type = "q4";                      // Quantization level (4-bit integer weights for memory savings)

// -- INIT -- 

/**
 * This function validates environmental WebGPU support before fetching and 
 * initializing the OuteTTS inference framework backend engine configurations.
 */
async function initializeWebGpuAndTTSModel() {
    try {
        const device = await getWebGpuDevice();
        if (!device) {
            self.postMessage({ status: "processing-unit-missing", data: "WebGPU is not supported." });
            return;
        }
        self.postMessage({status: "processing-unit-found" , device: device_name});

        // Initialize target ONNX model definition profiles metadata configurations structure layout mapping rules
        const model_config = new HFModelConfig_v1({
            model_path: model_name, language: language_name, dtype: data_type, device: device_name, 
        });

        // Instantiate inference pipeline interface hooks engine execution module links targets parameters
        tts_interface = await InterfaceHF({ model_version: "0.2", cfg: model_config});
        self.postMessage({status: "model-ready", device: device_name});
    } catch (tts_error) {
        self.postMessage({status: "gpu-or-model-init-error", data: tts_error.toString()});
    }
}

initializeWebGpuAndTTSModel();

// --- EXECUTION THREAD ---

/**
 * Background thread worker interface message receiver routing block logic handler.
 * Listens for requests, triggers pipeline operations, and returns processed speech structures.
 */
self.addEventListener("message", async (e) => {
    // Drop thread tracking processing metrics queries requests if framework context structures are missing
    if (!tts_interface) return;
    
    const { text, speaker_id, temperature, repetition_penalty, max_length } = e.data;
    try {
        if (!text) {
            return;
        }

        const startTime = performance.now();
        let speaker = tts_interface.load_default_speaker(speaker_id);
        const device = await getWebGpuDevice();

        // Compute absolute sentence splits using our parallel WGSL layout
        const sentences = await splitIntoSentencesWGSL(device, text);
        const wavBuffers = [];
        
        // Model generation execution loop
        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i].trim();
            if (!sentence) {
                continue;
            }

            // Report incremental chunk parsing process milestones state metrics to host layer UI hooks
            self.postMessage({ 
                status: "sentence-processing", 
                progress: `Processing sentence ${i+1} of ${sentences.length} ...` 
            });

            // Trigger core ONNX neural network weights calculations inside WebGPU execution contexts
            const audioOutputObject = await tts_interface.generate({
                text: sentence, temperature, repetition_penalty, max_length, speaker
            });

            
            const rawSamples = audioOutputObject.audio.data;

            // Pass the float vectors straight into our WGSL voice compute shader module
            const processedGpuSamples = await processVoiceBufferWGSL(device, rawSamples, 1.0);

            // Construct structural PCM data elements and inject the 44-byte RIFF metadata header
            const voiceBuffer = encodeWavHeader(processedGpuSamples, 24000); 
            wavBuffers.push(voiceBuffer);
        }

        // Merge structured voice blocks together safely using mergeWavBuffers
        const finalAudioBlob = mergeWavBuffers(wavBuffers);

        const endTime = performance.now();
        const totalDurationMs = endTime - startTime;
        
        // Run word and token counting WGSL shader 
        const { wordCount, tokenCount } = await processWordAndTokenCountWGSL(device, text);

        // Run TTS evaluation metrics calculation WGSL shader 
        const gpuStats = await processMetricsWGSL(device, totalDurationMs, sentences.length, wordCount, tokenCount);

        // Transmit the ultimate complete audio blob, merging wav buffers, alongside runtime benchmarking calculations, back to the parent controller
        self.postMessage({
            status: "speech-generation-complete",
            audio: finalAudioBlob,
            text: text,
            ttsDurationSec: gpuStats.ttsDurationSec,
            ttsDurationPerSentenceSec: gpuStats.ttsDurationPerSentenceSec,
            ttsDurationPerWordSec: gpuStats.ttsDurationPerWordSec,
            ttsDurationPerTokenSec: gpuStats.ttsDurationPerTokenSec,
            tokenCount: tokenCount,
            sentences: sentences, 
            device: device_name
        });
    } catch (tts_error) {
        self.postMessage({ status: "speech-generation-error", data: tts_error.toString() });
    }
});