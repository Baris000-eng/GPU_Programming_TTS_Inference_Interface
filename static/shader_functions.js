import {
    WEB_GPU_SENTENCE_SPLITTER_SHADER, 
    WEB_GPU_VOICE_PROCESSOR_SHADER, 
    WEB_GPU_METRICS_SHADER, 
    WEB_GPU_WORD_AND_TOKEN_COUNT_SHADER
} from './shaders.js';

let webGpuDevice = null;                     // Stored WebGPU device instance


/**
 * Initializes the system's WebGPU logical adapter and device.
 * Implements a singleton pattern to avoid redundant resource creation.
 * * @returns {Promise<GPUDevice|null>} The active logical WebGPU device, or null if unsupported.
 */
export async function getWebGpuDevice() {
    if (webGpuDevice) {
        return webGpuDevice;
    }
    if (!navigator.gpu) {
        return null; // WebGPU API context not available in this browser engine
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        return null; // No compatible physical graphics card found matching default criteria
    }
    // Request logical connection to physical unit
    webGpuDevice = await adapter.requestDevice();
    return webGpuDevice;
}


// --- WGSL DISPATCH HELPERS ---

/**
 * Dispatches a parallel compute shader to identify punctuation marks and split input
 * text into structurally distinct sentences directly within GPU memory VRAM.
 * * @param {GPUDevice} device - Active logical WebGPU hardware interface.
 * @param {string} text - The raw prompt text to segment.
 * @returns {Promise<string[]>} Array of sliced sentence strings.
 */
export async function splitIntoSentencesWGSL(device, text) {
    const textLength = text.length;
    if (textLength === 0) {
        return []; 
    }

    // Convert JavaScript UTF-16 characters into a predictable 32-bit integer array representation
    const codePoints = new Uint32Array(textLength);
    for (let i = 0; i < textLength; i++) {
        codePoints[i] = text.charCodeAt(i);
    }

    // Compile raw WGSL source text module into binary executable shaders
    const shaderModule = device.createShaderModule({ code: WEB_GPU_SENTENCE_SPLITTER_SHADER });

    // UNIFORM BUFFER: Host-to-Device parameters (Pass text length scalar value)
    const gpuInfoBuffer = device.createBuffer({
        mappedAtCreation: true, size: 4, usage: GPUBufferUsage.UNIFORM
    });
    new Uint32Array(gpuInfoBuffer.getMappedRange())[0] = textLength;
    gpuInfoBuffer.unmap(); // Hand ownership over to the GPU timeline

    // STORAGE BUFFER: Input data array containing text character code points
    const gpuTextBuffer = device.createBuffer({
        mappedAtCreation: true, size: codePoints.byteLength, usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(gpuTextBuffer.getMappedRange()).set(codePoints);
    gpuTextBuffer.unmap();

    // STORAGE BUFFER: Output data destination array to hold boundary evaluation markers (1 = End of Sentence, 0 = Neutral)
    // Allocate 4 bytes per character to store uint32 flags.
    const gpuFlagsBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // STAGING BUFFER: Host-accessible memory layout used for mapping out results back to CPU timeline
    const gpuReadBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Setup compute pipeline topology using automatic layout determination
    const computePipeline = device.createComputePipeline({
        layout: "auto", compute: { module: shaderModule, entryPoint: "findSentenceBoundaries" }
    });

    // Bind layout slots to matching entry points inside the execution context
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gpuInfoBuffer } },
            { binding: 1, resource: { buffer: gpuTextBuffer } },
            { binding: 2, resource: { buffer: gpuFlagsBuffer } }
        ]
    });

    // Build linear execution command queue chain
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    // Dynamic calculate dimension grids based on uniform workgroup execution block steps of 256 threads
    passEncoder.dispatchWorkgroups(Math.ceil(textLength / 256) || 1);
    passEncoder.end();

    // Copy calculated flags sequence from VRAM straight into the accessible staging zone
    commandEncoder.copyBufferToBuffer(gpuFlagsBuffer, 0, gpuReadBuffer, 0, textLength * 4);
    device.queue.submit([commandEncoder.finish()]);

    // Asynchronously resolve hardware-to-host operations
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const flagsResult = new Uint32Array(gpuReadBuffer.getMappedRange());

    const sentences = [];
    let lastSliceIndex = 0;

    // Linear slice parser based on GPU generated markers arrays
    for (let i = 0; i < textLength; i++) {
        if (flagsResult[i] === 1) {
            sentences.push(text.slice(lastSliceIndex, i + 1));
            lastSliceIndex = i + 1;
        }
    }

    // Catch trailing tokens or clean unmapped blocks remaining
    if (lastSliceIndex < textLength) {
        const remaining = text.slice(lastSliceIndex);
        if (remaining.trim()) {
            sentences.push(remaining);
        }
    }

    // Housekeeping: Flush active allocations and map slots to mitigate context memory leaks
    gpuReadBuffer.unmap();
    gpuInfoBuffer.destroy(); gpuTextBuffer.destroy(); gpuFlagsBuffer.destroy(); gpuReadBuffer.destroy();

    return sentences;
}

/**
 * Dispatches a parallel compute shader across audio vector arrays to perform 
 * float signal scaling, normalization, and volume management tasks in VRAM.
 * @param {GPUDevice} device - Active logical WebGPU hardware interface.
 * @param {Float32Array} rawFloatSamples - Generated linear PCM audio vectors.
 * @param {number} targetGain - Volume multiplier scale coefficient.
 * @returns {Promise<Float32Array>} Modified linear float samples array.
 */
export async function processVoiceBufferWGSL(device, rawFloatSamples, targetGain = 1.0) {
    const sampleCount = rawFloatSamples.length;
    const shaderModule = device.createShaderModule({ code: WEB_GPU_VOICE_PROCESSOR_SHADER });

    // Mix structural datatypes safely using individual raw ArrayBuffer byte view mappings
    const infoBufferData = new ArrayBuffer(8);
    const infoView = new DataView(infoBufferData);
    infoView.setUint32(0, sampleCount, true);  // 4-byte uint element at offset 0
    infoView.setFloat32(4, targetGain, true);  // 4-byte float multiplier at offset 4

    // UNIFORM BUFFER: Configuration struct metadata
    const gpuInfoBuffer = device.createBuffer({
        mappedAtCreation: true, size: 8, usage: GPUBufferUsage.UNIFORM
    });
    new Uint8Array(gpuInfoBuffer.getMappedRange()).set(new Uint8Array(infoBufferData));
    gpuInfoBuffer.unmap();

    // STORAGE BUFFER: Original unscaled raw waveform vectors array
    const gpuInputBuffer = device.createBuffer({
        mappedAtCreation: true, size: rawFloatSamples.byteLength, usage: GPUBufferUsage.STORAGE
    });
    new Float32Array(gpuInputBuffer.getMappedRange()).set(rawFloatSamples);
    gpuInputBuffer.unmap();

    // STORAGE BUFFER: Destination buffer for scaled results
    const gpuOutputBuffer = device.createBuffer({
        size: rawFloatSamples.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // STAGING BUFFER: Host read mapping target
    const gpuReadBuffer = device.createBuffer({
        size: rawFloatSamples.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const computePipeline = device.createComputePipeline({
        layout: "auto", compute: { module: shaderModule, entryPoint: "processVoiceSamples" }
    });

    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gpuInfoBuffer } },
            { binding: 1, resource: { buffer: gpuInputBuffer } },
            { binding: 2, resource: { buffer: gpuOutputBuffer } }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(sampleCount / 256) || 1);
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(gpuOutputBuffer, 0, gpuReadBuffer, 0, rawFloatSamples.byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    // Explicit slice copy to pull memory away from structural backing allocations safely before unmapping
    const processedArray = new Float32Array(gpuReadBuffer.getMappedRange().slice());
    
    gpuReadBuffer.unmap();
    gpuInfoBuffer.destroy(); gpuInputBuffer.destroy(); gpuOutputBuffer.destroy(); gpuReadBuffer.destroy();

    return processedArray;
}

/**
 * Constructs a structural, compliant 44-byte standard RIFF WAV metadata container header 
 * around raw linear PCM 16-bit signed integer audio configurations data blocks.
 * @param {Float32Array} samples - Normalized floating point wave channels.
 * @param {number} sampleRate - Playback sampling frequency (defaults to OuteTTS standard 24kHz).
 * @returns {Uint8Array} Binary array container combining structured headers and raw sample tracks.
 */
export function encodeWavHeader(samples, sampleRate = 24000) {
    // 44 bytes dedicated metadata headers slot + (2 bytes per sample allocation for Int16 conversion)
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF Identifier Group */
    view.setUint32(0, 0x52494646, false); // Big-Endian descriptor "RIFF"
    view.setUint32(4, 36 + samples.length * 2, true); // Chunk remaining file payload size limit metric
    view.setUint32(8, 0x57415645, false); // Big-Endian descriptor "WAVE"

    /* Sub-chunk 1: Format Details Context Block */
    view.setUint32(12, 0x666d7420, false); // Big-Endian header token "fmt"
    view.setUint32(16, 16, true);          // Size metrics configuration for standard Linear PCM blocks
    view.setUint16(20, 1, true);           // Audio format categorization tracking ID (1 = Uncompressed PCM)
    view.setUint16(22, 1, true);           // Total dynamic audio recording output tracks channels (1 = Mono)
    view.setUint32(24, sampleRate, true);  // Sampling rate metric configuration parameter
    view.setUint32(28, sampleRate * 2, true); // Byte rate computation calculation: SampleRate * Channels * (BitsPerSample / 8)
    view.setUint16(32, 2, true);           // Block alignment layout specification metric (Channels * BytesPerSample)
    view.setUint16(34, 16, true);          // Audio processing resolution metrics (16 bits depth quantizes)

    /* Sub-chunk 2: Audio Track Data Core Payload Element */
    view.setUint32(36, 0x64617461, false); // Big-Endian label "data"
    view.setUint32(40, samples.length * 2, true); // Core payload byte structure track tracking limits

    // Compress 32-bit float vectors [-1.0, 1.0] securely into 16-bit integer space bounds
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i])); // Hard dynamic clamping check guard logic
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); // Signed conversions scale operations
        offset += 2;
    }
    return new Uint8Array(buffer);
}

/**
 * Dispatches a parallel math metric shader module loop across system evaluation tracks
 * to compute duration ratios alongside parallel efficiency scores.
 * @param {GPUDevice} device - Active logical WebGPU hardware interface.
 * @param {number} totalDurationMs - Absolute runtime length computed on host side tracking.
 * @param {number} sentenceCount - Generated sentences tracking dimensions metrics.
 * @param {number} wordCount - Absolute total word allocations tracked across text maps.
 * @param {number} tokenCount - Structural inference matrix dimension counts.
 * @returns {Promise<Object>} Formatted evaluation tracking object strings.
 */
export async function processMetricsWGSL(device, totalDurationMs, sentenceCount, wordCount, tokenCount) {
    const shaderModule = device.createShaderModule({ code: WEB_GPU_METRICS_SHADER });
    const inputData = new Float32Array([totalDurationMs, sentenceCount, wordCount, tokenCount]);

    // UNIFORM BUFFER: Input execution pipeline parameters arrays
    const gpuInputBuffer = device.createBuffer({
        mappedAtCreation: true, size: inputData.byteLength, usage: GPUBufferUsage.UNIFORM
    });
    new Float32Array(gpuInputBuffer.getMappedRange()).set(inputData);
    gpuInputBuffer.unmap();

    // STORAGE BUFFER: Core results metrics destinations matrices array
    const gpuOutputBuffer = device.createBuffer({
        size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // STAGING BUFFER: Read validation target
    const gpuReadBuffer = device.createBuffer({
        size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const computePipeline = device.createComputePipeline({
        layout: "auto", compute: { module: shaderModule, entryPoint: "computeMetrics" }
    });

    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gpuInputBuffer } },
            { binding: 1, resource: { buffer: gpuOutputBuffer } }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1); // Single invocation workgroup item processing sequence
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(gpuOutputBuffer, 0, gpuReadBuffer, 0, 16);
    device.queue.submit([commandEncoder.finish()]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(gpuReadBuffer.getMappedRange());
    
    // Convert mathematical parameters into floating point string representations
    const output = {
        ttsDurationSec: results[0].toFixed(3),
        ttsDurationPerSentenceSec: results[1].toFixed(3),
        ttsDurationPerWordSec: results[2].toFixed(3),
        ttsDurationPerTokenSec: results[3].toFixed(3)
    };

    gpuReadBuffer.unmap();
    gpuInputBuffer.destroy(); gpuOutputBuffer.destroy(); gpuReadBuffer.destroy();

    return output;
}

/**
 * Dispatches a multi-binding parallel shader module targeting character space distributions
 * to accurately measure word gaps alongside contextual separation boundaries.
 * @param {GPUDevice} device - Active logical WebGPU hardware interface.
 * @param {string} text - Raw input processing text.
 * @returns {Promise<Object>} Cumulative semantic element extraction measurements tracking structures.
 */
export async function processWordAndTokenCountWGSL(device, text) {
    const textLength = text.length;
    if (textLength === 0) {
        return { wordCount: 0, tokenCount: 0 };
    }

    const codePoints = new Uint32Array(textLength);
    for (let i = 0; i < textLength; i++) {
        codePoints[i] = text.charCodeAt(i);
    }

    const shaderModule = device.createShaderModule({ code: WEB_GPU_WORD_AND_TOKEN_COUNT_SHADER });

    // UNIFORM BUFFER: Scalar size array allocation mapping configurations
    const gpuInfoBuffer = device.createBuffer({
        mappedAtCreation: true, size: 4, usage: GPUBufferUsage.UNIFORM
    });
    new Uint32Array(gpuInfoBuffer.getMappedRange())[0] = textLength;
    gpuInfoBuffer.unmap();

    // STORAGE BUFFER: Primary source character code arrays metrics tracks
    const gpuTextBuffer = device.createBuffer({
        mappedAtCreation: true, size: codePoints.byteLength, usage: GPUBufferUsage.STORAGE
    });
    new Uint32Array(gpuTextBuffer.getMappedRange()).set(codePoints);
    gpuTextBuffer.unmap();

    // STORAGE BUFFER: Word boundary tracking markers storage positions
    const gpuWordFlagsBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // STORAGE BUFFER: Token evaluation markers processing layouts arrays
    const gpuTokenFlagsBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // STAGING BUFFER: Read allocation target tracker targeting calculated words data structures
    const gpuReadWordBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // STAGING BUFFER: Read allocation target tracker targeting calculated tokens data structures
    const gpuReadTokenBuffer = device.createBuffer({
        size: textLength * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const computePipeline = device.createComputePipeline({
        layout: "auto", compute: { module: shaderModule, entryPoint: "countWordsAndTokens" }
    });

    // Map structural data buffers targets configuration matrix properties fields
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gpuInfoBuffer } },
            { binding: 1, resource: { buffer: gpuTextBuffer } },
            { binding: 2, resource: { buffer: gpuWordFlagsBuffer } },
            { binding: 3, resource: { buffer: gpuTokenFlagsBuffer } }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(textLength / 256) || 1);
    passEncoder.end();

    // Parallel extraction copying maps execution paths logic patterns tracks step
    commandEncoder.copyBufferToBuffer(gpuWordFlagsBuffer, 0, gpuReadWordBuffer, 0, textLength * 4);
    commandEncoder.copyBufferToBuffer(gpuTokenFlagsBuffer, 0, gpuReadTokenBuffer, 0, textLength * 4);
    device.queue.submit([commandEncoder.finish()]);

    // Concurrently await multiple read-map confirmation notifications streams
    await Promise.all([
        gpuReadWordBuffer.mapAsync(GPUMapMode.READ),
        gpuReadTokenBuffer.mapAsync(GPUMapMode.READ)
    ]);

    const wordFlagsResult = new Uint32Array(gpuReadWordBuffer.getMappedRange());
    const tokenFlagsResult = new Uint32Array(gpuReadTokenBuffer.getMappedRange());
    
    let wordCount = 0;
    let tokenCount = 0;

    // Linearly reduce mapping elements into single scalar tracking aggregations
    for (let i = 0; i < textLength; i++) {
        wordCount += wordFlagsResult[i];
        tokenCount += tokenFlagsResult[i];
    }

    gpuReadWordBuffer.unmap(); gpuReadTokenBuffer.unmap();
    gpuInfoBuffer.destroy(); gpuTextBuffer.destroy(); gpuWordFlagsBuffer.destroy();
    gpuTokenFlagsBuffer.destroy(); gpuReadWordBuffer.destroy(); gpuReadTokenBuffer.destroy();

    return { wordCount, tokenCount };
}