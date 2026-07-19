// --- SHADERS ---

/**
 * WGSL Sentence Splitter Shader
 * Scans string buffers in parallel to map absolute sentence boundaries.
 * * Mechanics:
 * Assigns one invocation per character array element. Each thread inspects its 
 * designated character to determine if it marks the end of a sentence.
 * Out-of-bounds reads are explicitly caught via uniform buffer length validation.
 */
export const WEB_GPU_SENTENCE_SPLITTER_SHADER = `
    struct ScalarInput {
        textLength : u32, // Total number of valid characters initialized in the storage buffer
    }

    // Bind Group Layout:
    // Binding 0: Uniform buffer containing read-only configuration scalars
    // Binding 1: Read-only storage array containing raw UTF-32 / ASCII scalar character values
    // Binding 2: Read-write storage array mapping boolean flags (0u or 1u) per character position
    @group(0) @binding(0) var<uniform> info : ScalarInput;
    @group(0) @binding(1) var<storage, read> textBuffer : array<u32>;
    @group(0) @binding(2) var<storage, read_write> sentenceFlags : array<u32>;

    // Helper: Evaluates if a given character code maps to terminal sentence punctuation.
    // ASCII values: 46 = '.', 33 = '!', 63 = '?'
    fn isPunctuation(code: u32) -> bool {
        return code == 46u || code == 33u || code == 63u;
    }

    // Helper: Evaluates if a given character code matches standard whitespace definitions.
    // ASCII values: 32 = Space, 9 = Horizontal Tab, 10 = Line Feed (\\n), 13 = Carriage Return (\\r)
    fn isSpace(code: u32) -> bool {
        return code == 32u || code == 9u || code == 10u || code == 13u;
    }

    @compute @workgroup_size(256)
    fn findSentenceBoundaries(@builtin(global_invocation_id) id : vec3<u32>) {
        let index = id.x;
        
        // Guard clause: Prevents processing padding threads beyond the actual text length.
        if (index >= info.textLength) { 
            return; 
        }

        let current_char = textBuffer[index];
        
        // Sentence Evaluation Logic:
        // A character terminates a sentence if it is terminal punctuation AND either:
        // 1. It is the absolute last character in the buffer, OR
        // 2. It is immediately followed by a trailing whitespace element.
        if (isPunctuation(current_char)) {
            if (index == info.textLength - 1u) {
                sentenceFlags[index] = 1u; // Flag set: End of document terminal boundary discovered
                return;
            } else if (isSpace(textBuffer[index + 1u])) {
                sentenceFlags[index] = 1u; // Flag set: Interstitial sentence boundary discovered
                return;
            }
            
            sentenceFlags[index] = 0u; // Flag cleared: Punctuation that does not terminate a sentence (e.g., decimal points, commas, and so on.)
        }
        sentenceFlags[index] = 0u; // Flag cleared: Internal sentence or structural character
    }
`;

/**
 * WGSL Voice Processor Shader
 * Executes parallel linear audio sample transformation and signal clipping.
 * * Mechanics:
 * Iterates through raw PCM float arrays to apply real-time amplitude scaling. 
 * Includes clamping to enforce safe headroom and prevent numerical speaker blowing or overflow.
 */
export const WEB_GPU_VOICE_PROCESSOR_SHADER = `
    struct AudioInfo {
        sampleCount : u32, // Total length of the audio frame buffer array
        gain        : f32, // Scalar multiplier representing volume boost/attenuation factor
    }

    // Bind Group Layout:
    // Binding 0: Uniform configuration data containing scalar floating-point gain properties
    // Binding 1: Read-only storage buffer containing raw, mono IEEE 754 float audio streams
    // Binding 2: Read-write storage target buffer designed to catch modified signal values
    @group(0) @binding(0) var<uniform> info : AudioInfo;
    @group(0) @binding(1) var<storage, read> rawAudioIn : array<f32>;
    @group(0) @binding(2) var<storage, read_write> processedAudioOut : array<f32>;

    @compute @workgroup_size(256)
    fn processVoiceSamples(@builtin(global_invocation_id) id : vec3<u32>) {
        let index = id.x;
        
        // Out-of-bounds safety check to isolate audio length from thread execution structures
        if (index >= info.sampleCount) { return; }

        let sample = rawAudioIn[index];
        let amplified = sample * info.gain; // Linearly scale current audio fragment
        
        // Enforce rigid clipping bounds. Restricts values strictly to [-1.0, 1.0] to prevent audio distortion artifacts.
        processedAudioOut[index] = clamp(amplified, -1.0, 1.0);
    }
`;

/**
 * WGSL Metrics Shader
 * Computes processing benchmarks and contextual operational metrics.
 * * Mechanics:
 * Designed to execute as a single thread (Workgroup size 1), acting as a localized 
 * reduction and calculation engine to avoid multi-pass overhead.
 */
export const WEB_GPU_METRICS_SHADER = `
    struct MetricsInput {
        totalDurationMs : f32, // Execution/Processing time measured in milliseconds
        sentenceCount   : f32, // Parsed sentence occurrences
        wordCount       : f32, // Parsed word occurrences 
        tokenCount      : f32, // Parsed character tokens
    }

    struct MetricsOutput {
        durationSec           : f32, // Total duration transformed to seconds
        durationPerSentenceSec: f32, // Mean processing latency window per sentence
        durationPerWordSec    : f32, // Mean processing latency window per word
        durationPerTokenSec   : f32, // Mean processing latency window per token
    }

    // Bind Group Layout:
    // Binding 0: Uniform data containing absolute counts calculated by prior parsing stages
    // Binding 1: Read-write storage block housing the computed statistical metrics output
    @group(0) @binding(0) var<uniform> metricsIn : MetricsInput;
    @group(0) @binding(1) var<storage, read_write> metricsOut : MetricsOutput;

    @compute @workgroup_size(1)
    fn computeMetrics(@builtin(global_invocation_id) id : vec3<u32>) {
        // Enforce single-lane execution boundary condition 
        if (id.x == 0u) {
            let sec = metricsIn.totalDurationMs / 1000.0; // Metric baseline normalization
            metricsOut.durationSec = sec;
            
            // Generate proportional TTS evaluation metrics 
            metricsOut.durationPerSentenceSec = sec / metricsIn.sentenceCount;
            metricsOut.durationPerWordSec    = sec / metricsIn.wordCount;
            metricsOut.durationPerTokenSec   = sec / metricsIn.tokenCount;
        }
    }
`;

/**
 * WGSL Word and Token Count Shader
 * Scans text arrays using a state-less boundary lookback pattern to record words and distinct tokens.
 */
export const WEB_GPU_WORD_AND_TOKEN_COUNT_SHADER = `
    struct ScalarInput {
        textLength : u32,
    }

    @group(0) @binding(0) var<uniform> info : ScalarInput;
    @group(0) @binding(1) var<storage, read> textBuffer : array<u32>;
    @group(0) @binding(2) var<storage, read_write> wordFlags : array<u32>;
    @group(0) @binding(3) var<storage, read_write> tokenFlags : array<u32>;

    fn isSpace(code: u32) -> bool {
        return code == 32u || code == 9u || code == 10u || code == 13u;
    }

    fn isAlpha(code: u32) -> bool {
        return (code >= 65u && code <= 90u) || (code >= 97u && code <= 122u);
    }

    fn isDigit(code: u32) -> bool {
        return code >= 48u && code <= 57u;
    }

    fn getCharType(code: u32) -> u32 {
        if (isSpace(code)) { return 1u; }
        if (isAlpha(code)) { return 2u; }
        if (isDigit(code)) { return 3u; }
        return 4u;
    }

    @compute @workgroup_size(256)
    fn countWordsAndTokens(@builtin(global_invocation_id) id : vec3<u32>) {
        let index = id.x;
        if (index >= info.textLength) { 
            return; 
        }

        let current_char = textBuffer[index];
        let current_type = getCharType(current_char);

        // ==========================================
        // WORD BOUNDARY DETECTION LOGIC
        // ==========================================
        if (current_type != 1u) {
            if (index == 0u || isSpace(textBuffer[index - 1u])) {
                wordFlags[index] = 1u;
            } else {
                wordFlags[index] = 0u;
            }
        } else {
            wordFlags[index] = 0u;
        }

        // ==========================================
        // TOKEN BOUNDARY DETECTION LOGIC
        // ==========================================
        
        // If the current character is a space, do not mark it as a token start.
        if (current_type == 1u) {
            tokenFlags[index] = 0u;
            return;
        }

        // If the text starts directly with a valid character, index 0 is the first token.
        if (index == 0u) {
            tokenFlags[index] = 1u;
            return;
        }

        let prev_char = textBuffer[index - 1u];
        let prev_type = getCharType(prev_char);
        
        let is_apostrophe = (current_char == 39u || current_char == 8217u);
        let was_apostrophe = (prev_char == 39u || prev_char == 8217u);

        // A token boundary is identified if the previous character was a space,
        // if the character type transitions, or if an apostrophe boundary is hit.
        if (prev_type == 1u || (current_type != prev_type) || is_apostrophe || was_apostrophe) {
            tokenFlags[index] = 1u;
        } else {
            tokenFlags[index] = 0u;
        }
    }
`;