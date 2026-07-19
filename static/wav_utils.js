/*
This function merges multiple independent WAV audio chunks into a single, smooth 
WAV file while removing the annoying background clicks and pops that usually happen when 
you glue independent audio clips together.
*/
export function mergeWavBuffers(buffers) {

    const typedBuffers = buffers.map(b => new Uint8Array(b));

    if (typedBuffers.length === 1) {
        return new Blob([typedBuffers[0]], { type: 'audio/wav' });
    }

    let totalDataSize = 0;
    const dataChunks = [];

    for (const buf of typedBuffers) {
        const bufView = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        let offset = 12;

        // Locate the data subchunk
        while (offset < bufView.byteLength - 8) {
            const chunkId = bufView.getUint32(offset, true);
            const chunkSize = bufView.getUint32(offset + 4, true);
            
            // Data marker found
            if (chunkId === 0x61746164) { 

                const subBuffer = buf.subarray(offset + 8, offset + 8 + chunkSize);
                dataChunks.push(subBuffer);
                totalDataSize += chunkSize;
                break;
            }
            offset += 8 + chunkSize;
        }
    }

    // Clone the first 44 bytes (header) of the first file and update the sizes
    const headerBuffer = typedBuffers[0].slice(0, 44);
    const headerView = new DataView(headerBuffer.buffer, headerBuffer.byteOffset, headerBuffer.byteLength);

    // Total file size minus 8 bytes for RIFF and file size fields
    headerView.setUint32(4, 36 + totalDataSize, true); 
    
    // Audio data size only
    headerView.setUint32(40, totalDataSize, true);     

    return new Blob([headerBuffer, ...dataChunks], { type: 'audio/wav' });
}
