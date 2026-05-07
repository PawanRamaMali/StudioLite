// AudioWorkletProcessor that batches mono float32 samples and posts them as
// int16 little-endian PCM. The page is expected to construct the AudioContext
// at sampleRate: 16000 so we don't need to resample here.
class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._batch = [];
    this._batchSamples = 0;
    // ~100 ms at 16 kHz -> 1600 samples per outbound chunk
    this._targetSamples = 1600;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Downmix to mono by averaging channels
    const channelCount = input.length;
    const frameCount = input[0].length;
    const mono = new Float32Array(frameCount);
    for (let ch = 0; ch < channelCount; ch++) {
      const data = input[ch];
      for (let i = 0; i < frameCount; i++) mono[i] += data[i];
    }
    if (channelCount > 1) {
      for (let i = 0; i < frameCount; i++) mono[i] /= channelCount;
    }

    this._batch.push(mono);
    this._batchSamples += frameCount;

    if (this._batchSamples >= this._targetSamples) {
      const merged = new Float32Array(this._batchSamples);
      let off = 0;
      for (const arr of this._batch) {
        merged.set(arr, off);
        off += arr.length;
      }
      this._batch = [];
      this._batchSamples = 0;

      const pcm = new Int16Array(merged.length);
      for (let i = 0; i < merged.length; i++) {
        let s = Math.max(-1, Math.min(1, merged[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-processor", PcmProcessor);
