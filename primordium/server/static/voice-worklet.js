// Microphone worklet: accumulate ~200ms, downsample to 16 kHz Int16,
// post to the main thread for the uplink.
class MicWorklet extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buf = [];
        this.samples = 0;
        this.chunk = Math.round(sampleRate * 0.2);
    }
    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) {
            this.buf.push(new Float32Array(ch));
            this.samples += ch.length;
            if (this.samples >= this.chunk) {
                const all = new Float32Array(this.samples);
                let o = 0;
                for (const b of this.buf) { all.set(b, o); o += b.length; }
                this.buf = []; this.samples = 0;
                const ratio = sampleRate / 16000;
                const n = Math.floor(all.length / ratio);
                const out = new Int16Array(n);
                for (let i = 0; i < n; i++) {
                    const v = all[Math.floor(i * ratio)];
                    out[i] = Math.max(-32768, Math.min(32767, v * 32767));
                }
                this.port.postMessage(out.buffer, [out.buffer]);
            }
        }
        return true;
    }
}
registerProcessor("mic-worklet", MicWorklet);
