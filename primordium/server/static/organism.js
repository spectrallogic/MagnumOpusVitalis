"use strict";
/* Primordium dashboard: a window into the organism, and its sense organs.
   Everything stays on this machine. */

const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;

let ws = null, embodied = false, state = null;
let sparkHist = [];
const DRIVE_ORDER = ["energy", "competence", "novelty", "social", "vitality"];
const TIDE_ORDER = ["arousal", "reward", "calm"];
const TIDE_COLORS = { arousal: "#ffd479", reward: "#9bffa0",
                      calm: "#7dcfff" };

/* ================= WebSocket ================= */
function connect() {
    ws = new WebSocket(`ws://127.0.0.1:${window.WS_PORT}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
        ws.send(JSON.stringify({ t: "hello", proto: 1, embodied: wantEmbody }));
        $("status").textContent = "connected";
    };
    ws.onclose = () => {
        $("status").textContent = "disconnected — retrying";
        setTimeout(connect, 2500);
    };
    ws.onmessage = ev => {
        if (typeof ev.data === "string") {
            const d = JSON.parse(ev.data);
            if (d.t === "welcome") {
                embodied = d.embodied;
                if (embodied) startSenses();
                $("status").textContent = embodied
                    ? "embodied — it can see and hear through you"
                    : "watching";
            } else if (d.t === "state") {
                state = d;
                updateHud(d);
            } else if (d.t === "pulse") {
                onPulse(d.events || []);
            }
        } else {
            const u8 = new Uint8Array(ev.data);
            const tag = u8[0];
            const blob = new Blob([u8.slice(1)],
                { type: tag === 0x82 ? "image/jpeg" : "image/png" });
            const url = URL.createObjectURL(blob);
            if (tag === 0x81) drawTo("imagine", url);
            else if (tag === 0x82) drawTo("retina", url);
            else if (tag === 0x83) pushDream(url);
            else if (tag === 0x84) drawTo("mind", url);
            else if (tag === 0x85) drawTo("easel", url);
        }
    };
}
function drawTo(id, url) {
    const img = new Image();
    img.onload = () => {
        const c = $(id).getContext("2d");
        c.imageSmoothingEnabled = false;
        c.drawImage(img, 0, 0, $(id).width, $(id).height);
        URL.revokeObjectURL(url);
    };
    img.src = url;
}
function pushDream(url) {
    const img = new Image();
    img.src = url;
    const box = $("dreams");
    box.appendChild(img);
    while (box.children.length > 4) box.removeChild(box.firstChild);
}

/* ================= senses uplink ================= */
let wantEmbody = false;
$("grant").addEventListener("click", () => {
    wantEmbody = true;
    $("perm-overlay").style.display = "none";
    if (ws && ws.readyState === 1) {
        ws.close();     // reconnect with embodied hello
    } else connect();
});
$("watch-only").addEventListener("click", () => {
    wantEmbody = false;
    $("perm-overlay").style.display = "none";
    if (!ws) connect();
});

async function startSenses() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 320, height: 240 },
            audio: { echoCancellation: false, noiseSuppression: false,
                     autoGainControl: false },
        });
        startCamera(stream);
        startMic(stream);
        startVoice();
    } catch (e) {
        $("status").textContent = "permission denied — watching only";
    }
}

function startCamera(stream) {
    const video = document.createElement("video");
    video.srcObject = stream;
    video.play();
    const cv = document.createElement("canvas");
    cv.width = 96; cv.height = 96;
    const cx = cv.getContext("2d");
    setInterval(() => {
        if (!ws || ws.readyState !== 1 || video.readyState < 2) return;
        const s = Math.min(video.videoWidth, video.videoHeight);
        cx.drawImage(video,
            (video.videoWidth - s) / 2, (video.videoHeight - s) / 2, s, s,
            0, 0, 96, 96);
        cv.toBlob(async b => {
            if (!b || !ws || ws.readyState !== 1) return;
            const buf = await b.arrayBuffer();
            const out = new Uint8Array(5 + buf.byteLength);
            out[0] = 0x01;
            new DataView(out.buffer).setUint32(1, Date.now() % 0xffffffff);
            out.set(new Uint8Array(buf), 5);
            ws.send(out.buffer);
        }, "image/jpeg", 0.7);
    }, 1000 / 6);
}

let audioCtx = null;
async function startMic(stream) {
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("/static/voice-worklet.js");
    const src = audioCtx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(audioCtx, "mic-worklet");
    src.connect(node);
    node.port.onmessage = ev => {
        if (!ws || ws.readyState !== 1) return;
        const pcm = new Uint8Array(ev.data);
        const out = new Uint8Array(5 + pcm.byteLength);
        out[0] = 0x02;
        new DataView(out.buffer).setUint32(1, Date.now() % 0xffffffff);
        out.set(pcm, 5);
        ws.send(out.buffer);
    };
}

/* ================= its voice (WebAudio synth) ================= */
let synth = null;
function startVoice() {
    if (!audioCtx) audioCtx = new AudioContext();
    const ac = audioCtx;
    const saw = ac.createOscillator(); saw.type = "sawtooth";
    saw.frequency.value = 140;
    const noiseBuf = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
    const nd = noiseBuf.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    const noise = ac.createBufferSource();
    noise.buffer = noiseBuf; noise.loop = true;
    const sawGain = ac.createGain(); sawGain.gain.value = 0.7;
    const noiseGain = ac.createGain(); noiseGain.gain.value = 0.2;
    saw.connect(sawGain); noise.connect(noiseGain);

    const centers = [200, 320, 510, 810, 1290, 2050, 3260, 4000];
    const bands = centers.map(f => {
        const bq = ac.createBiquadFilter();
        bq.type = "bandpass"; bq.frequency.value = f; bq.Q.value = 4;
        const g = ac.createGain(); g.gain.value = 0.0;
        sawGain.connect(bq); noiseGain.connect(bq); bq.connect(g);
        return { bq, g };
    });
    const master = ac.createGain(); master.gain.value = 0.0;
    const analyser = ac.createAnalyser(); analyser.fftSize = 256;
    bands.forEach(b => b.g.connect(master));
    master.connect(analyser); analyser.connect(ac.destination);
    saw.start(); noise.start();
    synth = { saw, bands, master, analyser };
    drawSpectrogram();
}
function updateVoice(v) {
    if (!synth || !v) return;
    const t = audioCtx.currentTime;
    synth.saw.frequency.setTargetAtTime(v.f0 || 140, t, 0.03);
    (v.gains || []).forEach((g, i) => {
        if (synth.bands[i])
            synth.bands[i].g.gain.setTargetAtTime(g * 0.5, t, 0.03);
    });
    const amp = muted ? 0 : (v.amp || 0) * 0.25;
    synth.master.gain.setTargetAtTime(amp, t, 0.03);
}
function drawSpectrogram() {
    const cv = $("spectro"), cx = cv.getContext("2d");
    const data = new Uint8Array(synth.analyser.frequencyBinCount);
    (function loop() {
        requestAnimationFrame(loop);
        synth.analyser.getByteFrequencyData(data);
        cx.drawImage(cv, -2, 0);
        cx.clearRect(cv.width - 2, 0, 2, cv.height);
        for (let y = 0; y < cv.height; y++) {
            const i = Math.floor((1 - y / cv.height) * data.length * 0.6);
            const v = data[i] / 255;
            cx.fillStyle = `rgba(155,232,216,${v})`;
            cx.fillRect(cv.width - 2, y, 2, 1);
        }
    })();
}

/* ================= the mind's ear: sonify its predicted mel =========== */
let mindSynth = null, mindsEyeOn = false;
function startMindAudio() {
    if (mindSynth) return;
    if (!audioCtx) audioCtx = new AudioContext();
    const ac = audioCtx;
    // 16 sines at mel-ish centers: the sound of its expectation of sound
    const centers = Array.from({ length: 16 }, (_, i) =>
        60 * Math.pow(6000 / 60, i / 15));
    const master = ac.createGain(); master.gain.value = 0.0;
    const oscs = centers.map(f => {
        const o = ac.createOscillator(); o.type = "sine"; o.frequency.value = f;
        const g = ac.createGain(); g.gain.value = 0.0;
        o.connect(g); g.connect(master); o.start();
        return g;
    });
    master.connect(ac.destination);
    mindSynth = { master, oscs };
}
function updateMindAudio(mel) {
    if (!mindSynth) return;
    const t = audioCtx.currentTime;
    if (!mindsEyeOn || !mel) {
        mindSynth.master.gain.setTargetAtTime(0, t, 0.1);
        return;
    }
    mindSynth.master.gain.setTargetAtTime(0.10, t, 0.1);
    // 32 mel bands -> 16 oscillator gains
    for (let i = 0; i < 16; i++) {
        const v = ((mel[i * 2] || 0) + (mel[i * 2 + 1] || 0)) / 2;
        mindSynth.oscs[i].gain.setTargetAtTime(
            clamp(v / 3, 0, 1) * 0.5, t, 0.08);
    }
}
$("minds-eye").addEventListener("click", () => {
    mindsEyeOn = !mindsEyeOn;
    startMindAudio();
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
    $("minds-eye").classList.toggle("on", mindsEyeOn);
    $("minds-eye").classList.toggle("off", !mindsEyeOn);
    $("mind").classList.toggle("closed", !mindsEyeOn);
    $("mind-state").textContent = mindsEyeOn ? "open" : "closed";
    if (ws && ws.readyState === 1)
        ws.send(JSON.stringify({ t: "minds_eye", v: mindsEyeOn }));
    if (!mindsEyeOn) updateMindAudio(null);
});

/* ================= chat (into its Wordstream) ================= */
function sendChat() {
    const box = $("chat-input");
    const text = box.value.trim();
    if (!text || !ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ t: "chat", text }));
    box.value = "";
}
$("chat-send").addEventListener("click", sendChat);
$("chat-input").addEventListener("keydown", e => {
    if (e.key === "Enter") sendChat();
    e.stopPropagation();
});

/* ================= controls ================= */
let muted = false;
$("mute").addEventListener("click", () => {
    muted = !muted;
    $("mute").textContent = muted ? "VOICE OFF" : "VOICE ON";
    $("mute").classList.toggle("off", muted);
    if (ws && ws.readyState === 1)
        ws.send(JSON.stringify({ t: "mute", v: muted }));
});
const imprintBtn = $("imprint");
function setImprint(on) {
    imprintBtn.classList.toggle("holding", on);
    if (ws && ws.readyState === 1)
        ws.send(JSON.stringify({ t: "imprint", on }));
}
imprintBtn.addEventListener("pointerdown", () => setImprint(true));
window.addEventListener("pointerup", () => setImprint(false));

/* ================= HUD ================= */
let seenEvents = new Set();
function updateHud(s) {
    // the server owns the mind's-eye state — adopt it if we drifted
    // (e.g. after a reconnect the local toggle can be stale)
    if (s.minds_eye != null && s.minds_eye !== mindsEyeOn) {
        mindsEyeOn = s.minds_eye;
        $("minds-eye").classList.toggle("on", mindsEyeOn);
        $("minds-eye").classList.toggle("off", !mindsEyeOn);
        $("mind").classList.toggle("closed", !mindsEyeOn);
        $("mind-state").textContent = mindsEyeOn ? "open" : "closed";
        if (!mindsEyeOn) updateMindAudio(null);
    }
    $("h-stage").textContent = s.stage;
    $("h-acuity").textContent = s.acuity || "";
    $("h-tick").textContent = s.tick;
    $("h-hz").textContent = (s.hz || 0).toFixed(1);
    $("h-felt").textContent = ((s.self || {}).felt_time || 0).toFixed(0);
    $("h-dil").textContent = ((s.self || {}).dilation || 1).toFixed(2);
    $("h-sleep").textContent = s.sleep.asleep ? "ASLEEP" : "awake";
    $("h-spress").textContent = (s.sleep.pressure || 0).toFixed(2);
    $("h-kin").textContent = (s.kinship || 0).toFixed(2);
    $("h-anchors").textContent = s.imprints || 0;
    $("h-std").textContent = (s.latent_std || 0).toFixed(3);
    $("h-loss").textContent = s.loss == null ? "—" : s.loss.toFixed(4);

    // the watch and the wheel — spikes against each stream's own
    // history, and prediction error at the scale of moments
    const wt = s.watch || {};
    if (wt.spikes != null) {
        $("h-watch").textContent = wt.spikes;
        const zs = Object.entries(wt.z || {})
            .map(([k, v]) => `${k} ${v}`).join(" · ");
        $("h-watch").parentElement.title = zs || "no streams yet";
    }
    const wl = s.wheel || {};
    $("h-wheel").textContent = (wl.loss != null)
        ? `${wl.turns}t · ${wl.loss.toFixed(4)}` : "warming";

    // the grip — whether its own hand measurably matters to its
    // canvas predictions (counterfactual ratio; 1.0 = ignored)
    const gr = s.grip || {};
    $("h-grip").textContent = gr.probes
        ? `ratio ${gr.ratio.toFixed(3)} · ${gr.probes} probes` : "unproven";

    // bus provenance — every substrate write, signed by its source
    const bp = s.bus_provenance || {};
    const srcs = Object.keys(bp);
    if (srcs.length) {
        const total = srcs.reduce((a, k) => a + (bp[k].writes || 0), 0);
        const top = srcs.slice().sort(
            (a, b) => (bp[b].writes || 0) - (bp[a].writes || 0))[0];
        $("h-prov").textContent = `${total} writes · top ${top}`;
        $("h-prov").parentElement.title = srcs.map(k =>
            `${k}: ${bp[k].writes} writes · norm ${(bp[k].mean_norm ?? 0).toFixed(3)}`
        ).join("\n");
    }

    // the gaze — the yellow box IS the crop it chose this instant
    const gz = s.gaze || {};
    if (gz.zoom != null) {
        const box = $("gaze-box");
        const side = gz.zoom * 100;
        const cx = (gz.x + 1) / 2 * 100, cy = (gz.y + 1) / 2 * 100;
        const l = clamp(cx - side / 2, 0, 100 - side);
        const t = clamp(cy - side / 2, 0, 100 - side);
        box.style.left = l + "%";
        box.style.top = t + "%";
        box.style.width = side + "%";
        box.style.height = side + "%";
        box.title = `gaze ${gz.x},${gz.y} zoom ${gz.zoom} · ` +
            `${gz.saccades} saccades · ${gz.releases} boredom releases`;
    }

    // the reach — how much of its life it can lean on, and what
    // that memory is measurably worth
    const rc = s.reach || {};
    if (rc.size != null) {
        $("h-reach").textContent = rc.size;
        $("h-reach-gain").textContent =
            rc.probes ? rc.gain.toFixed(4) : "unproven";
    }

    // the growing core — real anatomy, real capacity gates on hover
    const an = s.anatomy || {};
    if (an.params) {
        $("h-core").textContent =
            `${an.blocks}blk · ${(an.params / 1e6).toFixed(1)}M`;
        $("h-blooms").textContent = an.blooms || 0;
        const cg = s.cap_gates || {};
        $("h-core").parentElement.title = "capacity gates: " +
            Object.entries(cg).map(([k, v]) => `${k} ${v}`).join(" · ");
    }

    const gates = ((s.dev || {}).gates) || {};
    $("h-gates").innerHTML = Object.entries(gates).map(([k, v]) =>
        `<span class="gate">${k} <b>${Math.round(v * 100)}%</b></span>`).join("");

    const dr = (s.drives || {});
    const lv = dr.levels || {}, sp = dr.setpoints || {};
    $("h-drives").innerHTML = DRIVE_ORDER.map(d => {
        const l = lv[d] ?? 0, st = sp[d] ?? 0.5;
        return `<div class="gauge"><span class="name">${d}</span>
            <span class="bar"><span class="fill" style="width:${l * 100}%"></span>
            <span class="set" style="left:${st * 100}%"></span></span>
            <span class="val">${l.toFixed(2)}</span></div>`;
    }).join("");

    // the tide — every bar's tooltip lists the REAL events that raised it
    const tide = (s.tide || {}).levels || {};
    const tideCauses = (s.tide || {}).causes || {};
    $("h-chem").innerHTML = TIDE_ORDER.map(c => {
        const v = tide[c] ?? 0;
        const why = (tideCauses[c] || []).slice(-4)
            .map(e => `${e.delta > 0 ? "+" : ""}${e.delta} ${e.cause}`)
            .join("\n") || "no causes yet";
        return `<div class="gauge" title="${escapeHtml(why)}">
            <span class="name">${c}</span>
            <span class="bar"><span class="fill"
            style="width:${clamp(v / 2, 0, 1) * 100}%;background:${TIDE_COLORS[c]}"></span></span>
            <span class="val">${v.toFixed(2)}</span></div>`;
    }).join("");

    const acts = s.affect_acts || {};
    const installed = s.affects || [];
    if (installed.length) {
        $("h-affects").innerHTML = installed.map(a => {
            const z = clamp((acts[a.id] || 0) / 4, -1, 1);
            const left = z < 0 ? 50 + z * 50 : 50;
            const width = Math.abs(z) * 50;
            const sig = a.sig && a.sig.drives ? Object.entries(a.sig.drives)
                .filter(([, v]) => Math.abs(v) > 0.15)
                .map(([k, v]) => `${v > 0 ? "+" : "−"}${k}`).join(" ") : "";
            const val = a.sig ? ` val ${a.sig.valence}` : "";
            const prov = a.provenance === "innate" ? " innate" : "";
            return `<div class="affect${prov}" title="${sig}${val}">
                <span class="label">${a.id}</span>
                <span class="bar"><span class="zero"></span>
                <span class="fill" style="left:${left}%;width:${width}%"></span></span>
                </div>`;
        }).join("");
    }

    sparkHist.push(s.surprise || 1);
    if (sparkHist.length > 110) sparkHist.shift();
    const cv = $("spark"), cx2 = cv.getContext("2d");
    cx2.clearRect(0, 0, cv.width, cv.height);
    cx2.strokeStyle = "#9be8d8"; cx2.beginPath();
    const mx = Math.max(2, ...sparkHist);
    sparkHist.forEach((v, i) => {
        const x = (i / 109) * cv.width;
        const y = cv.height - (v / mx) * (cv.height - 4) - 2;
        i ? cx2.lineTo(x, y) : cx2.moveTo(x, y);
    });
    cx2.stroke();

    // wordstream: the room's transcript + its own keyboard, verbatim
    const wst = s.wordstream || {};
    const tr = wst.transcript || [];
    const trBox = $("transcript");
    const atBottom = trBox.scrollTop + trBox.clientHeight >= trBox.scrollHeight - 4;
    trBox.innerHTML = tr.length
        ? tr.map(m => `<div class="msg ${m.source}"><span class="src">${m.source}</span>${escapeHtml(m.text)}</div>`).join("")
        : `<span class="dim-note">no one has spoken to it yet</span>`;
    if (atBottom) trBox.scrollTop = trBox.scrollHeight;
    $("typed").textContent = wst.typed_recent || "";
    $("h-strokes").textContent = (s.easel || {}).strokes || 0;

    // fringe — the soft edge, one diamond per sprout, colored by its
    // MEASURED utility (counterfactual ablation on replay, not vibes)
    const sprouts = ((s.fringe || {}).sprouts) || [];
    if (sprouts.length) {
        $("h-fringe-merges").textContent =
            sprouts.reduce((a, sp) => a + (sp.merges || 0), 0);
        $("h-fringe").innerHTML = sprouts.map(sp => {
            const cls = sp.probes === 0 ? "unproven"
                : sp.util > 0 ? "up" : sp.util < 0 ? "down" : "unproven";
            return `<span class="sprout ${cls}" title="${sp.site} · util ${sp.util} · probes ${sp.probes} · merges ${sp.merges} · recycles ${sp.recycles}">◆</span>`;
        }).join("");
    }

    // dormant organs — rendered only from real Gatehouse state
    const organs = Object.entries(s.gates || {});
    if (organs.length) {
        $("h-organs").innerHTML = organs.map(([name, g]) => {
            const st = (g.state || "locked").toLowerCase();
            const prog = g.progress != null
                ? ` ${Math.round(g.progress * 100)}%` : "";
            const icon = st === "unlocked" ? "◉" : st === "eligible" ? "◎" : "⊘";
            const cls = st === "unlocked" ? "open"
                : st === "eligible" ? "eligible" : "";
            return `<span class="organ ${cls}" title="${escapeHtml(g.why || "")}">${icon} ${name}${prog}</span>`;
        }).join("");
    }

    updateVoice(s.voice);
    updateMindAudio(s.mind_mel);

    for (const ev of (s.events || [])) {
        const key = `${ev.kind}:${ev.at}`;
        if (seenEvents.has(key)) continue;
        seenEvents.add(key);
        toast(ev.kind === "stage_advance"
            ? `IT GREW — STAGE ${ev.stage}` : ev.kind.toUpperCase());
    }
}
function escapeHtml(t) {
    return String(t).replace(/[&<>"]/g,
        c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function toast(msg) {
    const d = document.createElement("div");
    d.className = "toast";
    d.textContent = msg;
    document.body.appendChild(d);
    setTimeout(() => d.remove(), 4200);
}

/* ================= PULSE — honest thought traffic =================
   One block = one event drained from the organism's ring, which is
   emitted only at real computation sites. Tooltip shows verbatim meta.
   Bursts of one kind (>8 per 200ms batch) merge into a single xN block
   that still carries the count — merged, never invented. */
const ZONES = ["WORLD", "SENSE", "CORE", "SELF", "EXPRESS", "KEEP", "SCAFFOLD"];
const KIND_COLORS = {
    tick: "#3f8f80", replay: "#7dcfff", dream: "#c9a3ff", decode: "#c9a3ff",
    distill: "#ffd479", consolidate: "#7dcfff", checkpoint: "#9bffa0",
    stage_advance: "#ffffff", babble_out: "#9be8d8", paint: "#ff9de2",
    chat_in: "#9bffa0", caregiver_msg: "#ffd479", caregiver_absent: "#ff7a97",
    affect_spike: "#ff7a97", gate_state: "#ffd479", voice_gate: "#9be8d8",
    caregiver_present: "#9bffa0", needles: "#ffd479",
    instincts_retired: "#ffd479", lodestar_released: "#ffd479",
    sprout_merge: "#9bffa0", sprout_recycle: "#556677",
    bloom: "#c9ffa0", recall: "#7dcfff", gaze_shift: "#ffd479",
    watch: "#ff9de2", slow_surprise: "#c9a3ff", grip: "#9be8d8",
};
const pulseCv = $("pulse"), pxc = pulseCv.getContext("2d");
let pulseBlocks = [];
function pulseResize() {
    pulseCv.width = pulseCv.clientWidth;
    pulseCv.height = pulseCv.clientHeight;
}
window.addEventListener("resize", pulseResize);
pulseResize();

function onPulse(events) {
    const byKind = {};
    for (const e of events) (byKind[e.kind] = byKind[e.kind] || []).push(e);
    for (const evs of Object.values(byKind)) {
        const list = evs.length > 8
            ? [Object.assign({}, evs[0], { n: evs.length })] : evs;
        for (const e of list) spawnBlock(e);
    }
}
function spawnBlock(e) {
    pulseBlocks.push({
        e,
        fi: Math.max(0, ZONES.indexOf(e.zone_from)),
        ti: Math.max(0, ZONES.indexOf(e.zone_to)),
        born: performance.now(),
        // DISCLOSED COSMETIC: lane and duration jitter are placement
        // only, so simultaneous blocks don't overlap. The event, its
        // kind, zones, and count are real; the scatter is not data.
        dur: 900 + Math.random() * 400,
        lane: 0.14 + Math.random() * 0.74,
        size: e.n ? Math.min(16, 8 + Math.log2(e.n) * 2)
            : (e.kind === "tick" ? 4 : 7),
        x: 0, y: 0, alive: true,
    });
    if (pulseBlocks.length > 220)
        pulseBlocks.splice(0, pulseBlocks.length - 220);
}
const easeIO = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
function drawPulse(now) {
    requestAnimationFrame(drawPulse);
    const w = pulseCv.width, h = pulseCv.height;
    if (!w || !h) { pulseResize(); return; }
    pxc.clearRect(0, 0, w, h);
    // zone columns
    pxc.font = "9px Consolas, monospace";
    pxc.textAlign = "center";
    ZONES.forEach((z, i) => {
        const x = (i + 0.5) / ZONES.length * w;
        pxc.fillStyle = "rgba(155,232,216,0.45)";
        pxc.fillText(z, x, 14);
        pxc.strokeStyle = "rgba(155,232,216,0.07)";
        pxc.beginPath(); pxc.moveTo(x, 22); pxc.lineTo(x, h - 8); pxc.stroke();
    });
    pulseBlocks = pulseBlocks.filter(b => {
        const t = (now - b.born) / b.dur;
        if (t > 1.8) return false;
        const fx = (b.fi + 0.5) / ZONES.length * w;
        const tx = (b.ti + 0.5) / ZONES.length * w;
        b.x = lerp(fx, tx, easeIO(clamp(t, 0, 1)));
        b.y = 24 + b.lane * (h - 36);
        const fade = t > 1 ? clamp(1 - (t - 1) / 0.8, 0, 1) : 1;
        const col = KIND_COLORS[b.e.kind] || "#9be8d8";
        pxc.globalAlpha = fade * (b.e.kind === "tick" ? 0.45 : 0.9);
        pxc.fillStyle = col;
        pxc.fillRect(b.x - b.size / 2, b.y - b.size / 2, b.size, b.size);
        if (b.e.n) {
            pxc.fillStyle = "#020407";
            pxc.fillText("x" + b.e.n, b.x, b.y + 3);
        }
        pxc.globalAlpha = 1;
        return true;
    });
}
requestAnimationFrame(drawPulse);

const tip = $("pulse-tip");
pulseCv.addEventListener("mousemove", ev => {
    const r = pulseCv.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 144;
    for (const b of pulseBlocks) {
        const d = (b.x - mx) ** 2 + (b.y - my) ** 2;
        if (d < bd) { bd = d; best = b; }
    }
    if (!best) { tip.style.display = "none"; return; }
    const e = best.e;
    tip.textContent = `${e.kind}${e.n ? " x" + e.n : ""}  ${e.zone_from}→${e.zone_to}\n`
        + JSON.stringify(e.meta || {});
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 14) + "px";
    tip.style.top = (ev.clientY + 10) + "px";
});
pulseCv.addEventListener("mouseleave", () => tip.style.display = "none");

/* ============ the plasma organism (decorative — a toggle now) ============ */
const plasma = $("plasma"), pc = plasma.getContext("2d");
let plasmaOn = false;
$("plasma-btn").addEventListener("click", () => {
    plasmaOn = !plasmaOn;
    plasma.classList.toggle("hidden", !plasmaOn);
    $("plasma-btn").classList.toggle("on", plasmaOn);
    $("plasma-btn").classList.toggle("off", !plasmaOn);
});
let W = 0, H = 0;
function resize() { W = plasma.width = innerWidth; H = plasma.height = innerHeight; }
window.addEventListener("resize", resize); resize();
const parts = Array.from({ length: 2200 }, () => ({
    a: Math.random() * Math.PI * 2,
    r: Math.pow(Math.random(), 0.6),
    sp: 0.2 + Math.random() * 0.8,
    ph: Math.random() * Math.PI * 2,
}));
let hue = 170, breath = 0.5, jitter = 0, coherence = 1, dimmer = 1;
function drawPlasma(now) {
    requestAnimationFrame(drawPlasma);
    if (!plasmaOn) return;
    const t = now / 1000;
    const s = state || {};
    const busNorm = ((s.bus || {}).state_norm || 0.5);
    const energy = (((s.drives || {}).levels) || {}).energy ?? 0.6;
    const ne = (((s.tide || {}).levels || {}).arousal || 0.2);
    const valence = (((s.drives || {})).reward_ema || 0);
    const cont = ((s.self || {}).continuity ?? 1);
    const asleep = ((s.sleep || {}).asleep) || false;

    hue = lerp(hue, valence >= 0 ? 165 - valence * 900 : 165 + Math.min(60, -valence * 900), 0.02);
    breath = lerp(breath, 0.35 + 0.12 * Math.min(busNorm / 4, 1) + 0.1 * energy, 0.03);
    jitter = lerp(jitter, ne * 6, 0.05);
    coherence = lerp(coherence, clamp((cont + 1) / 2, 0.2, 1), 0.03);
    dimmer = lerp(dimmer, asleep ? 0.25 : 1.0, 0.02);

    pc.fillStyle = "rgba(2,4,7,0.35)";
    pc.fillRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2;
    const R = Math.min(W, H) * breath * (1 + 0.03 * Math.sin(t * (asleep ? 0.6 : 1.6)));
    const kin = (s.kinship || 0);
    for (const p of parts) {
        p.a += p.sp * 0.004 * (0.4 + coherence) + (Math.random() - 0.5) * 0.002 * jitter;
        const wob = Math.sin(t * p.sp + p.ph) * (1 - coherence) * 0.25;
        const rr = R * (p.r + wob * 0.2);
        const x = cx + Math.cos(p.a) * rr + (Math.random() - 0.5) * jitter;
        const y = cy + Math.sin(p.a) * rr * 0.82 + (Math.random() - 0.5) * jitter;
        const depth = 0.4 + 0.6 * (1 - p.r);
        const h = hue + kin * 40 * (1 - p.r);   // kinship warms the core
        pc.fillStyle = `hsla(${h},70%,${45 + depth * 25}%,${(0.05 + depth * 0.3) * dimmer})`;
        pc.fillRect(x, y, 1.6 + depth, 1.6 + depth);
    }
}
requestAnimationFrame(drawPlasma);

/* boot: overlay first; connect happens on choice */
