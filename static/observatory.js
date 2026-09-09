/* Vitalis Observatory. Geometry is illustrative; every value and event comes
   from the engine. Model text enters the DOM only through textContent. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const num = (value, fallback = 0) => typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  const fmt = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  const clamp = (v, low = 0, high = 1) => Math.max(low, Math.min(high, num(v)));
  const text = (id, value) => { $(id).textContent = value == null ? '—' : String(value); };
  const node = (tag, className, value) => { const e = document.createElement(tag); if (className) e.className = className; if (value != null) e.textContent = value; return e; };
  const short = (value, n = 105) => String(value ?? '').length > n ? String(value).slice(0, n - 1) + '…' : String(value ?? '');
  const title = value => String(value ?? '').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  const colors = {calm:'#92d9be', curious:'#b8a1db', joy:'#dfbd84', trust:'#78beb8', stress:'#df9891', reward:'#dfbd84', arousal:'#b8a1db'};
  let current = null, pending = null, session = null, lastReceived = 0, lastTickAt = 0, lastTick = null;
  let paused = false, busy = false, view = 'map', selection = {type:'region', key:'bus'};
  let events = [], eventIds = new Set(), filter = 'all', memories = [], lastMemoryFetch = 0;
  let treeKey = '', source = null, connection = 'connecting', lastFetchError = '', phase = 0;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

  function setConnection(mode, detail) {
    connection = mode;
    document.body.dataset.connection = mode;
    const names = {connecting:'Connecting', live:'Engine live', stalled:'Engine stalled', offline:'Disconnected', error:'Engine error'};
    text('connection-label', names[mode] || mode);
    const message = detail || (mode === 'offline' ? 'Connection lost. The display holds the last received state and will reconnect automatically.' : mode === 'stalled' ? 'The connection is open, but the engine clock has stopped advancing. Visual activity is frozen.' : '');
    $('connection-notice').hidden = !message;
    text('connection-notice', message);
    $('send-message').disabled = busy || mode !== 'live';
    if (mode !== 'live') document.querySelectorAll('[data-wire]').forEach(e => e.dataset.active = 'false');
  }

  async function request(url, options = {}) {
    const response = await fetch(url, options);
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response.'); }
    if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function resetSession(id) {
    session = id; events = []; eventIds.clear(); memories = []; treeKey = ''; lastTick = null;
    current = null; pending = null; selection = {type:'region', key:'bus'};
    $('conversation-list').replaceChildren();
    restoreConversation();
  }

  function accept(snapshot) {
    if (!snapshot || snapshot.error || !snapshot.bus) {
      setConnection('error', snapshot?.error || 'Engine telemetry is unavailable.');
      return;
    }
    const id = snapshot.runtime?.session_id;
    if (id && id !== session) resetSession(id);
    lastReceived = performance.now();
    const tick = snapshot.bus.tick_count;
    if (tick !== lastTick) { lastTickAt = lastReceived; lastTick = tick; }
    setConnection(lastReceived - lastTickAt > 5000 ? 'stalled' : 'live');
    for (const e of snapshot.journal || []) {
      if (!e || eventIds.has(e.id)) continue;
      eventIds.add(e.id); events.push(e);
      if (e.kind === 'autonomous_reply') appendMessage('assistant', e.payload?.reply || '', null, true);
    }
    if (events.length > 240) {
      const dropped = events.splice(0, events.length - 240);
      dropped.forEach(e => eventIds.delete(e.id));
    }
    pending = snapshot;
    if (!paused) render(snapshot);
  }

  function meters(id, entries, max = 1) {
    const root = $(id);
    root.replaceChildren();
    if (!entries.length) { root.append(node('p','empty small','No measurements yet.')); return; }
    for (const [name, value] of entries) {
      const row = node('div','meter-row');
      const label = node('div','meter-label');
      label.append(node('span','',title(name)), node('span','',fmt(value)));
      const track = node('div','meter-track'), fill = node('span');
      fill.style.width = `${clamp(Math.abs(num(value)) / max) * 100}%`;
      fill.style.background = colors[name] || 'var(--teal)';
      track.append(fill); row.append(label, track); root.append(row);
    }
  }

  function dominant(s) {
    const entries = Object.entries(s.limbic?.blended || {}).filter(([,v]) => Number.isFinite(v));
    entries.sort((a,b) => b[1] - a[1]);
    return {entries, name: entries.length && Math.abs(entries[0][1]) > 1e-6 ? entries[0][0] : null, value: entries[0]?.[1]};
  }

  function behavior(s) {
    if (busy) return ['Responding', 'Your message is being processed by the model.'];
    const latestReply = [...events].reverse().find(e => ['reply_emitted','autonomous_reply'].includes(e.kind));
    const now = num(s.runtime?.server_time, Date.now()/1000);
    if (latestReply && now - latestReply.ts < 4) return ['Expressing', 'A response was just emitted into the conversation.'];
    if (num(s.executive?.pressure) >= num(s.executive?.effective_threshold, Infinity)) return ['Ready to speak', 'Executive pressure has reached the current speech threshold.'];
    if (num(s.speculative?.search?.depth_reached) > 1) return ['Considering futures', `The latest search reached ${s.speculative.search.depth_reached} levels of imagined consequences.`];
    if (s.subconscious?.surprise) return ['Exploring', 'The surprise channel promoted an unexpected candidate.'];
    if (s.recall) return ['Recalling', 'A previous experience has been retrieved into the current context.'];
    return ['Background activity', 'The latent state and subconscious continue evolving between interactions.'];
  }

  function render(s) {
    current = s;
    const bus = s.bus || {}, sub = s.subconscious || {}, spec = s.speculative || {}, ex = s.executive || {};
    const affect = dominant(s), [bName, bReason] = behavior(s);
    text('model-name', s.runtime?.model || 'Connected model');
    const seconds = Math.floor(num(bus.wall_seconds));
    text('session-clock', `${Math.floor(seconds/60)}m ${String(seconds%60).padStart(2,'0')}s · ${num(bus.tick_count).toLocaleString()} ticks`);
    text('dominant-affect', affect.name ? title(affect.name) : 'Neutral');
    $('affect-arc').style.strokeDasharray = `${clamp(Math.abs(num(affect.value))) * 100} 100`;
    meters('affect-bars', affect.entries.slice(0, 5));
    meters('modulation-bars', ['arousal','reward','calm','stress'].map(k => [k, s.neuromod?.[k]]), 1.5);
    text('behavior-name', bName); text('behavior-reason', bReason);
    const pressure = num(ex.pressure), threshold = num(ex.effective_threshold, 1), ceiling = Math.max(1, pressure, threshold) * 1.2;
    $('pressure-fill').style.width = `${clamp(pressure/ceiling)*100}%`;
    $('pressure-threshold').style.left = `${clamp(threshold/ceiling)*100}%`;
    text('pressure-value', `${fmt(ex.pressure)} / ${fmt(ex.effective_threshold)}`);
    text('continuity-value', fmt(s.self_model?.continuity)); text('memory-value', s.memory?.size);
    const period = s.flow_metrics?.flow?.last_wall_dt;
    text('cadence-value', num(period) > 0 ? `${fmt(1/period,1)} Hz` : '—');
    text('region-affect-value', affect.name ? title(affect.name) : 'Neutral baseline');
    text('region-memory-value', `${s.memory?.size ?? 0} stored traces`);
    text('region-sea-value', `${sub.l1_count ?? 0} resonant candidates`);
    text('region-peaks-value', `${sub.l1_count ?? 0} → ${sub.l2_count ?? 0} survivors`);
    text('region-expression-value', busy ? 'Forming a response' : `${fmt(ex.pressure)} speech pressure`);
    text('region-imagination-value', `${spec.search?.nodes ?? 0} nodes · depth ${spec.search?.depth_reached ?? 0}`);
    text('bus-magnitude', fmt(bus.state_norm));
    text('map-status', `TICK ${num(bus.tick_count).toLocaleString()}`);
    text('branch-count', spec.search?.nodes ?? 0);
    text('surfacing-thought', sub.intrusive_meta?.phrase || sub.intrusive_word || (sub.intrusive_source ? `A ${sub.intrusive_source.replaceAll('_',' ')} candidate is influencing the state.` : 'Waiting for a candidate to reach the upper layer.'));
    text('surfacing-source', sub.intrusive_meta?.epistemic || sub.intrusive_source || '—');
    const sources = {affect:['limbic'], memory:['self_model','consolidation'], sea:['subconscious_stack'], peaks:['subconscious_stack'], imagination:['speculative_futures','speculative_penumbra'], expression:['thought_feedback']};
    document.querySelectorAll('[data-wire]').forEach(e => {
      e.dataset.active = String(connection === 'live' && sources[e.dataset.wire].some(k => {
        const p = s.bus_provenance?.[k];
        return p && num(p.mean_norm) > .00001 && num(bus.tick_count) - num(p.last_clock) < 30;
      }));
    });
    if (view === 'futures') renderTree();
    if (view === 'memory' && performance.now() - lastMemoryFetch > 5000) loadMemory();
    renderInspector(); renderEvents(); drawField();
  }

  const regions = {
    bus: {heading:'Inside the latent bus', description:'The shared state combines influences from the engine’s regions and steers the model’s next computation.', tag:'MEASURED', metrics:s=>[['State magnitude',s.bus?.state_norm],['Velocity',s.bus?.velocity_norm],['Divergence',s.bus?.divergence],['Temperature',s.bus?.temperature]], note:'Below: recent recorded writes, not percentages of responsibility for an answer.'},
    affect: {heading:'Affective influence', description:'Extracted model directions are combined with persistent emotion states. This blend changes the intervention applied to the model.',tag:'MEASURED', metrics:s=>Object.entries(s.limbic?.blended || {}).map(([k,v])=>[title(k),v]),note:'Activation is a functional signal; it is not a measurement of subjective feeling.'},
    memory: {heading:'Memory & continuity', description:'Stored latent traces can return through sampling, recall, and the self-model’s memory channel.',tag:'STORED STATE', metrics:s=>[['Traces',s.memory?.size],['Confabulated',s.memory?.n_false],['Continuity',s.self_model?.continuity],['Importance',s.memory?.avg_importance]],note:'Use the Memory view to inspect trace labels and provenance.'},
    sea: {heading:'The noise sea',description:'Noise, token proposals, and stored memory feed the bottom layer. Relevance filtering lets some candidates rise toward awareness.',tag:'PROPOSALS',metrics:s=>[['Resonant peaks',s.subconscious?.l1_count],['Relational survivors',s.subconscious?.l2_count]],note:'The map represents the algorithm. The noise geometry is not decoded thought content.'},
    peaks: {heading:'What reaches the surface',description:'Candidates are reweighted against affect and recent direction of change. A surprise channel can promote a less familiar candidate.',tag:'SELECTION',metrics:s=>[['Survivors',s.subconscious?.l2_count],['Attention gain',s.salience?.gain],['Novelty',s.salience?.novelty],['Future buffer',s.subconscious?.evaluated_futures]],note:'The selected candidate perturbs the state; it is not a factual assertion.'},
    imagination: {heading:'Recursive imagination',description:'Children continue their parent’s generated context. Paths are weighed within a shared compute budget; selected hypotheses influence subsequent activity.',tag:'IMAGINED',metrics:s=>[['Depth reached',s.speculative?.search?.depth_reached],['Nodes',s.speculative?.search?.nodes],['Tokens used',s.speculative?.search?.tokens],['Rounds',s.speculative?.rounds_total]],note:'Open Future branches to inspect actual parent/child links. Scores are heuristics, not event probabilities.'},
    expression: {heading:'The urge to respond',description:'Executive pressure and its current threshold regulate spontaneous speech. A direct message also initiates model generation.',tag:'BEHAVIOR',metrics:s=>[['Pressure',s.executive?.pressure],['Threshold',s.executive?.effective_threshold],['Seconds since speech',s.executive?.wall_seconds_since_speech]],note:'The last response’s recorded context is available by clicking “Inspect response context” in the conversation.'}
  };

  function inspect(type, value) {
    selection = {type, ...(type === 'region' ? {key:value} : {value})};
    document.querySelectorAll('[data-region]').forEach(e => e.classList.toggle('selected', type === 'region' && e.dataset.region === value));
    renderInspector();
    if (current && matchMedia('(max-width: 960px)').matches) {
      $('inspector-heading').scrollIntoView({block:'center',behavior:reducedMotion.matches?'auto':'smooth'});
    }
  }

  function renderInspector() {
    if (!current) return;
    const root = $('inspector-metrics'), influences = $('influence-list');
    root.replaceChildren(); influences.replaceChildren();
    let heading, description, tag, metrics = [], note;
    if (selection.type === 'region') {
      const r = regions[selection.key];
      ({heading, description, tag, note} = r); metrics = r.metrics(current);
      if (selection.key === 'bus') {
        Object.entries(current.bus_provenance || {}).sort((a,b)=>num(b[1].mean_norm)-num(a[1].mean_norm)).slice(0,4).forEach(([k,v]) => {
          const row = node('div','influence-row'); row.append(node('span','',title(k)),node('strong','',`${fmt(v.mean_norm,3)} · ${v.writes} writes`)); influences.append(row);
        });
      }
    } else if (selection.type === 'future') {
      const f = selection.value;
      heading = `Imagined event ${f.id + 1}`; description = f.name; tag = 'HYPOTHESIS';
      metrics = [['Depth', f.depth],['Path score', f.utility],['Parent',f.parent_id == null ? 'Root' : `Event ${f.parent_id+1}`],['Local score', f.local_utility]];
      note = `Captured from search round ${f.round ?? '—'}. This is an imagined possibility; scores combine token confidence and an affect proxy.`;
    } else if (selection.type === 'memory') {
      const m = selection.value; heading = 'A retained trace'; description = m.tag; tag = title(memoryType(m));
      metrics = [['Importance',m.importance],['Confidence',m.confidence]];
      note = 'Observed means a recorded internal experience; it does not verify an external-world claim.';
    } else if (selection.type === 'context') {
      const c = selection.value; heading = 'Context of this response'; description = c.chosen_future?.word || 'No selected future was recorded for this response.'; tag = 'RECORDED';
      metrics = [['Dominant affect',title(c.dominant_emotion || 'neutral')],['Gate action',title(c.gate?.action || 'unavailable')],['Future score',c.chosen_future?.utility],['Turn',c.turn]];
      note = 'A snapshot of associated engine state, not a causal explanation or a transcript of hidden reasoning.';
    } else {
      const e = selection.value; heading = title(e.kind); description = eventText(e); tag = 'JOURNALED';
      metrics = [['Event',e.id],['Turn',e.turn]];
      const pre = node('pre','',JSON.stringify(e.payload || {},null,2)); influences.append(pre);
      note = 'This event was emitted at a computation site in the engine.';
    }
    text('inspector-heading',heading); text('inspector-description',description); text('inspector-tag',tag); text('inspector-note',note);
    for (const [label,value] of metrics) {
      const card = node('div'); card.append(node('span','',label),node('strong','',typeof value === 'number' ? fmt(value,Number.isInteger(value)?0:2) : (value ?? '—'))); root.append(card);
    }
  }

  function renderTree() {
    const spec = current?.speculative || {}, search = spec.search || {};
    text('search-budget', `${search.tokens ?? 0} tokens · ${search.attempts ?? 0} attempts${search.budget_exhausted ? ' · budget reached' : ''}`);
    const key = JSON.stringify([spec.rounds_total,search.tree]);
    if (treeKey === key) { requestAnimationFrame(drawTreeEdges); return; }
    treeKey = key;
    const root = $('future-tree'), nodes = search.tree || [];
    root.replaceChildren();
    if (!nodes.length) { root.append(node('p','empty','No imagined branches yet. The search needs available model time to explore possible futures.')); return; }
    const chosen = spec.futures?.find(f=>f.chosen), path = new Set(), byId = new Map(nodes.map(n=>[n.id,n]));
    let cursor = chosen?.id;
    while (cursor != null && byId.has(cursor) && !path.has(cursor)) { path.add(cursor); cursor = byId.get(cursor).parent_id; }
    const layout = node('div','tree-layout');
    const depths = [...new Set(nodes.map(n=>n.depth))].sort((a,b)=>a-b);
    for (const depth of depths) {
      const column = node('div','tree-column'); column.append(node('span','tree-depth',`DEPTH ${depth}`));
      for (const f of nodes.filter(n=>n.depth===depth)) {
        const button = node('button',`future-node${path.has(f.id)?' chosen-path':''}`);
        button.dataset.nodeId = f.id; button.dataset.parentId = f.parent_id ?? ''; button.setAttribute('aria-label',`Inspect imagined event ${f.id+1}: ${f.name}`);
        const top = node('span','node-top'); top.append(node('span','',`EVENT ${f.id+1}`),node('span','',chosen?.id===f.id?'SELECTED':path.has(f.id)?'ON PATH':'IMAGINED'));
        button.append(top,node('p','',f.name),node('span','node-score',`Path score ${fmt(f.utility,3)}`));
        button.addEventListener('click',()=> { inspect('future',{...f,round:spec.rounds_total}); root.querySelectorAll('.selected').forEach(e=>e.classList.remove('selected')); button.classList.add('selected'); });
        column.append(button);
      }
      layout.append(column);
    }
    root.append(layout); requestAnimationFrame(drawTreeEdges);
  }

  function drawTreeEdges() {
    if (view !== 'futures') return;
    const layout = $('future-tree').querySelector('.tree-layout'); if (!layout) return;
    layout.querySelector('svg')?.remove();
    const rect = layout.getBoundingClientRect(), svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('class','tree-edges'); svg.setAttribute('width',layout.scrollWidth); svg.setAttribute('height',layout.scrollHeight);
    svg.setAttribute('aria-hidden','true');
    layout.querySelectorAll('[data-node-id]').forEach(child=> {
      if (!child.dataset.parentId) return;
      const parent = [...layout.querySelectorAll('[data-node-id]')].find(e=>e.dataset.nodeId===child.dataset.parentId);
      if (!parent) return;
      const a=parent.getBoundingClientRect(),b=child.getBoundingClientRect();
      const x1=a.right-rect.left,y1=a.top+a.height/2-rect.top,x2=b.left-rect.left,y2=b.top+b.height/2-rect.top;
      const line=document.createElementNS('http://www.w3.org/2000/svg','path');
      line.setAttribute('d',`M${x1} ${y1} C${(x1+x2)/2} ${y1},${(x1+x2)/2} ${y2},${x2} ${y2}`);
      if (child.classList.contains('chosen-path')) line.setAttribute('class','chosen'); svg.append(line);
    });
    layout.prepend(svg);
  }

  function memoryType(m) { return m.epistemic || (m.false ? 'confabulated' : 'observed'); }
  let memoryLoading = false;
  async function loadMemory() {
    if (memoryLoading) return;
    memoryLoading = true; lastMemoryFetch = performance.now();
    try { memories = (await request('/api/memory')).traces || []; lastFetchError=''; }
    catch (e) { lastFetchError = e.message; }
    finally { memoryLoading=false; if (!paused) renderMemories(); }
  }
  function renderMemories() {
    const root=$('memory-grid'), term=$('memory-search').value.trim().toLowerCase(), type=$('memory-filter').value;
    root.replaceChildren();
    if (lastFetchError) { root.append(node('p','empty',`Could not load memory: ${lastFetchError}`)); return; }
    const selected=[...memories].reverse().filter(m=>(type==='all'||memoryType(m)===type)&&String(m.tag).toLowerCase().includes(term));
    if (!selected.length) { root.append(node('p','empty',memories.length?'No traces match this filter.':'No memory traces have been recorded yet.')); return; }
    for (const m of selected) {
      const button=node('button','memory-card'); const kind=memoryType(m);
      button.append(node('span',`tag ${kind==='confabulated'?'amber':kind==='imagined'?'lavender':'mint'}`,title(kind)),node('p','',m.tag || 'Unlabelled trace'),node('span','',`Importance ${fmt(m.importance)} · confidence ${fmt(m.confidence)}`));
      button.addEventListener('click',()=>inspect('memory',m)); root.append(button);
    }
  }

  const eventKinds={future_considered:['Imagined a possibility','thought','#b8a1db'],word_chosen:['Selected a direction','thought','#92d9be'],emotion_snapshot:['Affect changed','affect','#dfbd84'],forecast_opened:['Registered a forecast','thought','#78beb8'],forecast_resolved:['Checked an expectation','behavior','#dfbd84'],gate_fired:['Adjusted its stance','behavior','#df9891'],reply_emitted:['Responded','behavior','#92d9be'],autonomous_reply:['Spoke spontaneously','behavior','#b8a1db']};
  function eventText(e) {
    const p=e.payload || {};
    if(e.kind==='emotion_snapshot')return `${title(p.dominant)} · activation ${fmt(p.value,3)}`;
    if(e.kind==='gate_fired')return `${title(p.action)} · divergence score ${fmt(p.misalignment)}`;
    if(e.kind==='forecast_resolved')return `${p.phrase || 'Forecast'} · ${title(p.status)}`;
    return p.word || p.reply || p.phrase || title(e.kind);
  }
  function renderEvents() {
    text('event-count',`${events.length} events`);
    const root=$('activity-feed'), scroll=root.scrollTop;
    root.replaceChildren();
    const selected=events.filter(e=>filter==='all'||(eventKinds[e.kind]?.[1]===filter)).slice(-70).reverse();
    if(!selected.length){root.append(node('p','empty small','Waiting for events in this category.'));return;}
    for(const e of selected){
      const [name,,color]=eventKinds[e.kind]||[title(e.kind),'behavior','#8c9c9e'];
      const item=node('div','event'); item.style.setProperty('--event-color',color); item.tabIndex=0; item.setAttribute('role','button'); item.setAttribute('aria-label',`Inspect event: ${name}`);
      const head=node('div','event-head'); const date=new Date(num(e.ts)*1000); head.append(node('span','',name),node('time','',date.toLocaleTimeString([],{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'})));
      item.append(head,node('p','',short(eventText(e),160)));
      const action=()=>inspect('event',e); item.addEventListener('click',action);item.addEventListener('keydown',k=>{if(k.key==='Enter'||k.key===' '){k.preventDefault();action();}});root.append(item);
    }
    root.scrollTop=scroll;
  }

  function appendMessage(role, content, context, autonomous=false) {
    if(!content) return;
    const root=$('conversation-list');root.querySelector('.conversation-empty')?.remove();
    const message=node('div',`message ${role}${autonomous?' autonomous':''}`);
    message.append(node('span','message-who',role==='user'?'YOU':role==='error'?'CONNECTION':'VITALIS'+(autonomous?' · SPONTANEOUS':'')),document.createTextNode(content));
    if(context){const b=node('button','context-button','Inspect response context ↗');b.type='button';b.addEventListener('click',()=>inspect('context',context));message.append(b);}
    root.append(message);while(root.children.length>60)root.firstElementChild.remove();root.scrollTop=root.scrollHeight;
  }
  async function restoreConversation() {
    const expected=session;
    try{
      const data=await request('/api/conversation');
      if(session!==expected||busy)return;
      $('conversation-list').replaceChildren();
      for(const m of data.messages||[]){if(['user','assistant'].includes(m.role))appendMessage(m.role,m.content);}
      if(!$('conversation-list').children.length)$('conversation-list').append(node('div','conversation-empty','Begin a conversation. Watch what changes inside.'));
    }catch(e){text('conversation-status','Conversation history unavailable');}
  }
  async function sendMessage(e) {
    e.preventDefault(); const input=$('message-input'),message=input.value.trim();
    if(!message||busy||connection!=='live')return;
    busy=true; input.value=''; $('send-message').disabled=true;
    appendMessage('user',message);text('conversation-status','The model is responding…');
    if(current&&!paused)render(current);
    try{
      const data=await request('/api/talk',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,max_tokens:100})});
      appendMessage('assistant',data.response,data.context); if(data.state)accept(data.state);
    }catch(err){appendMessage('error',err.message);if(!input.value)input.value=message;}
    finally{busy=false;$('send-message').disabled=connection!=='live';text('conversation-status','Your words become part of the context');if(current&&!paused)render(current);input.focus();}
  }

  function selectView(next, focus=false) {
    if(!['map','futures','memory'].includes(next))return;
    view=next;
    document.querySelectorAll('[data-view]').forEach(b=>{const selected=b.dataset.view===next;b.setAttribute('aria-selected',selected);b.tabIndex=selected?0:-1;if(selected&&focus)b.focus();});
    ['map','futures','memory'].forEach(k=>$(`view-${k}`).hidden=k!==next);
    history.replaceState(null,'',next==='map'?location.pathname:`#${next}`);
    if(next==='futures'&&current)renderTree();if(next==='memory'){renderMemories();loadMemory();}if(next==='map')drawField();
  }
  function drawField() {
    const canvas=$('field-canvas'),box=canvas.getBoundingClientRect();if(!box.width)return;
    const ratio=Math.min(devicePixelRatio||1,2);canvas.width=box.width*ratio;canvas.height=box.height*ratio;
    const c=canvas.getContext('2d');if(!c)return;c.scale(ratio,ratio);
    const size=box.width,cx=size/2,cy=box.height/2,velocity=clamp(num(current?.bus?.velocity_norm)/4),magnitude=clamp(num(current?.bus?.state_norm)/8);
    if(!reducedMotion.matches&&connection==='live'&&!paused)phase=num(current?.bus?.tick_count)*.018;
    const glow=c.createRadialGradient(cx,cy,size*.12,cx,cy,size*.49);glow.addColorStop(0,'#68baa715');glow.addColorStop(.65,'#68baa718');glow.addColorStop(1,'#68baa700');c.fillStyle=glow;c.fillRect(0,0,size,box.height);
    for(let ring=0;ring<7;ring++){
      c.beginPath();
      for(let j=0;j<=160;j++){
        const a=j/160*Math.PI*2, base=size*(.31+ring*.018);
        const wave=(Math.sin(a*3+phase+ring*.22)+Math.cos(a*5-phase*.5+ring*.1))*(1+velocity*4);
        const r=base+wave;
        const x=cx+Math.cos(a)*r,y=cy+Math.sin(a)*r;
        j?c.lineTo(x,y):c.moveTo(x,y);
      }
      c.closePath();c.strokeStyle=`rgba(146,217,190,${.12+ring*.065+magnitude*.06})`;c.lineWidth=ring===6?1:.65;c.stroke();
    }
    for(let i=0;i<12;i++){
      const a=i/12*Math.PI*2+phase*.07,r=size*.46;
      c.beginPath();c.arc(cx+Math.cos(a)*r,cy+Math.sin(a)*r, i%3===0?1.5:.8,0,Math.PI*2);c.fillStyle='#74b5a98c';c.fill();
    }
  }

  function connect() {
    source=new EventSource('/api/stream');
    source.onmessage=e=>{try{accept(JSON.parse(e.data));}catch{setConnection('error','The engine sent invalid telemetry. Waiting for a valid update.');}};
    source.onerror=()=>setConnection('offline');
  }
  document.querySelectorAll('[data-region]').forEach(b=>b.addEventListener('click',()=>inspect('region',b.dataset.region)));
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>selectView(b.dataset.view)));
  document.querySelector('[role=tablist]').addEventListener('keydown',e=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();
    const views=['map','futures','memory'],index=views.indexOf(view);
    selectView(views[e.key==='Home'?0:e.key==='End'?2:(index+(e.key==='ArrowRight'?1:2))%3],true);
  });
  document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(f=>f.setAttribute('aria-pressed',f===b));renderEvents();}));
  $('pause-view').addEventListener('click',()=>{
    paused=!paused;document.body.classList.toggle('view-paused',paused);$('pause-view').setAttribute('aria-pressed',paused);
    $('pause-view').setAttribute('aria-label',paused?'Resume visualization':'Pause visualization');text('pause-view',paused?'▷':'Ⅱ');
    if(!paused&&pending){render(pending);if(view==='memory')renderMemories();}
  });
  $('refresh-memory').addEventListener('click',loadMemory);$('memory-search').addEventListener('input',renderMemories);$('memory-filter').addEventListener('change',renderMemories);
  $('conversation-form').addEventListener('submit',sendMessage);
  $('message-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('conversation-form').requestSubmit();}});
  $('export-session').addEventListener('click',async()=>{
    const button=$('export-session');button.disabled=true;
    try{const data=await request('/api/export');data.observatory={events,viewed_state:current,view_paused:paused};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=`vitalis-session-${new Date().toISOString().replaceAll(':','-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    catch(e){setConnection('error',`Export failed: ${e.message}`);}finally{button.disabled=false;}
  });
  window.addEventListener('hashchange',()=>selectView(location.hash.slice(1)||'map'));
  new ResizeObserver(()=>{drawField();drawTreeEdges();}).observe($('circuit'));
  window.addEventListener('resize',drawTreeEdges);
  window.addEventListener('pagehide',()=>source?.close());
  window.addEventListener('pageshow',e=>{if(e.persisted)connect();});
  setInterval(()=>{
    const now=performance.now();
    if(lastReceived&&now-lastReceived>5000)setConnection('offline');
    else if(lastReceived&&now-lastTickAt>5000)setConnection('stalled');
  },1000);
  meters('modulation-bars',['arousal','reward','calm','stress'].map(k=>[k,null]),1.5);
  setConnection('connecting');selectView(location.hash.slice(1)||'map');inspect('region','bus');drawField();connect();
})();
