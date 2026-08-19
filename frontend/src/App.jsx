import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import "./professional.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";
const CHAT_KEY = "sickleguide_chats_v5";
const THEME_KEY = "sickleguide_theme_v5";
const GRAPH_POSITIONS_KEY = "sickleguide_graph_positions_v1";

const CORE_SOURCES = [
  ["ASH — Sickle Cell Disease Clinical Practice Guidelines.pdf", "ASH clinical practice guidelines", "Clinical Guideline"],
  ["Evidence-BasedManagement ofSickle Cell Disease.pdf", "Evidence-Based Management of Sickle Cell Disease", "Evidence Review"],
  ["Watermarked ASH SCD Transfusion Pocket Guide.pdf", "ASH SCD Transfusion Pocket Guide", "Pocket Guide"],
  ["WHO consolidated guidelinesfor the management of commonchildhood illness.pdf", "WHO childhood illness guidelines", "WHO Guideline"],
  ["WHO recommendations on themanagement of sickle-cell diseaseduring pregnancy, childbirth andthe interpregnancy period.pdf", "WHO pregnancy recommendations", "WHO Guideline"],
];

const id = () => crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
const newChat = () => ({ id: id(), title: "New conversation", createdAt: Date.now(), updatedAt: Date.now(), messages: [] });
const shortTitle = (text) => {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > 42 ? `${clean.slice(0, 42)}…` : clean || "New conversation";
};

function MarkdownText({ text }) {
  const blocks = String(text || "").split(/\n\n+/);
  return <div className="rich-answer">{blocks.map((block, i) => <p key={i}>{block.split("\n").map((line, j) => <span key={j}>{line}{j < block.split("\n").length - 1 && <br />}</span>)}</p>)}</div>;
}

function LiveEvaluation({ evaluation }) {
  if (!evaluation) return null;
  const pct = value => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
  return <div className="live-evaluation-card">
    <div className="live-evaluation-head"><div><span className="eyebrow">AUTOMATIC QUALITY CHECK</span><h3>Response evaluation</h3></div><span className={`quality-pill ${evaluation.grounded && evaluation.citations_valid ? "good" : "review"}`}>{evaluation.grounded && evaluation.citations_valid ? "Verified" : "Needs review"}</span></div>
    <div className="live-evaluation-grid">
      <div><span>Grounding</span><strong>{evaluation.grounded ? "Pass" : "Review"}</strong></div>
      <div><span>Citations</span><strong>{evaluation.citations_valid ? "Valid" : "Review"}</strong></div>
      <div><span>Evidence used</span><strong>{evaluation.evidence_count ?? "—"}</strong></div>
      <div><span>Precision@5</span><strong>{pct(evaluation["precision@5"])}</strong></div>
      <div><span>Recall@5</span><strong>{pct(evaluation["recall@5"])}</strong></div>
      <div><span>MRR</span><strong>{evaluation.mrr == null ? "—" : Number(evaluation.mrr).toFixed(3)}</strong></div>
    </div>
    {evaluation.precision_status && evaluation.precision_status !== "benchmark case" && <p className="evaluation-note">Precision/Recall are shown when the question belongs to the labeled evaluation set. Grounding and citation checks run automatically for every answer.</p>}
  </div>;
}

function App() {
  const [page, setPage] = useState("chat");
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");
  const [chats, setChats] = useState(() => {
    try { return JSON.parse(localStorage.getItem(CHAT_KEY)) || [newChat()]; } catch { return [newChat()]; }
  });
  const [activeChatId, setActiveChatId] = useState(() => localStorage.getItem(`${CHAT_KEY}_active`) || null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [streamSources, setStreamSources] = useState([]);
  const [streamStage, setStreamStage] = useState("");
  const [streamEvaluation, setStreamEvaluation] = useState(null);
  const [error, setError] = useState("");

  const graphRef = useRef(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [], total_nodes: 0, total_edges: 0 });
  const [graphView, setGraphView] = useState("overview");
  const [graphMaxNodes, setGraphMaxNodes] = useState(140);
  const [graphSearch, setGraphSearch] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState("");
  const [visibleTypes, setVisibleTypes] = useState([]);
  const [visibleRelations, setVisibleRelations] = useState([]);
  const [allTypes, setAllTypes] = useState([]);
  const [allRelations, setAllRelations] = useState([]);
  const [graphPositions, setGraphPositions] = useState(() => {
    try { return JSON.parse(localStorage.getItem(GRAPH_POSITIONS_KEY)) || {}; } catch { return {}; }
  });

  const [dataFiles, setDataFiles] = useState([]);
  const [dataStats, setDataStats] = useState({ total_files: 0, total_chunks: 0 });
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const [evaluationMode, setEvaluationMode] = useState(false);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [evaluationStage, setEvaluationStage] = useState("");
  const [evaluationProgress, setEvaluationProgress] = useState(0);

  const activeChat = useMemo(() => chats.find(c => c.id === activeChatId) || chats[0] || null, [chats, activeChatId]);

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem(THEME_KEY, theme); }, [theme]);
  useEffect(() => { localStorage.setItem(CHAT_KEY, JSON.stringify(chats)); }, [chats]);
  useEffect(() => { if (activeChat?.id) { setActiveChatId(activeChat.id); localStorage.setItem(`${CHAT_KEY}_active`, activeChat.id); } }, [activeChat]);
  useEffect(() => { localStorage.setItem(GRAPH_POSITIONS_KEY, JSON.stringify(graphPositions)); }, [graphPositions]);

  const updateChat = (chatId, updater) => setChats(items => items.map(c => c.id === chatId ? updater(c) : c));
  const createNewChat = () => { const c = newChat(); setChats(items => [c, ...items]); setActiveChatId(c.id); setPage("chat"); setQuery(""); setError(""); setStreamingAnswer(""); setStreamEvaluation(null); };
  const deleteChat = (chatId) => {
    const next = chats.filter(c => c.id !== chatId);
    const fallback = next.length ? next : [newChat()];
    setChats(fallback); if (activeChatId === chatId) setActiveChatId(fallback[0].id);
  };

  const askSickleGuide = async () => {
    const clean = query.trim();
    if (!clean || loading || !activeChat) return;
    setLoading(true); setError(""); setStreamingAnswer(""); setStreamSources([]); setStreamEvaluation(null); setStreamStage("Searching clinical evidence...");
    const userMessage = { id: id(), role: "user", content: clean, createdAt: Date.now() };
    const history = activeChat.messages.map(m => ({ role: m.role, content: m.content }));
    updateChat(activeChat.id, c => ({ ...c, title: c.messages.length ? c.title : shortTitle(clean), updatedAt: Date.now(), messages: [...c.messages, userMessage] }));
    setQuery("");
    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify({ query: clean, chat_id: activeChat.id, history }) });
      if (!response.ok || !response.body) throw new Error(await response.text() || "Unable to start the response.");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let answer = ""; let sources = []; let liveEvaluation = null;
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n"); buffer = events.pop() || "";
        for (const event of events) {
          const line = event.split("\n").find(x => x.startsWith("data:")); if (!line) continue;
          const payload = JSON.parse(line.slice(5).trim());
          if (payload.type === "status") setStreamStage(payload.message || "Working...");
          if (payload.type === "token") { answer += payload.content || ""; setStreamingAnswer(answer); setStreamStage("Writing verified answer..."); }
          if (payload.type === "sources") { sources = payload.sources || []; setStreamSources(sources); }
          if (payload.type === "live_evaluation") { liveEvaluation = payload.evaluation || null; setStreamEvaluation(liveEvaluation); }
          if (payload.type === "error") throw new Error(payload.message || "Response failed.");
        }
      }
      if (answer) updateChat(activeChat.id, c => ({ ...c, updatedAt: Date.now(), messages: [...c.messages, { id: id(), role: "assistant", content: answer, sources, liveEvaluation, createdAt: Date.now() }] }));
    } catch (e) { setError(e?.message || "Could not connect to SickleGuide."); }
    finally { setLoading(false); setStreamStage(""); }
  };

  const handleKeyDown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askSickleGuide(); } };

  const loadGraph = async () => {
    setGraphLoading(true); setGraphError("");
    try {
      const params = new URLSearchParams({ view: graphView, max_nodes: String(graphMaxNodes) });
      const res = await fetch(`${API_BASE_URL}/graph?${params}`); const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Graph request failed.");
      const nodes = (data.nodes || []).map(n => graphPositions[n.id] ? { ...n, fx: graphPositions[n.id].x, fy: graphPositions[n.id].y } : n);
      setGraphData({ nodes, links: data.edges || [], total_nodes: data.total_nodes || 0, total_edges: data.total_edges || 0 });
      const types = data.available_entity_types || []; const relations = data.available_relations || [];
      setAllTypes(types); setAllRelations(relations); setVisibleTypes(types); setVisibleRelations(relations); setSelectedNode(null);
    } catch (e) { setGraphError(e?.message || "Could not load the knowledge graph."); }
    finally { setGraphLoading(false); }
  };
  useEffect(() => { if (page === "graph") loadGraph(); if (page === "data") loadData(); }, [page]);

  const graphFilteredData = useMemo(() => {
    const needle = graphSearch.trim().toLowerCase();
    const nodes = graphData.nodes.filter(n => visibleTypes.includes(n.type) && (!needle || String(n.name || "").toLowerCase().includes(needle) || String(n.id).toLowerCase().includes(needle)));
    const ids = new Set(nodes.map(n => n.id));
    const links = graphData.links.filter(l => visibleRelations.includes(l.relation) && ids.has(typeof l.source === "object" ? l.source.id : l.source) && ids.has(typeof l.target === "object" ? l.target.id : l.target));
    return { nodes, links };
  }, [graphData, visibleTypes, visibleRelations, graphSearch]);

  const neighbors = useMemo(() => {
    if (!selectedNode) return [];
    const ids = new Set();
    graphData.links.forEach(l => { const s = typeof l.source === "object" ? l.source.id : l.source; const t = typeof l.target === "object" ? l.target.id : l.target; if (s === selectedNode.id) ids.add(t); if (t === selectedNode.id) ids.add(s); });
    return graphData.nodes.filter(n => ids.has(n.id));
  }, [selectedNode, graphData]);

  const focusNode = n => { if (!n) return; setSelectedNode(n); graphRef.current?.centerAt(n.x, n.y, 500); graphRef.current?.zoom(4, 500); };
  const fitGraph = () => graphRef.current?.zoomToFit(500, 55);
  const saveNodePosition = n => { if (!n?.id || typeof n.x !== "number") return; setGraphPositions(p => ({ ...p, [n.id]: { x: n.x, y: n.y } })); };

  const loadData = async () => {
    try { const res = await fetch(`${API_BASE_URL}/data`); const data = await res.json(); if (!res.ok) throw new Error(data.detail || "Dataset loading failed."); setDataFiles(data.files || []); setDataStats({ total_files: data.total_files || 0, total_chunks: data.total_chunks || 0 }); }
    catch (e) { setError(e?.message || "Could not load the knowledge base."); }
  };
  const uploadPDF = async file => {
    if (!file) return; if (!file.name.toLowerCase().endsWith(".pdf")) { setUploadMessage("Only PDF files are supported."); return; }
    setUploadLoading(true); setUploadMessage("Processing and indexing document...");
    try { const form = new FormData(); form.append("file", file); const res = await fetch(`${API_BASE_URL}/data/upload`, { method: "POST", body: form }); const data = await res.json(); if (!res.ok) throw new Error(data.detail || "Upload failed."); setUploadMessage(`✓ ${data.filename} added — ${data.chunks_added} chunks indexed.`); await loadData(); }
    catch (e) { setUploadMessage(e?.message || "Upload failed."); } finally { setUploadLoading(false); }
  };

  const runEvaluation = async () => {
    setEvaluationLoading(true); setEvaluationResult(null); setEvaluationProgress(4); setEvaluationStage("Preparing evaluation dataset...");
    const stages = ["Preparing evaluation dataset...", "Running retrieval metrics...", "Evaluating reranking...", "Checking grounding and citations...", "Building quality report..."];
    let timer = setInterval(() => setEvaluationProgress(p => Math.min(90, p + 7)), 900);
    try {
      for (let i = 1; i < stages.length; i++) { setEvaluationStage(stages[i]); await new Promise(r => setTimeout(r, 500)); }
      const res = await fetch(`${API_BASE_URL}/evaluation/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ full: evaluationMode }) });
      const data = await res.json(); if (!res.ok) throw new Error(data.detail || "Evaluation failed.");
      setEvaluationResult(data.report); setEvaluationProgress(100); setEvaluationStage("Evaluation complete");
    } catch (e) { setError(e?.message || "Evaluation failed."); setEvaluationProgress(0); }
    finally { clearInterval(timer); setEvaluationLoading(false); }
  };

  const renderSidebar = () => (
    <aside className={`sidebar pro-sidebar ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-topline"><button className="icon-button" onClick={() => setCollapsed(v => !v)} title={collapsed ? "Open sidebar" : "Close sidebar"}>{collapsed ? "☰" : "‹"}</button>{!collapsed && <span className="sidebar-caption">SICKLEGUIDE</span>}</div>
      <div className="sidebar-brand"><div className="logo-mark">S</div>{!collapsed && <div><div className="sidebar-title">SickleGuide</div><div className="sidebar-subtitle">Clinical evidence assistant</div></div>}</div>
      <button className="new-chat-button" onClick={createNewChat}>{collapsed ? "+" : "+ New chat"}</button>
      <nav className="main-nav">
        {[['chat','Chat','◌'],['graph','Knowledge Graph','◇'],['pipeline','Pipeline','↗'],['data','Knowledge Base','▣'],['evaluation','Evaluation','◎']].map(([key,label,icon]) => <button key={key} className={`nav-item ${page === key ? "active" : ""}`} onClick={() => setPage(key)}><span>{icon}</span>{!collapsed && label}</button>)}
      </nav>
      {!collapsed && <><div className="sidebar-section-label">Conversations</div><div className="chat-history">{chats.slice().sort((a,b)=>b.updatedAt-a.updatedAt).map(c => <div className={`history-item ${c.id === activeChatId ? "active" : ""}`} key={c.id}><button onClick={() => { setActiveChatId(c.id); setPage("chat"); }}>{c.title}</button><button className="history-delete" onClick={() => deleteChat(c.id)}>×</button></div>)}</div><div className="sidebar-footer"><button className="theme-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")}>{theme === "light" ? "☾ Dark mode" : "☀ Light mode"}</button></div></>}
    </aside>
  );

  const renderChat = () => (
    <div className="chat-screen">
      <header className="chat-header"><div><span className="eyebrow">CLINICAL ASSISTANT</span><h1>{activeChat?.title || "SickleGuide"}</h1></div><div className="safe-pill"><span className="online-dot"/> Evidence grounded</div></header>
      <section className="chat-area">
        {!activeChat?.messages?.length ? <div className="chat-empty pro-empty"><div className="large-logo">S</div><div className="gradient-badge">Evidence → Retrieval → Verification</div><h2>How can I help?</h2><p>Ask a clinical question and get an evidence-grounded response with traceable sources.</p><div className="quick-grid">{["What treatments were evaluated for acute chest syndrome?","What is recommended for secondary stroke prevention?","What are the recommendations during pregnancy?"] .map(q => <button key={q} onClick={() => setQuery(q)}>{q}<span>↗</span></button>)}</div></div> : <div className="messages">{activeChat.messages.map(m => <div className={`message-row ${m.role}`} key={m.id}><div className="message-avatar">{m.role === "user" ? "You" : "S"}</div><div className="message-body"><div className="message-role">{m.role === "user" ? "You" : "SickleGuide"}</div><div className="message-bubble">{m.role === "assistant" ? <MarkdownText text={m.content}/> : m.content}</div>{m.sources?.length > 0 && <div className="source-strip"><div className="source-strip-title">Evidence used</div>{m.sources.map(s => <div className="source-chip" key={`${s.evidence_id}-${s.citation}`}>[{s.evidence_id}] {s.citation}</div>)}</div>}{m.role === "assistant" && m.liveEvaluation && <LiveEvaluation evaluation={m.liveEvaluation}/>}</div></div>)}{loading && <div className="message-row assistant"><div className="message-avatar">S</div><div className="message-body"><div className="message-role">SickleGuide</div><div className="stream-box"><div className="stream-stage"><span className="spinner"/>{streamStage}</div>{streamingAnswer && <div className="stream-answer"><MarkdownText text={streamingAnswer}/><span className="cursor">▌</span></div>}{streamEvaluation && <LiveEvaluation evaluation={streamEvaluation}/>} {streamSources.length > 0 && <div className="source-strip compact"><div className="source-strip-title">Evidence</div>{streamSources.map(s => <div key={`${s.evidence_id}-${s.citation}`} className="source-chip">[{s.evidence_id}] {s.citation}</div>)}</div>}</div></div></div>}</div>}
        {error && <div className="error-box">{error}</div>}
      </section>
      <div className="composer-area"><div className="composer"><textarea value={query} onChange={e => setQuery(e.target.value)} onKeyDown={handleKeyDown} placeholder="Message SickleGuide..." rows={1} disabled={loading}/><button onClick={askSickleGuide} disabled={loading || !query.trim()}>{loading ? "…" : "↑"}</button></div><div className="composer-note">Responses are generated from retrieved clinical evidence. Automatic quality checks run after every answer.</div></div>
    </div>
  );

  const renderGraph = () => (
    <div className="page graph-page"><div className="page-toolbar"><div><div className="eyebrow">KNOWLEDGE GRAPH</div><h1>Explore clinical relationships</h1><p className="page-description">Move concepts freely, zoom into the graph, search entities and inspect every property of a selected node.</p></div><div className="graph-actions"><button className="secondary-button" onClick={fitGraph}>Fit view</button><button className="primary-button" onClick={loadGraph}>Refresh</button></div></div>
      <div className="graph-workspace"><aside className="graph-sidebar"><div className="graph-search"><span>⌕</span><input value={graphSearch} onChange={e=>setGraphSearch(e.target.value)} placeholder="Search concepts..."/></div><div className="control-block"><label>Graph view</label><div className="view-options">{[['overview','Clinical Overview'],['treatments','Treatments'],['complications','Complications'],['transfusion','Transfusion'],['pregnancy','Pregnancy']].map(v=><button key={v[0]} className={`view-option ${graphView===v[0]?"active":""}`} onClick={()=>setGraphView(v[0])}>{graphView===v[0]?"●":"○"} {v[1]}</button>)}</div></div><div className="control-block"><div className="control-heading"><label>Visible nodes</label><span>{graphFilteredData.nodes.length}</span></div><input className="range" type="range" min="40" max="250" step="10" value={graphMaxNodes} onChange={e=>setGraphMaxNodes(Number(e.target.value))}/><button className="control-apply" onClick={loadGraph}>Apply</button></div><div className="graph-tip"><strong>Graph controls</strong><p>Drag nodes to arrange them. Drag empty space to pan. Scroll to zoom. Click any node for full details.</p></div>{selectedNode && <div className="selected-card"><div className="selected-label">SELECTED ENTITY</div><h3>{selectedNode.name}</h3><span className="selected-type">{selectedNode.type}</span><div className="selected-stats"><strong>{neighbors.length}</strong><span>connected concepts</span></div><button className="focus-button" onClick={()=>focusNode(selectedNode)}>Focus on entity</button><div className="neighbor-list">{neighbors.slice(0,8).map(n=><button key={n.id} onClick={()=>focusNode(n)}>{n.name}</button>)}</div></div>}</aside>
        <div className="graph-canvas-card">{graphLoading ? <div className="graph-loading"><span className="spinner large"/><strong>Building clinical graph…</strong><p>Preparing the selected medical view.</p></div> : graphError ? <div className="graph-loading"><strong>Graph unavailable</strong><p>{graphError}</p></div> : <ForceGraph2D ref={graphRef} graphData={graphFilteredData} nodeId="id" linkSource="source" linkTarget="target" nodeLabel={n=>`${n.name}\n${n.type}`} linkLabel={l=>l.relation || ""} nodeColor={n=>selectedNode?.id===n.id ? "#7c87f2" : "#6878a0"} nodeRelSize={6} linkColor={()=>theme==="dark"?"rgba(185,195,220,.3)":"rgba(91,105,137,.28)"} linkWidth={l=>{if(!selectedNode)return 1;const s=typeof l.source==="object"?l.source.id:l.source;const t=typeof l.target==="object"?l.target.id:l.target;return s===selectedNode.id||t===selectedNode.id?3:.8;}} linkDirectionalArrowLength={4} linkDirectionalArrowRelPos={1} cooldownTicks={90} d3AlphaDecay={0.03} d3VelocityDecay={0.24} onNodeClick={n=>focusNode(n)} onNodeDragEnd={n=>saveNodePosition(n)} nodeCanvasObjectMode={()=>"after"} nodeCanvasObject={(n,ctx,scale)=>{const size=Math.max(9/scale,2.5);ctx.font=`${size}px Inter,sans-serif`;ctx.textAlign="center";ctx.fillStyle=theme==="dark"?"#edf2fa":"#2b3448";ctx.fillText(String(n.name||"").length>24?`${String(n.name).slice(0,24)}…`:n.name,n.x,n.y+12);}}/>}<div className="graph-canvas-hud"><div><strong>{graphFilteredData.nodes.length}</strong><span>visible nodes</span></div><div><strong>{graphFilteredData.links.length}</strong><span>relations</span></div><div><strong>{graphView}</strong><span>view</span></div></div>{selectedNode && <div className="node-details"><div className="node-details-head"><div><span>NODE DETAILS</span><h2>{selectedNode.name}</h2></div><button onClick={()=>setSelectedNode(null)}>×</button></div><div className="detail-type">{selectedNode.type}</div><div className="detail-grid">{Object.entries(selectedNode).filter(([k])=>!['x','y','vx','vy','index','fx','fy'].includes(k)).map(([k,v])=><div className="detail-item" key={k}><span>{k.replace(/_/g,' ')}</span><strong>{typeof v === 'object' ? JSON.stringify(v) : String(v ?? '—')}</strong></div>)}</div><div className="node-relations"><h3>Connections</h3>{neighbors.length ? neighbors.map(n=><button key={n.id} onClick={()=>focusNode(n)}>{n.name}<span>{n.type}</span></button>) : <p>No visible connections.</p>}</div></div>}</div>
      </div>
    </div>
  );

  const renderData = () => { const coreNames = new Set(CORE_SOURCES.map(x=>x[0])); const uploaded = dataFiles.filter(f=>!coreNames.has(f.name)); return <div className="page"><div className="page-toolbar"><div><div className="eyebrow">KNOWLEDGE BASE</div><h1>Clinical evidence</h1><p className="page-description">Manage the evidence available to the retrieval pipeline.</p></div></div><div className="dataset-summary"><div className="summary-card"><span>Core documents</span><strong>5</strong></div><div className="summary-card"><span>Indexed documents</span><strong>{dataStats.total_files}</strong></div><div className="summary-card"><span>Indexed chunks</span><strong>{dataStats.total_chunks}</strong></div></div><section className="source-section"><div className="section-heading"><div><div className="eyebrow">ORIGINAL DATASET</div><h2>Core sources</h2></div></div><div className="dataset-grid">{CORE_SOURCES.map(([name,short,kind])=><div className="dataset-card core" key={name}><div className="pdf-icon">PDF</div><div className="dataset-main"><div className="dataset-badges"><span className="badge-core">CORE</span><span className="badge-type">{kind}</span></div><h3>{short}</h3><p>{name}</p></div></div>)}</div></section><section className="source-section"><div className="section-heading"><div><div className="eyebrow">EXTEND</div><h2>Add clinical PDF</h2></div></div><div className={`dropzone ${dragActive?"drag-active":""}`} onClick={()=>document.getElementById("pdf-upload")?.click()} onDragOver={e=>{e.preventDefault();setDragActive(true)}} onDragLeave={()=>setDragActive(false)} onDrop={e=>{e.preventDefault();setDragActive(false);uploadPDF(e.dataTransfer.files?.[0])}}><div className="dropzone-icon">↑</div><div className="dropzone-content"><strong>Drop a PDF here</strong><p>Or click to choose a clinical document.</p><span>Parse → Clean → Chunk → Embed → Index</span></div><input id="pdf-upload" hidden type="file" accept=".pdf,application/pdf" disabled={uploadLoading} onChange={e=>{uploadPDF(e.target.files?.[0]);e.target.value=""}}/></div>{uploadMessage&&<div className="upload-message">{uploadMessage}</div>}</section>{uploaded.length>0&&<section className="source-section"><div className="section-heading"><div><div className="eyebrow">ADDITIONAL</div><h2>Uploaded documents</h2></div></div><div className="dataset-grid">{uploaded.map(f=><div className="dataset-card uploaded" key={f.name}><div className="pdf-icon">PDF</div><div className="dataset-main"><span className="badge-uploaded">UPLOADED</span><h3>{f.name}</h3><p>{f.size_mb} MB</p><div className="dataset-meta"><span>{f.chunks} chunks</span><span>Searchable</span></div></div></div>)}</div></section>}</div>; };

  const renderPipeline = () => <div className="page"><div className="page-toolbar"><div><div className="eyebrow">SYSTEM ARCHITECTURE</div><h1>From evidence to verified answer</h1><p className="page-description">A transparent view of the retrieval, reasoning and safety pipeline.</p></div></div><div className="pipeline-grid">{[['01','PDF Sources','Curated clinical guidelines and evidence.'],['02','Ingestion','Page-aware parsing and metadata extraction.'],['03','Cleaning','Extraction noise is removed before retrieval.'],['04','Chunking','Context-preserving chunks with citations.'],['05','Embeddings','Clinical content is represented for semantic retrieval.'],['06','Hybrid Retrieval','Dense, lexical and graph signals are fused.'],['07','Reranking','Strongest evidence is promoted to the top.'],['08','Generation','Answer is produced from retrieved evidence.'],['09','Grounding','Unsupported claims are checked.'],['10','Citations','Evidence references are validated.'],['11','Safety','The system fails closed when evidence is insufficient.']].map(s=><div className="pipeline-card" key={s[0]}><div className="stage-number">{s[0]}</div><div><h3>{s[1]}</h3><p>{s[2]}</p></div></div>)}</div></div>;

  const renderEvaluation = () => { const r=evaluationResult?.retrieval?.summary; const e=evaluationResult?.end_to_end?.summary; const metrics=[['Precision@5',r?`${(r['candidate_precision@5']*100).toFixed(1)}%`:"—"],['Reranked Precision@5',r?`${(r['reranked_precision@5']*100).toFixed(1)}%`:"—"],['Recall@5',r?`${(r['candidate_recall@5']*100).toFixed(1)}%`:"—"],['Reranked Recall@5',r?`${(r['reranked_recall@5']*100).toFixed(1)}%`:"—"],['MRR',r?Number(r.mrr).toFixed(3):"—"],['Grounding',e?`${(e.grounded_rate*100).toFixed(1)}%`:"—"],['Citation validity',e?`${(e.citation_valid_rate*100).toFixed(1)}%`:"—"],['Answer coverage',e?`${(e.answer_term_coverage*100).toFixed(1)}%`:"—"]]; return <div className="page"><div className="page-toolbar"><div><div className="eyebrow">QUALITY LAB</div><h1>Evaluation</h1><p className="page-description">Measure retrieval precision, recall, reranking, grounding, citations and end-to-end answer quality.</p></div><button className="primary-button" onClick={runEvaluation} disabled={evaluationLoading}>{evaluationLoading?"Running…":"Run evaluation"}</button></div><div className="evaluation-controls"><div><h3>Evaluation mode</h3><p>Retrieval is faster. Full mode also evaluates generation and grounding.</p></div><label className="toggle"><input type="checkbox" checked={evaluationMode} onChange={e=>setEvaluationMode(e.target.checked)}/><span/>Full end-to-end</label></div>{evaluationLoading&&<div className="evaluation-progress"><div className="progress-head"><strong>{evaluationStage}</strong><span>{evaluationProgress}%</span></div><div className="progress-track"><div style={{width:`${evaluationProgress}%`}}/></div><div className="progress-steps"><span>Dataset</span><span>Retrieval</span><span>Reranking</span><span>Grounding</span><span>Report</span></div></div>}<div className="metric-grid">{metrics.map(m=><div className="metric-card" key={m[0]}><span>{m[0]}</span><strong>{m[1]}</strong></div>)}</div><div className="evaluation-method-grid">{[['Retrieval','Precision@K, Recall@K and source recall measure evidence relevance and coverage.'],['Reranking','Precision and recall show whether useful evidence reaches the top after reranking.'],['Grounding','Checks medical claims against retrieved evidence.'],['Citations','Checks evidence references before output.'],['Safety','Prevents unsupported behavior.'],['End-to-End','Measures the full SickleGuide pipeline.']].map(x=><div className="evaluation-method" key={x[0]}><div className="method-icon">✓</div><div><h3>{x[0]}</h3><p>{x[1]}</p></div></div>)}</div></div>; };

  const pageContent = page === "chat" ? renderChat() : page === "graph" ? renderGraph() : page === "data" ? renderData() : page === "pipeline" ? renderPipeline() : renderEvaluation();
  return <div className="app-shell"><button className="mobile-sidebar-toggle" onClick={()=>setCollapsed(v=>!v)}>☰</button>{renderSidebar()}<main className="main-shell">{pageContent}</main></div>;
}

export default App;
