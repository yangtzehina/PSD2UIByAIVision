import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Boxes, BrainCircuit, Download, FileJson, FileText, Image as ImageIcon, KeyRound, Layers, Play, RotateCcw, Save, Upload } from "lucide-react";
import "./styles.css";

const nodeTypes = ["Screen", "Container", "Image", "Icon", "Text", "Button", "Input", "Toggle", "Slider", "ScrollView", "List", "Grid", "Background", "Unknown"];

const palette = {
  Screen: "#0f172a",
  Container: "#ef4444",
  Image: "#f59e0b",
  Icon: "#8b5cf6",
  Text: "#2563eb",
  Button: "#14b8a6",
  Input: "#06b6d4",
  Toggle: "#84cc16",
  Slider: "#eab308",
  ScrollView: "#22c55e",
  List: "#22c55e",
  Grid: "#22c55e",
  Background: "#64748b",
  Unknown: "#f97316",
};

const semanticRules = [
  "Do not invent pixel coordinates.",
  "Return one item per useful candidate id when possible.",
  "Prefer PSD layer text over OCR guesses.",
  "Do not return Screen for candidates; Screen is a synthetic root created by the program.",
  "Do not downgrade a concrete local type to Unknown unless the local type is already Unknown.",
  "Do not reclassify local Text candidates as Image, Container, or decorative controls.",
  "Do not switch high-confidence concrete candidates across type families.",
  "Only change Container into Button/Input/Toggle/Slider when component_group_id evidence is present.",
  "Use Unknown for ambiguous decorative elements.",
  "Use List/Grid/ScrollView only for repeated or scrollable regions.",
  "style may be empty unless a supplied text style is clearly useful.",
  "parent_candidate_id may be empty when uncertain.",
  "component_group_id may be empty; use it only when multiple candidates should be wrapped into one UI component.",
];

const openAISemanticsSchema = {
  type: "object",
  additionalProperties: false,
  required: ["items"],
  properties: {
    items: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["candidate_id", "type", "confidence", "role", "text", "style", "layout", "parent_candidate_id", "component_group_id"],
        properties: {
          candidate_id: { type: "string" },
          type: { type: "string", enum: nodeTypes },
          confidence: { type: "number", minimum: 0, maximum: 1 },
          role: { type: "string" },
          text: { type: "string" },
          style: { type: "string" },
          layout: { type: "string" },
          parent_candidate_id: { type: "string" },
          component_group_id: { type: "string" },
        },
      },
    },
  },
};

function App() {
  const [imageUrl, setImageUrl] = useState("");
  const [graphImageUrl, setGraphImageUrl] = useState("");
  const [renderDiffImageUrl, setRenderDiffImageUrl] = useState("");
  const [semanticImageDataUrl, setSemanticImageDataUrl] = useState("");
  const [uiir, setUiir] = useState(null);
  const [baselineUiir, setBaselineUiir] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [layers, setLayers] = useState([]);
  const [visionQuarantined, setVisionQuarantined] = useState([]);
  const [visionRejected, setVisionRejected] = useState([]);
  const [semanticPatches, setSemanticPatches] = useState([]);
  const [renderReview, setRenderReview] = useState(null);
  const [xml, setXml] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [corrections, setCorrections] = useState([]);
  const [goldenDecisions, setGoldenDecisions] = useState([]);
  const [providerResult, setProviderResult] = useState(null);
  const [treeMode, setTreeMode] = useState("uiir");
  const [boxFilter, setBoxFilter] = useState("all");
  const [stageMode, setStageMode] = useState("box");
  const nodes = useMemo(() => flattenNodes(uiir?.root), [uiir]);
  const psdTree = useMemo(() => buildPsdTree(nodes), [nodes]);
  const baselineNodes = useMemo(() => flattenNodes(baselineUiir?.root), [baselineUiir]);
  const diffByNodeId = useMemo(() => buildNodeDiffs(baselineNodes, nodes), [baselineNodes, nodes]);
  const semanticPatchByCandidate = useMemo(() => groupSemanticPatches(semanticPatches), [semanticPatches]);
  const enrichedCandidates = useMemo(() => enrichCandidates(candidates, semanticPatchByCandidate), [candidates, semanticPatchByCandidate]);
  const rejectedBoxes = useMemo(() => rejectedProposalBoxes(visionRejected), [visionRejected]);
  const quarantinedBoxes = useMemo(() => proposalBoxes(visionQuarantined, "openai-vision-quarantined"), [visionQuarantined]);
  const usingCandidates = enrichedCandidates.length > 0;
  const sourceBoxes = usingCandidates ? enrichedCandidates : nodes;
  const boxes = useMemo(() => filterReviewBoxes(sourceBoxes, rejectedBoxes, quarantinedBoxes, boxFilter), [sourceBoxes, rejectedBoxes, quarantinedBoxes, boxFilter]);
  const selectedCandidate = boxes.find((item) => item.id === selectedId) || null;
  const selectedNode = nodes.find((item) => item.id === selectedId) || null;
  const selected = selectedCandidate || selectedNode;
  const selectedUsesCandidate = Boolean(selectedCandidate);
  const selectedGoldenDecision = selected ? getGoldenDecision(goldenDecisions, selected, selectedUsesCandidate) : null;
  const stageImageUrl = stageMode === "graph" ? graphImageUrl || imageUrl : stageMode === "render" ? renderDiffImageUrl || imageUrl : imageUrl;
  const stageAlt = stageMode === "graph" ? "Graph overlay" : stageMode === "render" ? "Render diff" : "PSD composite";

  const loadDemo = () => {
    const demo = createDemoData();
    setImageUrl(demo.imageDataUrl);
    setGraphImageUrl("");
    setRenderDiffImageUrl("");
    setSemanticImageDataUrl(demo.imageDataUrl);
    setUiir(demo.uiir);
    setBaselineUiir(null);
    setComparison(null);
    setCandidates(demo.candidates);
    setLayers(demo.layers);
    setVisionQuarantined([]);
    setVisionRejected([]);
    setSemanticPatches([]);
    setRenderReview(null);
    setXml(demo.xml);
    setCorrections([]);
    setGoldenDecisions([]);
    setSelectedId("");
    setProviderResult(null);
  };

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("demo") === "1") {
      loadDemo();
    }
  }, []);

  const upsertCorrection = (patch) => {
    if (!selected) return;
    const identity = selectedUsesCandidate ? { candidate_id: selected.id } : { node_id: selected.id };
    setCorrections((items) => {
      const index = items.findIndex((item) => sameTarget(item, identity));
      const next = [...items];
      if (index >= 0) {
        next[index] = compactCorrection({ ...next[index], ...patch });
      } else {
        next.push(compactCorrection({ ...identity, ...patch }));
      }
      return next.filter((item) => Object.keys(item).length > Object.keys(identity).length);
    });
  };

  const resetSelected = () => {
    if (!selected) return;
    const identity = selectedUsesCandidate ? { candidate_id: selected.id } : { node_id: selected.id };
    setCorrections((items) => items.filter((item) => !sameTarget(item, identity)));
  };

  const upsertGoldenDecision = (decision, includeEdits = false) => {
    if (!selected) return;
    const identity = decisionIdentityFor(selected, selectedUsesCandidate);
    const correction = getCorrection(corrections, selected, selectedUsesCandidate);
    const patch = includeEdits ? decisionOverrides(mergeTarget(selected, correction)) : {};
    setGoldenDecisions((items) => {
      const index = items.findIndex((item) => item.target_kind === identity.target_kind && item.target_id === identity.target_id);
      const next = [...items];
      const value = compactDecision({ ...identity, decision, ...patch });
      if (index >= 0) {
        next[index] = value;
      } else {
        next.push(value);
      }
      return next;
    });
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Layers size={24} />
          <div>
            <h1>UIIR Inspector</h1>
            <p>PSD analysis review surface</p>
          </div>
        </div>

        <FileInput icon={<ImageIcon size={18} />} label="Composite/Overlay PNG" accept="image/*" onFile={readImage(setImageUrl, setSemanticImageDataUrl)} />
        <FileInput icon={<ImageIcon size={18} />} label="graph_overlay.png" accept="image/*" onFile={readImage(setGraphImageUrl)} />
        <FileInput icon={<ImageIcon size={18} />} label="render_diff.png" accept="image/*" onFile={readImage(setRenderDiffImageUrl)} />
        <FileInput icon={<FileJson size={18} />} label="uiir.json" accept=".json,application/json" onFile={readJson(setUiir)} />
        <FileInput icon={<FileJson size={18} />} label="baseline uiir.json" accept=".json,application/json" onFile={readJson(setBaselineUiir)} />
        <FileInput icon={<FileJson size={18} />} label="comparison.json" accept=".json,application/json" onFile={readJson(setComparison)} />
        <FileInput icon={<Boxes size={18} />} label="candidates.json" accept=".json,application/json" onFile={readJson(setCandidates)} />
        <FileInput icon={<Layers size={18} />} label="layer_metadata.json" accept=".json,application/json" onFile={readLayers(setLayers)} />
        <FileInput icon={<FileJson size={18} />} label="vision_quarantined.json" accept=".json,application/json" onFile={readJson(setVisionQuarantined)} />
        <FileInput icon={<FileJson size={18} />} label="vision_rejected.json" accept=".json,application/json" onFile={readJson(setVisionRejected)} />
        <FileInput icon={<FileJson size={18} />} label="semantic_patches.json" accept=".json,application/json" onFile={readJson(setSemanticPatches)} />
        <FileInput icon={<FileJson size={18} />} label="render_review.json" accept=".json,application/json" onFile={readJson(setRenderReview)} />
        <FileInput icon={<FileText size={18} />} label="uiir.xml" accept=".xml,text/xml" onFile={readText(setXml)} />
        <FileInput icon={<FileJson size={18} />} label="corrections.json" accept=".json,application/json" onFile={readCorrections(setCorrections)} />
        <FileInput icon={<FileJson size={18} />} label="golden_decisions.json" accept=".json,application/json" onFile={readGoldenDecisions(setGoldenDecisions)} />
        <button className="demoButton" type="button" onClick={loadDemo}>
          <BrainCircuit size={18} />
          <span>Load demo sample</span>
        </button>

        <section className="stats">
          <Metric label="Canvas" value={uiir ? `${uiir.width} x ${uiir.height}` : "Waiting"} />
          <Metric label="Nodes" value={nodes.length || "0"} />
          <Metric label="Boxes" value={boxes.length || "0"} />
          <Metric label="Layers" value={layers.length || "0"} />
          <Metric label="Quarantine" value={quarantinedBoxes.length || "0"} />
          <Metric label="Rejected" value={rejectedBoxes.length || "0"} />
          <Metric label="Render Issues" value={renderReview?.issue_count ?? renderReview?.issues?.length ?? "0"} />
          <Metric label="Edits" value={corrections.length || "0"} />
          <Metric label="Decisions" value={goldenDecisions.length || "0"} />
          <Metric label="Diffs" value={Object.keys(diffByNodeId).length || "0"} />
        </section>

        <section className="reviewFilters">
          <header>Review View</header>
          <div className="segmented vertical">
            {[
              ["box", "Box Overlay"],
              ["graph", "Graph Overlay"],
              ["render", "Render Diff"],
            ].map(([value, label]) => (
              <button key={value} className={stageMode === value ? "active" : ""} type="button" onClick={() => setStageMode(value)}>
                {label}
              </button>
            ))}
          </div>
        </section>

        <section className="reviewFilters">
          <header>Review Filter</header>
          <div className="segmented vertical">
            {[
              ["all", "All"],
              ["local", "Local"],
              ["accepted", "GPT Accepted"],
              ["quarantined", "GPT Quarantined"],
              ["rejected", "GPT Rejected"],
              ["semantic", "Semantic Patch"],
            ].map(([value, label]) => (
              <button key={value} className={boxFilter === value ? "active" : ""} type="button" onClick={() => setBoxFilter(value)}>
                {label}
              </button>
            ))}
          </div>
        </section>

        <DiffPanel comparison={comparison} diffCount={Object.keys(diffByNodeId).length} renderReview={renderReview} />
        <ProviderSmokePanel
          imageDataUrl={semanticImageDataUrl}
          candidates={candidates}
          layers={layers}
          result={providerResult}
          onResult={setProviderResult}
        />

        <CorrectionEditor
          selected={selected}
          usingCandidates={selectedUsesCandidate}
          correction={selected ? getCorrection(corrections, selected, selectedUsesCandidate) : null}
          onChange={upsertCorrection}
          onReset={resetSelected}
          onExport={() => exportCorrections(corrections)}
          correctionCount={corrections.length}
        />
        <GoldenDecisionPanel
          selected={selected}
          usingCandidates={selectedUsesCandidate}
          decision={selectedGoldenDecision}
          onDecide={upsertGoldenDecision}
          onExport={() => exportGoldenDecisions(goldenDecisions)}
          decisionCount={goldenDecisions.length}
        />
      </aside>

      <section className="workspace">
        <div className="stage">
          {stageImageUrl ? (
            <div className="imageWrap">
              <img src={stageImageUrl} alt={stageAlt} />
              {stageMode === "box" ? boxes.map((box, index) => {
                const correction = getCorrection(corrections, box, usingCandidates);
                return (
                  <OverlayBox
                    key={`${box.id || index}-${index}`}
                    box={correction?.bbox || box.bbox || box}
                    label={box.id || `b${index + 1}`}
                    type={correction?.type || box.type || box.type_hint || "Unknown"}
                    changed={Boolean(diffByNodeId[box.id])}
                    ignored={correction?.ignored === true}
                    active={selectedId === box.id}
                    onClick={() => setSelectedId(box.id || "")}
                  />
                );
              }) : null}
            </div>
          ) : (
            <div className="empty">
              <Upload size={38} />
              <span>Load CLI output files to inspect the inferred UI structure.</span>
            </div>
          )}
        </div>

        <div className="panels">
          <section className="panel">
            <header>
              <span>{treeMode === "uiir" ? "UIIR Tree" : "PSD Tree"}</span>
              <div className="segmented">
                <button className={treeMode === "uiir" ? "active" : ""} onClick={() => setTreeMode("uiir")}>UIIR</button>
                <button className={treeMode === "psd" ? "active" : ""} onClick={() => setTreeMode("psd")}>PSD</button>
              </div>
            </header>
            <div className="tree">{uiir?.root ? <Tree node={treeMode === "uiir" ? uiir.root : psdTree} selectedId={selectedId} onSelect={setSelectedId} corrections={corrections} diffs={diffByNodeId} /> : <Muted text="Load uiir.json" />}</div>
          </section>
          <section className="panel">
            <header>XML</header>
            <pre className="xml">{xml || "Load uiir.xml"}</pre>
          </section>
        </div>
      </section>
    </main>
  );
}

function DiffPanel({ comparison, diffCount, renderReview }) {
  if (!comparison && !diffCount && !renderReview) return null;
  return (
    <section className="diffPanel">
      <header>OpenAI Diff</header>
      {comparison ? (
        <div className="diffStats">
          <Metric label="Status" value={comparison.status || "unknown"} />
          <Metric label="Model" value={comparison.model || "n/a"} />
          <Metric label="Base Pixel" value={comparison.baseline?.avg_pixel_similarity ?? "n/a"} />
          <Metric label="OpenAI Pixel" value={comparison.openai?.avg_pixel_similarity ?? "n/a"} />
        </div>
      ) : null}
      <div className="diffList">
        {(comparison?.items || []).map((item) => (
          <div className="diffItem" key={item.name}>
            <strong>{item.name}</strong>
            <span>{item.type_changes?.length || 0} type changes</span>
            <span>unknown {signed(item.unknown_delta)}</span>
            <span>pixel {signed(item.pixel_similarity_delta)}</span>
          </div>
        ))}
        {!comparison ? <Muted text="Load comparison.json" /> : null}
      </div>
      {renderReview ? (
        <div className="renderIssues">
          <strong>Render Review</strong>
          {(renderReview.issues || []).slice(0, 6).map((issue) => (
            <div className="diffItem" key={issue.id || `${issue.type}-${issue.reason}`}>
              <strong>{issue.type || "issue"}</strong>
              <span>{issue.severity || "n/a"}</span>
              <span>{issue.reason || ""}</span>
            </div>
          ))}
          {!(renderReview.issues || []).length ? <Muted text="No render issues" /> : null}
        </div>
      ) : null}
    </section>
  );
}

function ProviderSmokePanel({ imageDataUrl, candidates, layers, result, onResult }) {
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const [token, setToken] = useState("");
  const [model, setModel] = useState("gpt-5.5");
  const [apiMode, setApiMode] = useState("responses");
  const [detail, setDetail] = useState("original");
  const [running, setRunning] = useState(false);
  const ready = Boolean(imageDataUrl && candidates.length && token.trim() && model.trim() && baseUrl.trim());
  const summary = result?.parsed ? summarizeSemanticItems(result.parsed.items || []) : null;

  const runSmoke = async () => {
    setRunning(true);
    onResult(null);
    try {
      const next = await runProviderSemanticSmoke({
        baseUrl,
        token,
        model,
        apiMode,
        detail,
        imageDataUrl,
        candidates,
        layers,
      });
      onResult(next);
    } catch (error) {
      onResult({
        ok: false,
        error: friendlyProviderError(error),
        provider: { baseUrl: normalizedBaseUrl(baseUrl), model, apiMode, detail },
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="providerPanel">
      <header>
        <span>Provider Smoke</span>
        <BrainCircuit size={17} />
      </header>
      <div className="providerBody">
        <div className="tokenNote">
          <KeyRound size={15} />
          <span>Token stays in this browser tab. It is never saved to the repository.</span>
        </div>
        <label>
          <span>Base URL</span>
          <input value={baseUrl} placeholder="https://api.openai.com/v1" onChange={(event) => setBaseUrl(event.target.value)} />
        </label>
        <label>
          <span>Token</span>
          <input type="password" value={token} autoComplete="off" placeholder="Provider API key" onChange={(event) => setToken(event.target.value)} />
        </label>
        <div className="providerGrid">
          <label>
            <span>Model</span>
            <input value={model} onChange={(event) => setModel(event.target.value)} />
          </label>
          <label>
            <span>API</span>
            <select value={apiMode} onChange={(event) => setApiMode(event.target.value)}>
              <option value="responses">responses</option>
              <option value="chat-completions">chat</option>
            </select>
          </label>
        </div>
        <label>
          <span>Detail</span>
          <select value={detail} onChange={(event) => setDetail(event.target.value)}>
            <option value="original">original</option>
            <option value="high">high</option>
            <option value="low">low</option>
            <option value="auto">auto</option>
          </select>
        </label>
        <div className="providerActions">
          <button type="button" onClick={runSmoke} disabled={!ready || running} title="Run provider semantic smoke">
            <Play size={16} />
            <span>{running ? "Running" : "Run"}</span>
          </button>
          <button type="button" onClick={() => setToken("")} disabled={!token} title="Clear token">
            <KeyRound size={16} />
          </button>
          <button type="button" onClick={() => result && exportProviderResult(result)} disabled={!result} title="Download provider result">
            <Download size={16} />
          </button>
        </div>
        <div className="providerReadiness">
          <span className={imageDataUrl ? "ok" : ""}>image</span>
          <span className={candidates.length ? "ok" : ""}>{candidates.length || 0} candidates</span>
          <span className={layers.length ? "ok" : ""}>{layers.length || 0} layers</span>
        </div>
        {result ? (
          <div className={`providerResult ${result.ok ? "ok" : "failed"}`}>
            <strong>{result.ok ? "OK" : "Failed"}</strong>
            {result.error ? <span>{result.error}</span> : null}
            {summary ? (
              <>
                <span>{summary.count} returned items</span>
                <span>role {summary.roleFill}% · layout {summary.layoutFill}% · parent {summary.parentFill}%</span>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function FileInput({ icon, label, accept, onFile }) {
  return (
    <label className="fileInput">
      {icon}
      <span>{label}</span>
      <input type="file" accept={accept} onChange={(event) => event.target.files?.[0] && onFile(event.target.files[0])} />
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CorrectionEditor({ selected, usingCandidates, correction, onChange, onReset, onExport, correctionCount }) {
  const display = mergeTarget(selected, correction);
  return (
    <section className="editor">
      <header>
        <span>{selected ? selected.id : "No selection"}</span>
        <div className="editorActions">
          <button type="button" onClick={onReset} disabled={!selected} title="Reset selected correction">
            <RotateCcw size={16} />
          </button>
          <button type="button" onClick={onExport} disabled={!correctionCount} title="Export corrections.json">
            <Save size={16} />
          </button>
        </div>
      </header>
      {selected ? (
        <div className="editorBody">
          <label>
            <span>Type</span>
            <select value={display.type || "Unknown"} onChange={(event) => onChange({ type: event.target.value })}>
              {nodeTypes.filter((type) => type !== "Screen" || !usingCandidates).map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Role</span>
            <input value={display.role || ""} onChange={(event) => onChange({ role: event.target.value })} />
          </label>
          <label>
            <span>Text</span>
            <input value={display.text || ""} onChange={(event) => onChange({ text: event.target.value })} />
          </label>
          <label>
            <span>Style</span>
            <input value={display.style || ""} onChange={(event) => onChange({ style: event.target.value })} />
          </label>
          <label>
            <span>Layout</span>
            <select value={display.layout || ""} onChange={(event) => onChange({ layout: event.target.value })}>
              <option value="">None</option>
              <option value="vertical">vertical</option>
              <option value="horizontal">horizontal</option>
              <option value="grid">grid</option>
              <option value="freeform">freeform</option>
            </select>
          </label>
          <label>
            <span>Parent</span>
            <input value={display.parent_id || ""} onChange={(event) => onChange({ parent_id: event.target.value })} />
          </label>
          <BBoxEditor bbox={display.bbox} onChange={(bbox) => onChange({ bbox })} />
          <NodeDetails node={display} />
          <label className="check">
            <input type="checkbox" checked={display.ignored === true} onChange={(event) => onChange({ ignored: event.target.checked })} />
            <span>Ignored</span>
          </label>
        </div>
      ) : (
        <Muted text="Select a box or tree node" />
      )}
    </section>
  );
}

function GoldenDecisionPanel({ selected, usingCandidates, decision, onDecide, onExport, decisionCount }) {
  const canDecide = Boolean(selected);
  const identity = selected ? decisionIdentityFor(selected, usingCandidates) : null;
  return (
    <section className="decisionPanel">
      <header>
        <span>Golden Decision</span>
        <button type="button" onClick={onExport} disabled={!decisionCount} title="Export golden_decisions.json">
          <Save size={16} />
        </button>
      </header>
      {selected ? (
        <>
          <div className="decisionTarget">
            <span>{identity.target_kind}</span>
            <strong>{identity.target_id}</strong>
          </div>
          <div className="decisionActions">
            <button type="button" onClick={() => onDecide("accept")} disabled={!canDecide}>Accept</button>
            <button type="button" onClick={() => onDecide("reject")} disabled={!canDecide}>Reject</button>
            <button type="button" onClick={() => onDecide("edit", true)} disabled={!canDecide}>Edit</button>
            <button type="button" onClick={() => onDecide("ignore")} disabled={!canDecide}>Ignore</button>
          </div>
          <div className="decisionStatus">
            <span>Current</span>
            <strong>{decision?.decision || "none"}</strong>
          </div>
        </>
      ) : (
        <Muted text="Select a quarantined proposal or node" />
      )}
    </section>
  );
}

function NodeDetails({ node }) {
  const metadata = node.metadata || {};
  const refs = node.sourceRefs || node.source_refs || [];
  const rejected = metadata.openaiRejected || [];
  const semanticPatches = metadata.openaiSemanticPatches || metadata.externalSemanticPatches || [];
  const related = metadata.relatedCandidateIds || metadata.related_candidate_ids || [];
  return (
    <div className="nodeDetails">
      <div>
        <span>Source</span>
        <strong>{refs.join(", ") || "n/a"}</strong>
      </div>
      <div>
        <span>PSD Path</span>
        <strong>{metadata.psdPath || "n/a"}</strong>
      </div>
      <div>
        <span>Grouping</span>
        <strong>{metadata.groupingReason || (metadata.component ? "component" : "n/a")}</strong>
      </div>
      <div>
        <span>Proposal</span>
        <strong>{metadata.proposalReason || metadata.openaiVisionProposal?.reason || "n/a"}</strong>
      </div>
      <div>
        <span>Rejected</span>
        <strong>{metadata.rejectionReason || metadata.quarantineReason || rejected.map((item) => item.reason).filter(Boolean).join(", ") || "n/a"}</strong>
      </div>
      <div>
        <span>Related</span>
        <strong>{related.join(", ") || "n/a"}</strong>
      </div>
      <div>
        <span>Semantic</span>
        <strong>{semanticPatches.length ? `${semanticPatches.length} patch` : "n/a"}</strong>
      </div>
      <div>
        <span>Decision</span>
        <strong>{metadata.goldenDecision?.decision || "n/a"}</strong>
      </div>
    </div>
  );
}

function BBoxEditor({ bbox, onChange }) {
  const box = normalizeBox(bbox) || { x: 0, y: 0, w: 0, h: 0 };
  return (
    <div className="bboxGrid">
      {["x", "y", "w", "h"].map((key) => (
        <label key={key}>
          <span>{key}</span>
          <input
            type="number"
            value={box[key] ?? 0}
            onChange={(event) =>
              onChange({
                ...box,
                [key]: Number(event.target.value),
              })
            }
          />
        </label>
      ))}
    </div>
  );
}

function OverlayBox({ box, label, type, active, ignored, changed, onClick }) {
  const bbox = normalizeBox(box);
  if (!bbox || !bbox.w || !bbox.h) return null;
  const color = ignored ? "#6b7280" : palette[type] || palette.Unknown;
  return (
    <button
      className={`overlayBox ${active ? "active" : ""} ${ignored ? "ignored" : ""} ${changed ? "changed" : ""}`}
      style={{
        left: `${bbox.x}px`,
        top: `${bbox.y}px`,
        width: `${bbox.w}px`,
        height: `${bbox.h}px`,
        borderColor: color,
        color,
      }}
      onClick={onClick}
      title={`${label} ${type}`}
    >
      <span style={{ background: color }}>{label}</span>
    </button>
  );
}

function Tree({ node, selectedId, onSelect, corrections, diffs }) {
  const correction = getCorrection(corrections, node, false);
  const display = mergeTarget(node, correction);
  const bbox = normalizeBox(display.bbox);
  const diff = diffs?.[node.id];
  return (
    <div className={`treeNode ${diff ? "changed" : ""}`}>
      <button className={selectedId === node.id ? "selected" : ""} onClick={() => onSelect(node.id)}>
        <span className="type" style={{ color: palette[display.type] || palette.Unknown }}>{display.type}</span>
        <span>{node.id}</span>
        <small>{bbox ? `${bbox.x},${bbox.y},${bbox.w},${bbox.h}` : ""}</small>
        {diff ? <small>{diff.before}{" -> "}{diff.after}</small> : null}
        {display.text ? <em>{display.text}</em> : null}
      </button>
      {node.children?.length ? (
        <div className="children">
          {node.children.map((child) => (
            <Tree key={child.id} node={child} selectedId={selectedId} onSelect={onSelect} corrections={corrections} diffs={diffs} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function buildNodeDiffs(baselineNodes, currentNodes) {
  if (!baselineNodes.length || !currentNodes.length) return {};
  const baseline = Object.fromEntries(baselineNodes.map((node) => [node.id, node]));
  const diffs = {};
  for (const node of currentNodes) {
    const before = baseline[node.id];
    if (before && before.type !== node.type) {
      diffs[node.id] = { before: before.type, after: node.type };
    }
  }
  return diffs;
}

function signed(value) {
  if (value === null || value === undefined) return "n/a";
  return Number(value) > 0 ? `+${value}` : `${value}`;
}

function Muted({ text }) {
  return <div className="muted">{text}</div>;
}

function sameTarget(left, right) {
  return (left.candidate_id && left.candidate_id === right.candidate_id) || (left.node_id && left.node_id === right.node_id);
}

function sameDecisionTarget(left, right) {
  return left.target_kind === right.target_kind && left.target_id === right.target_id;
}

function getCorrection(corrections, target, usingCandidates) {
  if (!target) return null;
  const key = usingCandidates ? "candidate_id" : "node_id";
  return corrections.find((item) => item[key] === target.id) || null;
}

function getGoldenDecision(decisions, target, usingCandidates) {
  if (!target) return null;
  const identity = decisionIdentityFor(target, usingCandidates);
  return decisions.find((item) => sameDecisionTarget(item, identity)) || null;
}

function decisionIdentityFor(target, usingCandidates) {
  const source = String(target.source || "");
  if (source.includes("openai-vision-quarantined") || source.includes("openai-vision-rejected")) {
    return { target_kind: "proposal", target_id: target.proposal_id || stripTargetPrefix(target.id) };
  }
  return usingCandidates ? { target_kind: "candidate", target_id: target.id } : { target_kind: "node", target_id: target.id };
}

function stripTargetPrefix(value) {
  const text = String(value || "");
  return text.includes(":") ? text.split(":").pop() : text;
}

function mergeTarget(target, correction) {
  if (!target) return {};
  return {
    id: target.id,
    type: target.type || target.type_hint || "Unknown",
    bbox: target.bbox,
    role: target.role || "",
    text: target.text || "",
    style: target.style || "",
    layout: target.layout || "",
    parent_id: target.parent_hint || "",
    sourceRefs: target.sourceRefs || target.source_refs || [],
    metadata: target.metadata || {},
    ignored: false,
    ...correction,
  };
}

function decisionOverrides(target) {
  const fields = {};
  for (const key of ["bbox", "type", "role", "text", "style", "layout", "parent_id"]) {
    if (target[key] !== "" && target[key] !== null && target[key] !== undefined) {
      fields[key] = target[key];
    }
  }
  return fields;
}

function compactDecision(item) {
  const compacted = {};
  for (const [key, value] of Object.entries(item)) {
    if (value === "" || value === null || value === undefined) continue;
    compacted[key] = value;
  }
  return compacted;
}

function compactCorrection(item) {
  const compacted = {};
  for (const [key, value] of Object.entries(item)) {
    if (value === "" || value === null || value === undefined) continue;
    if (key === "ignored" && value === false) continue;
    compacted[key] = value;
  }
  return compacted;
}

function normalizeBox(box) {
  if (!box) return null;
  if (Array.isArray(box)) {
    const [x, y, w, h] = box;
    return { x, y, w, h };
  }
  return box;
}

function flattenNodes(root) {
  if (!root) return [];
  const result = [];
  const visit = (node) => {
    result.push(node);
    node.children?.forEach(visit);
  };
  visit(root);
  return result;
}

function groupSemanticPatches(patches) {
  const grouped = {};
  const items = Array.isArray(patches) ? patches : patches?.items || patches?.patches || [];
  for (const patch of items) {
    if (!patch?.candidate_id) continue;
    grouped[patch.candidate_id] = [...(grouped[patch.candidate_id] || []), patch];
  }
  return grouped;
}

function enrichCandidates(candidates, semanticPatchByCandidate) {
  return candidates.map((candidate) => {
    const externalSemanticPatches = semanticPatchByCandidate[candidate.id] || [];
    if (!externalSemanticPatches.length) return candidate;
    return {
      ...candidate,
      metadata: {
        ...(candidate.metadata || {}),
        externalSemanticPatches,
      },
    };
  });
}

function rejectedProposalBoxes(rejected) {
  return proposalBoxes(rejected, "openai-vision-rejected");
}

function proposalBoxes(proposals, source) {
  const items = Array.isArray(proposals) ? proposals : proposals?.items || [];
  return items.map((item, index) => {
    const proposalId = item.proposal_id || `r${index + 1}`;
    return {
      id: `${source}:${proposalId}`,
      proposal_id: proposalId,
      target_kind: "proposal",
      type_hint: item.type || "Unknown",
      bbox: item.bbox,
      source,
      source_refs: [`${source}:${proposalId}`],
      role: item.role || "",
      text: item.text || "",
      metadata: {
        rejectionReason: item.rejectionReason || item.rejection_reason || "",
        quarantineReason: item.quarantineReason || item.quarantine_reason || "",
        proposalReason: item.reason || "",
        goldenDecision: item.goldenDecision || null,
        relatedCandidateIds: item.related_candidate_ids || [],
      },
    };
  });
}

function filterReviewBoxes(sourceBoxes, rejectedBoxes, quarantinedBoxes, filter) {
  if (filter === "rejected") return rejectedBoxes;
  if (filter === "quarantined") return quarantinedBoxes;
  if (filter === "local") return sourceBoxes.filter((box) => !isGptAcceptedBox(box) && !hasSemanticPatch(box));
  if (filter === "accepted") return sourceBoxes.filter(isGptAcceptedBox);
  if (filter === "semantic") return sourceBoxes.filter(hasSemanticPatch);
  return sourceBoxes;
}

function isGptAcceptedBox(box) {
  const metadata = box.metadata || {};
  return box.source === "openai-vision-proposal" || Boolean(metadata.openaiVision?.accepted) || Boolean(metadata.openaiVisionProposals?.length);
}

function hasSemanticPatch(box) {
  const metadata = box.metadata || {};
  return Boolean(metadata.openaiSemanticPatches?.length || metadata.externalSemanticPatches?.length || metadata.openaiRejected?.length);
}

function buildPsdTree(nodes) {
  const root = {
    id: "psd-root",
    type: "Screen",
    bbox: { x: 0, y: 0, w: 0, h: 0 },
    sourceRefs: ["psd"],
    metadata: { psdPath: "PSD" },
    children: [],
  };
  const layerNodes = nodes.filter((node) => !node.metadata?.component && firstLayerRef(node));
  const byLayerId = Object.fromEntries(layerNodes.map((node) => [firstLayerRef(node), { ...node, children: [] }]));
  for (const node of Object.values(byLayerId)) {
    const parentId = node.metadata?.psdParentId;
    const parent = parentId ? byLayerId[parentId] : null;
    if (parent && parent !== node) {
      parent.children.push(node);
    } else {
      root.children.push(node);
    }
  }
  sortPsdTree(root);
  return root;
}

function firstLayerRef(node) {
  return (node.sourceRefs || []).find((ref) => typeof ref === "string" && ref.startsWith("layer:"));
}

function sortPsdTree(node) {
  node.children?.sort((left, right) => {
    const leftDepth = left.metadata?.psdDepth ?? 0;
    const rightDepth = right.metadata?.psdDepth ?? 0;
    return leftDepth - rightDepth || String(left.metadata?.psdPath || "").localeCompare(String(right.metadata?.psdPath || ""));
  });
  node.children?.forEach(sortPsdTree);
}

function readImage(setter, dataSetter) {
  return async (file) => {
    setter(URL.createObjectURL(file));
    if (dataSetter) {
      dataSetter(await fileToDataUrl(file));
    }
  };
}

function readJson(setter) {
  return async (file) => setter(JSON.parse(await file.text()));
}

function readLayers(setter) {
  return async (file) => {
    const parsed = JSON.parse(await file.text());
    setter(Array.isArray(parsed) ? parsed : parsed.layers || []);
  };
}

function readCorrections(setter) {
  return async (file) => {
    const parsed = JSON.parse(await file.text());
    setter(Array.isArray(parsed) ? parsed : parsed.corrections || []);
  };
}

function readGoldenDecisions(setter) {
  return async (file) => {
    const parsed = JSON.parse(await file.text());
    setter(Array.isArray(parsed) ? parsed : parsed.decisions || []);
  };
}

function readText(setter) {
  return async (file) => setter(await file.text());
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function exportCorrections(corrections) {
  const payload = JSON.stringify({ version: "0.1", corrections }, null, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "corrections.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportGoldenDecisions(decisions) {
  const payload = JSON.stringify({ version: "0.1", decisions }, null, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "golden_decisions.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

async function runProviderSemanticSmoke({ baseUrl, token, model, apiMode, detail, imageDataUrl, candidates, layers }) {
  const payload = {
    task: "Classify PSD UI candidates and refine UI semantic hints.",
    prompt_version: "semantic_v2_browser_smoke",
    rules: semanticRules,
    candidates: candidates.slice(0, 220).map(candidateSummary),
    layers: layers.slice(0, 260).map(layerSummary),
  };
  const prompt = [
    "You are refining a PSD-to-UI intermediate representation.",
    "The image has candidate boxes overlaid with ids.",
    "Return semantic classifications that match the supplied JSON schema.",
    "",
    JSON.stringify(payload),
  ].join("\n");
  const endpoint = `${normalizedBaseUrl(baseUrl)}/${apiMode === "chat-completions" ? "chat/completions" : "responses"}`;
  const requestBody = apiMode === "chat-completions"
    ? chatCompletionsBody(model, prompt, imageDataUrl, detail)
    : responsesBody(model, prompt, imageDataUrl, detail);
  const started = performance.now();
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(requestBody),
  });
  const rawResponse = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${rawResponse.slice(0, 700)}`);
  }
  const responseJson = JSON.parse(rawResponse);
  const rawText = responseText(responseJson);
  const parsed = JSON.parse(rawText);
  return {
    ok: true,
    seconds: Math.round(performance.now() - started) / 1000,
    provider: {
      baseUrl: normalizedBaseUrl(baseUrl),
      model,
      apiMode,
      detail,
      tokenPresent: true,
    },
    request: {
      candidate_count: payload.candidates.length,
      layer_count: payload.layers.length,
      prompt_version: payload.prompt_version,
      schema: "uiir_semantics",
    },
    parsed,
    raw: rawText,
  };
}

function responsesBody(model, prompt, imageDataUrl, detail) {
  return {
    model,
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: prompt },
          { type: "input_image", image_url: imageDataUrl, detail },
        ],
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "uiir_semantics",
        strict: true,
        schema: openAISemanticsSchema,
      },
    },
  };
}

function chatCompletionsBody(model, prompt, imageDataUrl, detail) {
  return {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageDataUrl, detail: detail === "original" ? "high" : detail } },
        ],
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "uiir_semantics",
        strict: true,
        schema: openAISemanticsSchema,
      },
    },
  };
}

function candidateSummary(candidate) {
  return {
    id: candidate.id,
    bbox: normalizeBox(candidate.bbox),
    source: candidate.source || "",
    type_hint: candidate.type_hint || candidate.type || "Unknown",
    confidence: Number(candidate.confidence || 0),
    name: candidate.name || candidate.metadata?.name || "",
    text: candidate.text || "",
    style: typeof candidate.style === "string" ? candidate.style : JSON.stringify(candidate.style || {}),
    role: candidate.role || "",
    asset: candidate.asset || "",
    source_refs: candidate.source_refs || candidate.sourceRefs || [],
  };
}

function layerSummary(layer) {
  return {
    id: layer.id,
    name: layer.name || "",
    path: layer.path || "",
    kind: layer.kind || "",
    bbox: normalizeBox(layer.bbox),
    visible: layer.visible !== false,
    is_group: Boolean(layer.is_group || layer.isGroup),
    text: layer.text || "",
    style: layer.style || {},
  };
}

function responseText(responseJson) {
  if (responseJson.output_text) return String(responseJson.output_text);
  const outputChunks = [];
  for (const output of responseJson.output || []) {
    for (const content of output.content || []) {
      if (content.text) outputChunks.push(content.text);
    }
  }
  if (outputChunks.length) return outputChunks.join("");
  const message = responseJson.choices?.[0]?.message;
  if (typeof message?.content === "string") return message.content;
  if (Array.isArray(message?.content)) {
    return message.content.map((item) => item.text || "").join("");
  }
  throw new Error("Provider response did not contain text output.");
}

function summarizeSemanticItems(items) {
  const count = items.length || 0;
  if (!count) return { count: 0, roleFill: 0, layoutFill: 0, parentFill: 0 };
  return {
    count,
    roleFill: Math.round((items.filter((item) => item.role).length / count) * 100),
    layoutFill: Math.round((items.filter((item) => item.layout).length / count) * 100),
    parentFill: Math.round((items.filter((item) => item.parent_candidate_id).length / count) * 100),
  };
}

function friendlyProviderError(error) {
  const message = error?.message || String(error);
  if (message.includes("Failed to fetch")) {
    return "Request blocked or unreachable. Check base URL, HTTPS, provider CORS, and network access.";
  }
  return message;
}

function normalizedBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function exportProviderResult(result) {
  const payload = JSON.stringify({ ...result, provider: { ...result.provider, tokenPresent: Boolean(result.provider?.tokenPresent) } }, null, 2);
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "provider-smoke-result.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function createDemoData() {
  const width = 420;
  const height = 260;
  const imageDataUrl = createDemoImage(width, height);
  const candidates = [
    {
      id: "c1",
      bbox: { x: 50, y: 36, w: 320, h: 178 },
      source: "demo",
      type_hint: "Container",
      confidence: 0.82,
      name: "Dialog",
      source_refs: ["layer:1"],
    },
    {
      id: "c2",
      bbox: { x: 145, y: 155, w: 130, h: 44 },
      source: "demo",
      type_hint: "Button",
      confidence: 0.76,
      name: "confirm_btn_bg",
      source_refs: ["layer:3"],
    },
    {
      id: "c3",
      bbox: { x: 174, y: 168, w: 72, h: 18 },
      source: "demo",
      type_hint: "Text",
      confidence: 0.9,
      text: "Confirm",
      name: "confirm_text",
      parent_hint: "c2",
      source_refs: ["layer:4"],
    },
    {
      id: "c4",
      bbox: { x: 118, y: 70, w: 184, h: 28 },
      source: "demo",
      type_hint: "Text",
      confidence: 0.88,
      text: "Reward Unlocked",
      name: "title",
      parent_hint: "c1",
      source_refs: ["layer:2"],
    },
  ];
  const layers = [
    { id: "layer:1", name: "dialog panel", path: "dialog panel", kind: "group", bbox: candidates[0].bbox, visible: true, is_group: true, text: "", style: {} },
    { id: "layer:2", name: "title", path: "dialog panel/title", kind: "type", bbox: candidates[3].bbox, visible: true, is_group: false, text: "Reward Unlocked", style: { fontSize: 24 } },
    { id: "layer:3", name: "confirm_btn_bg", path: "dialog panel/confirm_btn_bg", kind: "pixel", bbox: candidates[1].bbox, visible: true, is_group: false, text: "", style: {} },
    { id: "layer:4", name: "confirm_text", path: "dialog panel/confirm_text", kind: "type", bbox: candidates[2].bbox, visible: true, is_group: false, text: "Confirm", style: { fontSize: 16 } },
  ];
  const uiir = {
    version: "0.1",
    source: "browser-demo",
    width,
    height,
    assetsRoot: "assets/",
    root: {
      id: "n1",
      type: "Screen",
      bbox: { x: 0, y: 0, w: width, h: height },
      confidence: 1,
      sourceRefs: ["document"],
      metadata: {},
      children: [
        {
          id: "n2",
          type: "Container",
          bbox: candidates[0].bbox,
          confidence: 0.82,
          sourceRefs: ["layer:1"],
          metadata: { psdPath: "dialog panel" },
          children: [
            { id: "n3", type: "Text", bbox: candidates[3].bbox, confidence: 0.88, sourceRefs: ["layer:2"], text: "Reward Unlocked", metadata: { psdPath: "dialog panel/title" }, children: [] },
            {
              id: "n4",
              type: "Button",
              bbox: { x: 145, y: 155, w: 130, h: 44 },
              confidence: 0.78,
              sourceRefs: ["layer:3", "layer:4"],
              role: "primary_action",
              metadata: { component: true, groupingReason: "demo_button_group" },
              children: [
                { id: "n5", type: "Background", bbox: candidates[1].bbox, confidence: 0.76, sourceRefs: ["layer:3"], metadata: { psdPath: "dialog panel/confirm_btn_bg" }, children: [] },
                { id: "n6", type: "Text", bbox: candidates[2].bbox, confidence: 0.9, sourceRefs: ["layer:4"], text: "Confirm", metadata: { psdPath: "dialog panel/confirm_text" }, children: [] },
              ],
            },
          ],
        },
      ],
    },
  };
  const xml = [
    '<UIIR version="0.1" source="browser-demo" width="420" height="260">',
    '  <Assets root="assets/" />',
    '  <Node id="n1" type="Screen" bbox="0,0,420,260" confidence="1.000" sourceRefs="document">',
    '    <Node id="n2" type="Container" bbox="50,36,320,178" confidence="0.820" sourceRefs="layer:1">',
    '      <Node id="n3" type="Text" bbox="118,70,184,28" confidence="0.880" sourceRefs="layer:2" text="Reward Unlocked" />',
    '      <Node id="n4" type="Button" bbox="145,155,130,44" confidence="0.780" sourceRefs="layer:3 layer:4" role="primary_action" groupingReason="demo_button_group">',
    '        <Node id="n5" type="Background" bbox="145,155,130,44" confidence="0.760" sourceRefs="layer:3" />',
    '        <Node id="n6" type="Text" bbox="174,168,72,18" confidence="0.900" sourceRefs="layer:4" text="Confirm" />',
    "      </Node>",
    "    </Node>",
    "  </Node>",
    "</UIIR>",
  ].join("\n");
  return { imageDataUrl, candidates, layers, uiir, xml };
}

function createDemoImage(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#eef1f5";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  roundRect(context, 50, 36, 320, 178, 12);
  context.fill();
  context.strokeStyle = "#d4dbe7";
  context.lineWidth = 2;
  roundRect(context, 50, 36, 320, 178, 12);
  context.stroke();
  context.fillStyle = "#172033";
  context.font = "700 24px system-ui, sans-serif";
  context.fillText("Reward Unlocked", 118, 92);
  context.fillStyle = "#64748b";
  context.font = "14px system-ui, sans-serif";
  context.fillText("PSD layers can become UIIR components.", 91, 126);
  context.fillStyle = "#14b8a6";
  roundRect(context, 145, 155, 130, 44, 8);
  context.fill();
  context.fillStyle = "#ffffff";
  context.font = "700 16px system-ui, sans-serif";
  context.fillText("Confirm", 177, 183);
  drawDemoBox(context, "c1", 50, 36, 320, 178, "#ef4444");
  drawDemoBox(context, "c2", 145, 155, 130, 44, "#14b8a6");
  drawDemoBox(context, "c3", 174, 168, 72, 18, "#2563eb");
  drawDemoBox(context, "c4", 118, 70, 184, 28, "#2563eb");
  return canvas.toDataURL("image/png");
}

function drawDemoBox(context, label, x, y, w, h, color) {
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.strokeRect(x, y, w, h);
  context.fillStyle = color;
  context.fillRect(x, Math.max(0, y - 18), 28, 18);
  context.fillStyle = "#ffffff";
  context.font = "700 11px system-ui, sans-serif";
  context.fillText(label, x + 4, Math.max(13, y - 5));
}

function roundRect(context, x, y, w, h, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + w - radius, y);
  context.quadraticCurveTo(x + w, y, x + w, y + radius);
  context.lineTo(x + w, y + h - radius);
  context.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  context.lineTo(x + radius, y + h);
  context.quadraticCurveTo(x, y + h, x, y + h - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

createRoot(document.getElementById("root")).render(<App />);
