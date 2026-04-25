import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Boxes, FileJson, FileText, Image as ImageIcon, Layers, RotateCcw, Save, Upload } from "lucide-react";
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

function App() {
  const [imageUrl, setImageUrl] = useState("");
  const [uiir, setUiir] = useState(null);
  const [baselineUiir, setBaselineUiir] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [xml, setXml] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [corrections, setCorrections] = useState([]);
  const [treeMode, setTreeMode] = useState("uiir");
  const nodes = useMemo(() => flattenNodes(uiir?.root), [uiir]);
  const psdTree = useMemo(() => buildPsdTree(nodes), [nodes]);
  const baselineNodes = useMemo(() => flattenNodes(baselineUiir?.root), [baselineUiir]);
  const diffByNodeId = useMemo(() => buildNodeDiffs(baselineNodes, nodes), [baselineNodes, nodes]);
  const usingCandidates = candidates.length > 0;
  const boxes = usingCandidates ? candidates : nodes;
  const selectedCandidate = candidates.find((item) => item.id === selectedId) || null;
  const selectedNode = nodes.find((item) => item.id === selectedId) || null;
  const selected = selectedCandidate || selectedNode;
  const selectedUsesCandidate = Boolean(selectedCandidate);

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

        <FileInput icon={<ImageIcon size={18} />} label="Composite PNG" accept="image/*" onFile={readImage(setImageUrl)} />
        <FileInput icon={<FileJson size={18} />} label="uiir.json" accept=".json,application/json" onFile={readJson(setUiir)} />
        <FileInput icon={<FileJson size={18} />} label="baseline uiir.json" accept=".json,application/json" onFile={readJson(setBaselineUiir)} />
        <FileInput icon={<FileJson size={18} />} label="comparison.json" accept=".json,application/json" onFile={readJson(setComparison)} />
        <FileInput icon={<Boxes size={18} />} label="candidates.json" accept=".json,application/json" onFile={readJson(setCandidates)} />
        <FileInput icon={<FileText size={18} />} label="uiir.xml" accept=".xml,text/xml" onFile={readText(setXml)} />
        <FileInput icon={<FileJson size={18} />} label="corrections.json" accept=".json,application/json" onFile={readCorrections(setCorrections)} />

        <section className="stats">
          <Metric label="Canvas" value={uiir ? `${uiir.width} x ${uiir.height}` : "Waiting"} />
          <Metric label="Nodes" value={nodes.length || "0"} />
          <Metric label="Boxes" value={boxes.length || "0"} />
          <Metric label="Edits" value={corrections.length || "0"} />
          <Metric label="Diffs" value={Object.keys(diffByNodeId).length || "0"} />
        </section>

        <DiffPanel comparison={comparison} diffCount={Object.keys(diffByNodeId).length} />

        <CorrectionEditor
          selected={selected}
          usingCandidates={selectedUsesCandidate}
          correction={selected ? getCorrection(corrections, selected, selectedUsesCandidate) : null}
          onChange={upsertCorrection}
          onReset={resetSelected}
          onExport={() => exportCorrections(corrections)}
          correctionCount={corrections.length}
        />
      </aside>

      <section className="workspace">
        <div className="stage">
          {imageUrl ? (
            <div className="imageWrap">
              <img src={imageUrl} alt="PSD composite" />
              {boxes.map((box, index) => {
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
              })}
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

function DiffPanel({ comparison, diffCount }) {
  if (!comparison && !diffCount) return null;
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

function NodeDetails({ node }) {
  const metadata = node.metadata || {};
  const refs = node.sourceRefs || node.source_refs || [];
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

function getCorrection(corrections, target, usingCandidates) {
  if (!target) return null;
  const key = usingCandidates ? "candidate_id" : "node_id";
  return corrections.find((item) => item[key] === target.id) || null;
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

function readImage(setter) {
  return (file) => setter(URL.createObjectURL(file));
}

function readJson(setter) {
  return async (file) => setter(JSON.parse(await file.text()));
}

function readCorrections(setter) {
  return async (file) => {
    const parsed = JSON.parse(await file.text());
    setter(Array.isArray(parsed) ? parsed : parsed.corrections || []);
  };
}

function readText(setter) {
  return async (file) => setter(await file.text());
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

createRoot(document.getElementById("root")).render(<App />);
