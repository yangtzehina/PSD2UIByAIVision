# PSD UIIR

本项目是一个本地 PSD UI 识别 MVP：输入真实 PSD/PSB，输出通用 `uiir.xml`、`uiir.json`、候选框、合成图和图层素材目录。后续 Unity 插件可以消费 XML 生成 UGUI Prefab。

核心原则：

- 坐标和素材来自本地 PSD 解析，不让模型猜像素。
- UIIR 会在 PSD 图层树上提升运行时组件，例如把 `按钮背景 + 按钮文字` 合成一个 `Button`，原始图层仍作为子节点保留。
- 本地视觉候选框补足不规范命名或简化图层。
- GPT-5.5 默认不启用；启用后可以先提出视觉漏检候选，再做语义归类、角色判断和层级修正。
- 云端请求只发送合成图、编号候选框和必要元数据，不上传 PSD 原文件。

## 安装

```bash
python3 -m pip install -e .
```

如果要启用 OpenCV 视觉候选框：

```bash
python3 -m pip install -e ".[vision]"
```

如果要启用可选 OCR 候选框，需要本机已安装 Tesseract，再安装 Python 适配器：

```bash
python3 -m pip install -e ".[ocr]"
```

## CLI

```bash
uiir extract path/to/input.psd --out out/input
```

输出结构：

```text
out/input/
  composite.png
  candidates.json
  layer_metadata.json
  uiir.json
  uiir.xml
  overlay.png
  assets/layers/*.png
```

启用 GPT-5.5 语义增强：

```bash
export OPENAI_API_KEY=...
uiir extract path/to/input.psd --out out/input --use-openai --model gpt-5.5
```

`--use-openai` 会发送 `overlay.png`、候选框摘要、图层元数据摘要；不会发送 PSD 原文件。

启用 GPT-5.5 真视觉提案：

```bash
uiir extract path/to/input.psd --out out/input --use-openai \
  --model gpt-5.5 \
  --openai-vision-proposals
```

视觉提案阶段会额外发送 `composite.png` 和候选框 overlay，让模型提出漏检元素、拆分建议和组件合并建议。模型返回的 bbox 只进入 `VisionProposal` 审计流：本地程序会 clamp 到画布、过滤异常面积、和已有候选做 IoU 去重，并把新增候选的 confidence 上限固定为 `0.55`。GPT 不直接写 XML，也不会覆盖 PSD 图层坐标。

默认策略是 precision-first：

```bash
uiir extract path/to/input.psd --out out/input --use-openai \
  --openai-vision-proposals \
  --vision-policy strict \
  --document-kind auto
```

`--vision-policy` 支持 `audit|strict|balanced`：`audit` 只审计不改 candidates，`strict` 只接受和本地候选有重叠的提案或安全的组件关系，`balanced` 保留较积极的实验策略。`--document-kind auto|screen|asset_sheet` 会把 `ui.psd` 这类控件素材表分流成 `asset_sheet`，避免把 sprite sheet 拆分结果直接塞进运行时屏幕树。输出目录会新增：

```text
vision_request_summary.json
vision_proposals.json
vision_accepted.json
vision_quarantined.json
vision_rejected.json
relations.json
semantic_patches.json
vision_tiles/*.png
```

也可以切到第三方 OpenAI-compatible API 网关或模型供应商。项目只需要一个兼容 Responses API、图像输入和 Structured Outputs 的 endpoint：

```bash
export THIRD_PARTY_API_KEY=...
uiir extract path/to/input.psd --out out/input --use-openai \
  --provider-name third-party \
  --api-key-env THIRD_PARTY_API_KEY \
  --base-url https://gateway.example.com/v1 \
  --api-mode responses \
  --model provider/model-id
```

`--api-key-env` 只记录环境变量名，不会把密钥写入 `openai_request_summary.json`。如果不传 `--base-url`，会尝试读取 `UIIR_OPENAI_BASE_URL` 或 `OPENAI_BASE_URL`。如果供应商不支持图像输入或 JSON Schema 结构化输出，这条语义增强链路仍会失败，需要为该供应商单独做 adapter。

第三方如果没有 Responses API，但兼容 OpenAI Chat Completions 的图像输入和 JSON Schema 输出，可以先切换：

```bash
uiir extract path/to/input.psd --out out/input --use-openai \
  --provider-name third-party \
  --api-key-env THIRD_PARTY_API_KEY \
  --base-url https://gateway.example.com/v1 \
  --api-mode chat-completions \
  --model provider/model-id
```

启用本地 OCR 候选框：

```bash
uiir extract path/to/input.psd --out out/input --ocr
```

应用人工修正：

```bash
uiir extract path/to/input.psd --out out/input --corrections corrections.json
```

查看 JSON Schema：

```bash
uiir schema
```

## 测试集、批处理与评测

下载公开 PSD fixture。默认 `parser-smoke` 来自 psd-tools、ag-psd、webtoon/psd 和 Baum2 的公开测试样本：

```bash
uiir fixtures list
uiir fixtures download --set parser-smoke --out fixtures/parser-smoke --limit 20
```

下载小型游戏 UI PSD fixture。默认 `game-ui-smoke` 来自 OpenGameArt 的小型 PSD/ZIP 资源，用于验证真实游戏 UI 场景：

```bash
uiir fixtures download --set game-ui-smoke --out fixtures/game-ui-smoke
uiir batch fixtures/game-ui-smoke --out out/game-ui-smoke --use-openai false
uiir evaluate out/game-ui-smoke --report out/game-ui-smoke/metrics.json
```

`game-ui-smoke` 目前包含 RPG Game UI、User Interface、2D UI Kit、Golden UI 和 UI Elements。`fixtures.manifest.json` 会记录 `license`、`source_url`、`attribution`、`sha256` 和 `expected`，下载的 PSD/ZIP 只作为本地研究测试数据，不提交进仓库。

如果遇到 GitHub API rate limit，可以设置 `GITHUB_TOKEN` 后重试。单个 fixture 源失败时，下载器会继续处理其他源，并把 warning 写入 `fixtures.manifest.json`。

批量提取：

```bash
uiir batch fixtures/parser-smoke --out out/parser-smoke --use-openai false
```

批处理会在输出目录写入 `report.json`，每个 PSD 样本各自生成一套 `composite.png`、`overlay.png`、`uiir.json`、`uiir.xml` 和素材目录。fixture manifest 中 `expected: "skip"` 的样本如果失败，会记为 skipped 而不是 failed。

评测输出：

```bash
uiir evaluate out/parser-smoke --report out/parser-smoke/metrics.json
uiir evaluate out/parser-smoke --golden goldens --report out/parser-smoke/metrics.json
```

评测会检查 UIIR JSON 基本 schema、生成 `preview.png`，并和 `composite.png` 计算像素相似度。提供 golden UIIR 时，还会输出类型 F1、bbox 平均 IoU 和树结构距离近似值。

如果页面被自动识别为 `asset_sheet`，评测仍会生成 `replay_preview.png` 和 `diagnostic_overlay.png`，但不会把屏幕像素回放相似度作为主指标。素材表更关注元素拆分、类型、bbox、proposal 和 relation 指标。

## 基础回归与 OpenAI 语义回归

基础回归不依赖网络模型，适合每次代码改动后运行：

```bash
python3 -m unittest discover -s tests
cd inspector && npm run build
uiir batch fixtures/game-ui-smoke --out out/game-ui-smoke --use-openai false
uiir evaluate out/game-ui-smoke --report out/game-ui-smoke/metrics.json
```

OpenAI 语义回归是独立 smoke，不进入普通 CI。它默认只跑 2 个 PSD，并比较本地 baseline 与 GPT-5.5 语义增强后的差异：

```bash
export OPENAI_API_KEY=...
uiir compare-openai fixtures/game-ui-smoke --out out/openai-smoke --model gpt-5.5 --limit 2
uiir review-run out/openai-smoke
```

没有 `OPENAI_API_KEY` 时，`compare-openai` 会写入 `comparison.json` 并标记为 skipped，不影响基础回归。OpenAI 分支只发送 `overlay.png`、候选框摘要和图层元数据摘要，不上传 PSD 原文件；模型只改 `type`、`role`、`text`、`style`、`layout`、`parent_hint` 等语义字段，不生成坐标或 XML。

如果要测试真视觉识别，把视觉提案阶段打开。推荐仍然只跑 1-2 个 PSD，先确认 provider 支持图片输入和 JSON Schema：

```bash
uiir compare-openai fixtures/openai-smoke --out out/vision-smoke \
  --provider-name third-party \
  --api-key-env UIIR_PROVIDER_API_KEY \
  --base-url "$UIIR_OPENAI_BASE_URL" \
  --api-mode chat-completions \
  --model gpt-5.5 \
  --limit 2 \
  --openai-vision-proposals \
  --vision-policy strict \
  --document-kind auto
uiir review-run out/vision-smoke
```

这条链路会先生成 baseline，再在 OpenAI 分支执行：本地候选 -> GPT 视觉提案 -> 本地准入/隔离/融合 -> 重新绘制 overlay -> GPT 语义补丁 -> UIIR/XML。`comparison.json` 会额外统计 `vision.created/merged/quarantined/rejected`、语义补丁 accepted/rejected、无效 parent hint、Unknown 变化、document kind、vision policy 和 render pixel similarity 变化。

持续自我迭代可以跑固定策略矩阵，生成本地榜单：

```bash
uiir iterate-openai fixtures/openai-smoke --out out/runs/provider-vision \
  --provider-name third-party \
  --api-key-env UIIR_PROVIDER_API_KEY \
  --base-url "$UIIR_OPENAI_BASE_URL" \
  --api-mode chat-completions \
  --model gpt-5.5 \
  --limit 2
```

`iterate-openai` 会依次跑 `semantic_v2 + audit/strict/balanced`，输出 `leaderboard.json` 和 `leaderboard.md`，用 schema、pixel gate、invalid parent、Unknown、type changes、semantic fill 等指标排序。

## PSD-Aware Graph-of-Mark

在 SoM 编号 overlay 之外，下一层可解释信号是本地关系图：把 PSD parent、几何包含、同行同列、同尺寸、重复模式、文字压在图片上、候选组件组合等关系写成 `ui_graph.json`，并渲染为 `graph_overlay.png`。GPT 分支可以读取这张图来做关系确认和局部审查，但仍不能直接写 XML 或覆盖 PSD 坐标。

离线生成单个样本的关系图：

```bash
uiir graph build out/game-ui-smoke/interface --out out/game-ui-smoke/interface
```

渲染审查会对比 `composite.png`、`replay_preview.png` 和 `diagnostic_overlay.png`，输出 `render_diff.png` 与 `render_review.json`。发现的 missing/extra/misclassified 区域只进入审计和 quarantine，不会直接污染 UIIR：

```bash
uiir review-render out/game-ui-smoke/interface --out out/game-ui-smoke/interface
```

OpenAI 对比和迭代可以打开关系图、渲染审查和主动学习队列：

```bash
uiir compare-openai fixtures/openai-smoke --out out/graph-vision-smoke \
  --provider-name third-party \
  --api-key-env UIIR_PROVIDER_API_KEY \
  --base-url "$UIIR_OPENAI_BASE_URL" \
  --api-mode chat-completions \
  --model gpt-5.5 \
  --limit 2 \
  --openai-vision-proposals \
  --vision-policy strict \
  --graph-overlay \
  --render-review \
  --focus-tiles \
  --parser-fidelity \
  --curation-report
```

如果 `--prompts relation_v1` 或 `--prompt-version relation_v1`，OpenAI 分支会在普通语义增强后额外执行 Graph-of-Mark relation review。模型输出只会写入 `relation_patches.json`、`relation_quarantined.json` 和候选 metadata，不会改写 `uiir.xml` 或已有 bbox。

渲染审查后的高风险区域可以裁成局部 tile，方便后续 tile 级视觉审查：

```bash
uiir focus build out/game-ui-smoke/interface --out out/game-ui-smoke/interface
```

主动学习采样会把 quarantine 多、prompt/policy 分歧大、golden 指标低、关系图复杂、render issue 多的样本排到前面，减少人工确认成本：

```bash
uiir curate out/graph-vision-smoke --golden goldens/local --out out/curation
```

新增产物包括 `ui_graph.json`、`graph_overlay.png`、`render_review.json`、`render_diff.png`、`focus_tiles.json`、`relation_patches.json`、`parser_fidelity.json`、`curation_queue.json` 和 `curation_queue.md`。`leaderboard` 现在也会记录 `relation_precision`、`relation_recall`、`component_group_f1`、relation patch 数、focus tile 数、`render_review_issue_count` 和 `curation_value_score`。

## 并行探索命令

本地视觉 adapter 对照不会默认下载任何模型权重。`uied` 是轻量 OpenCV/UIED-style baseline；OmniParser、SAM、PaddleOCR 只写 skipped manifest，方便后续本地接权重：

```bash
uiir adapter list
uiir adapter run out/game-ui-smoke/interface --adapter uied --out out/adapter-uied/interface
uiir adapter run out/game-ui-smoke/interface --adapter omniparser --out out/adapter-omniparser/interface
```

Rico importer 只导入本地已有 screenshot + view hierarchy，不下载数据集，输出临时 UIIR golden：

```bash
uiir dataset rico-import /path/to/rico-local --out out/rico-import --limit 20
uiir evaluate out/rico-import/goldens --golden out/rico-import/goldens --report out/rico-import/metrics.json
```

Parser fidelity 报告用于衡量 psd-tools 提取覆盖率，并可选探测 PhotoshopAPI 是否本地可用：

```bash
uiir fidelity out/game-ui-smoke/interface --out out/game-ui-smoke/interface
uiir fidelity out/game-ui-smoke/interface --probe-photoshopapi
```

## Quarantine-to-Golden 闭环

`strict` 策略会把无本地几何证据的新视觉提案写入 `vision_quarantined.json`。人工确认后，可以把这些隔离提案沉淀成本地 golden：

```bash
uiir golden build \
  --psd fixtures/openai-smoke/opengameart-rpg-game-ui/interface.psd \
  --run out/vision-smoke/openai/interface \
  --decisions goldens/local/interface/golden_decisions.json \
  --out goldens/local/interface
```

`golden_decisions.json` 由 Inspector 导出，格式如下：

```json
{
  "version": "0.1",
  "decisions": [
    {
      "decision": "accept",
      "target_kind": "proposal",
      "target_id": "p1",
      "type": "Button",
      "role": "primary_action"
    }
  ]
}
```

支持的 `decision` 是 `accept|reject|edit|ignore`，支持的 `target_kind` 是 `proposal|candidate|node|relation`。接受的 quarantined proposal 会变成 `source="human-accepted-vision-proposal"`、`confidence=0.90` 的人工候选，并保留 `sourceRefs=["openai-vision:<proposal_id>"]`。`goldens/local/` 已在 `.gitignore` 中忽略，适合存本地人工标注和商业 PSD 样本。

有 golden 后再跑自我迭代：

```bash
uiir iterate-openai fixtures/openai-smoke --out out/runs/provider-vision \
  --provider-name third-party \
  --api-key-env UIIR_PROVIDER_API_KEY \
  --base-url "$UIIR_OPENAI_BASE_URL" \
  --api-mode chat-completions \
  --model gpt-5.5 \
  --limit 2 \
  --golden goldens/local \
  --prompts semantic_v2,semantic_v3 \
  --policies audit,strict,balanced
```

每个子 run 会写 `experiment_manifest.json`，只记录 `api_key_env`、`api_key_present` 和 `base_url_present`，不记录 token 或真实 base URL。`leaderboard` 会综合 schema、pixel gate、invalid parent、Unknown、type changes、golden F1、proposal recall、relation precision/recall/F1、component group F1、render review issues 和 quarantine usefulness 排序。

第三方 provider 的 smoke 方式相同，只是把 key 环境变量和 base URL 换掉：

```bash
export THIRD_PARTY_API_KEY=...
uiir compare-openai fixtures/game-ui-smoke --out out/provider-smoke \
  --provider-name third-party \
  --api-key-env THIRD_PARTY_API_KEY \
  --base-url https://gateway.example.com/v1 \
  --api-mode responses \
  --model provider/model-id \
  --limit 2
```

`comparison.json` 会记录模型、`prompt_version`、schema 结果、像素相似度变化、Unknown 节点变化、role/layout/parent_hint 填充率变化和类型变化。`review-run` 会生成 `review.json` 与 `review.md`，用于持续发现退化样本。当前语义 prompt 版本记录在 `prompt_versions/semantic_v2.json`，后续迭代可以继续新增 `semantic_v3.json` 等版本横向比较。

## 本地检查器

检查器是纯前端 React/Vite 应用，支持手动加载 CLI 输出的 `composite.png`、`uiir.json`、`candidates.json`、`uiir.xml` 和 `corrections.json`。选中候选框或树节点后，可以修改类型、角色、文本、样式、布局、父级、bbox 和 ignored 标记，并导出新的 `corrections.json`。

```bash
cd inspector
npm install
npm run dev
```

打开 Vite 提供的本地 URL，在页面中选择 CLI 输出文件即可检查 UI 树和候选框。

## GitHub Pages 在线检查器

仓库包含 GitHub Pages 自动部署 workflow。推送到 `main` 后，Inspector 会构建为静态页面：

[https://yangtzehina.github.io/PSD2UIByAIVision/](https://yangtzehina.github.io/PSD2UIByAIVision/)

在线页面可以直接上传本地 CLI 输出产物进行检查：

- `overlay.png` 或 `composite.png`
- `candidates.json`
- `layer_metadata.json`
- 可选 `uiir.json`、`uiir.xml`、`comparison.json`
- 可选 `vision_quarantined.json`、`vision_rejected.json`、`semantic_patches.json`

也可以点击 `Load demo sample` 直接加载一个内置小样例，再用 `Provider Smoke` 面板填写第三方 OpenAI-compatible `Base URL`、`Token`、`Model` 和 `API mode`，从浏览器直接调用 provider 做语义测试。Token 只存在当前浏览器标签页内，不写入仓库、不写入导出的结果文件。因为这是浏览器直连，第三方接口必须允许 CORS；如果 provider 不允许网页跨域请求，需要改用本地 CLI 的 `uiir compare-openai`。

检查器的 Review Filters 可以切换 `All / Local / GPT Accepted / GPT Quarantined / GPT Rejected / Semantic Patch`。选中节点时会显示视觉提案原因、隔离/拒绝原因、相关 candidate id 和语义补丁审计。`Golden Decision` 面板支持 Accept / Reject / Edit / Ignore，并导出 `golden_decisions.json`；原 `corrections.json` 导出仍保留用于快速修正单次输出。

如果 GitHub Pages 没有自动出现，请在仓库 `Settings -> Pages` 中把 source 设为 `GitHub Actions`，然后重新运行 `Deploy Inspector to GitHub Pages` workflow。

人工修正格式：

```json
{
  "version": "0.1",
  "corrections": [
    {
      "candidate_id": "c12",
      "type": "Button",
      "bbox": { "x": 820, "y": 760, "w": 280, "h": 72 },
      "text": "确定",
      "parent_id": "c4"
    }
  ]
}
```

## XML 格式

```xml
<UIIR version="0.1" source="screen.psd" width="1920" height="1080">
  <Assets root="assets/" />
  <Node id="n1" type="Screen" bbox="0,0,1920,1080" confidence="1.000" sourceRefs="document">
    <Node id="n2" type="Button" bbox="820,760,280,72" confidence="0.850" sourceRefs="layer:12" text="确定" />
  </Node>
</UIIR>
```

节点类型固定为：

`Screen, Container, Image, Icon, Text, Button, Input, Toggle, Slider, ScrollView, List, Grid, Background, Unknown`

每个节点都包含 `bbox`、`confidence`、`sourceRefs`，可选 `role`、`text`、`style`、`layout`、`asset`、`interaction`。

## PSD 树到 UIIR 组件树

PSD 和 UIIR 都是树，但含义不同：PSD 是设计图层树，UIIR 是运行时组件树。提取流程会先保留 PSD 的父子关系，再做组件化提升：

```text
PSD:
背景
  按钮背景
  按钮文字
  文本

UIIR:
Background
  Button
    Background
    Text
  Text
```

组件节点的 `bbox` 来自子节点 union，组件自身可以没有 `asset`；原始 PSD 图层节点仍保留在 children 中。`uiir.json` 的节点 `metadata` 会记录 `psdParentId`、`psdPath`、`psdDepth`、`groupingReason` 等信息，`sourceRefs` 可回溯到原始 `layer:*`。

Inspector 的树面板可以切换 `UIIR` 和 `PSD`，用于对照运行时组件树和原始图层树。选中节点时会显示 source refs、PSD path 和 grouping reason。
