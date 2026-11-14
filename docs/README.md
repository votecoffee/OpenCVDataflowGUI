# Documentation map

## Plan, build and review an application

| Task | Start here |
| --- | --- |
| Install and make a first graph | [README](../README.md), [workflow guide](workflow-guide.md) |
| Work with a coding chat | [Assistant development](assistant-development.md), [application skill](../skills/opencv-dataflow-apps/SKILL.md) |
| Discover blocks and automate saved projects | [JSON/Python API](../skills/opencv-dataflow-apps/references/api.md) |
| Define requirements, inputs and acceptance checks | [Application plan template](application-plan-template.md) |
| Find runnable projects and image rights | [Example catalog](../examples/README.md) |
| Check implemented tools and remaining work | [Coverage](opencv-coverage.md), [checklist](vision-coverage-checklist.md), [roadmap](roadmap.md) |
| Look up controls and native bindings | [Block parameters](block-parameters.md), [binding inventory](opencv-bindings.tsv) |

## Runtime and extension contracts

| Topic | Reference |
| --- | --- |
| Blocks, persistence, errors, conversions and scheduling | [Architecture](architecture.md) |
| Reusable subgraphs and interfaces | [ToolBlocks](toolblocks.md) |
| Custom Python blocks | [Plugin skill](../skills/opencv-dataflow-plugins/SKILL.md), [plugin contract](../skills/opencv-dataflow-plugins/references/plugin-contract.md) |
| Dialogs, captures and worker training | [Plugin dialogs](plugin-dialogs.md) |
| Pixel formats and high-depth images | [Image formats](image-formats.md) |
| Regions, poses, pyramids and spatial provenance | [Workflow guide](workflow-guide.md), [architecture](architecture.md) |
| Tickets, branching, batching and joins | [Control flow](control-flow.md), [input coordination](input-coordination.md) |
| Acquisition queues and memory | [Input queues](input-queues.md), [sequence memory](sequence-memory.md) |
| Recipes and sealed trained assets | [Recipes](recipes.md) |
| Judgments, reasons and ticket histories | [Ticket results](ticket-results.md) |
| 3D/depth/point clouds | [3D vision](vision-3d.md) |
| ML/DNN and optional dependencies | [ML](ml.md) |
| scikit-image tools | [scikit-image](scikit-image.md) |

For narrower tools, search file names under `docs/` and the machine-readable block
catalog. The catalog reflects the running environment. Optional providers are
listed without loading them; resolving a particular type can require its package.
