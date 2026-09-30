# GitHub portfolio release checklist

This checklist covers the two presentation steps that cannot be applied by repository code.

## Repository social preview

1. Open the GitHub repository and select **Settings → General**.
2. In **Social preview**, upload
   [`docs/assets/github-social-preview.png`](assets/github-social-preview.png).
3. Confirm that the preview is 1280×640 and contains no browser chrome or personal account UI.

The same checked-in image is used by the portfolio's Open Graph and Twitter Card metadata. Rebuild
it with `python scripts/build_social_preview.py`.

## About panel

Use this description:

> Evidence-first multimodal post-training, agent evaluation, distributed correctness, and inference systems—with auditable code, results, and claim boundaries.

Set the website to `https://powermachine.github.io/foundation-model-lab/` and suggested topics to:
`multimodal`, `llm`, `lora`, `agent-evaluation`, `ml-systems`, `distributed-training`,
`inference`, and `reproducible-research`.

## Optional 45-second walkthrough

Record the deployed site at 1920×1080 without browser bookmarks, personal profiles, or unrelated
tabs. A concise shot list is:

| Time | View | Reviewer signal |
|---:|---|---|
| 0–6 s | Landing hero and headline metrics | Scope and positioning |
| 6–16 s | Multimodal post-training case study | Actual GPU evidence and claim boundary |
| 16–26 s | Agent evaluation case study | Adversarial controls, retry, and resume |
| 26–34 s | Distributed correctness case study | Semantic invariants and exact continuation |
| 34–41 s | Technical report → JSON → source links | End-to-end auditability |
| 41–45 s | Evidence ledger and boundaries | Research judgment rather than demo inflation |

Prefer MP4/WebM hosted in a GitHub Release or a professional video host. Keep the static social
preview in the README instead of committing a large animated GIF.
