# Understand Anything Dashboard Access Guide

This is the working runbook for Crop Yield Prediction.

## Project Graph Location

The dashboard reads:

`C:\Users\dhyan\Documents\DP Code's\Machine Learning\Projects\Crop-Yield-Prediction\.understand-anything\knowledge-graph.json`

## Recommended: Run From Crop-Yield-Prediction Folder

Open PowerShell in:

`C:\Users\dhyan\Documents\DP Code's\Machine Learning\Projects\Crop-Yield-Prediction`

Run:

```powershell
$env:GRAPH_DIR="$PWD"
$env:UNDERSTAND_ACCESS_TOKEN="crop-yield-kg-20260518"
pnpm --dir "C:\Users\dhyan\.understand-anything\Crop-Yield-Prediction\Understand-Anything\understand-anything-plugin" --filter @understand-anything/dashboard dev -- --host 127.0.0.1 --port 5173
```

Open:

`http://127.0.0.1:5173/?token=crop-yield-kg-20260518`

## Alternative: Run From Isolated Understand Anything Workspace

Open PowerShell in:

`C:\Users\dhyan\.understand-anything\Crop-Yield-Prediction\Understand-Anything\understand-anything-plugin`

Run:

```powershell
$env:GRAPH_DIR="C:\Users\dhyan\Documents\DP Code's\Machine Learning\Projects\Crop-Yield-Prediction"
$env:UNDERSTAND_ACCESS_TOKEN="crop-yield-kg-20260518"
pnpm --filter @understand-anything/dashboard dev -- --host 127.0.0.1 --port 5173
```

## Refresh Graph Then Preview

If code changed and you want a fresh graph first, run the Understand Anything refresh flow in Codex:

`/understand --full`

Then start dashboard with one of the command sets above.

## Stop Dashboard

Press `Ctrl + C` in the terminal running the dashboard command.

If the server is running in background:

`Stop-Process -Id <PID>`

## Troubleshooting

- `127.0.0.1 refused to connect`
  - Dashboard process is not running. Start it again.
- `Access Token Required` or `Forbidden: missing or invalid token`
  - URL token must match `UNDERSTAND_ACCESS_TOKEN`.
- `Cannot find module ... vite.js`
  - Reinstall isolated workspace deps:
  - `pnpm --dir "C:\Users\dhyan\.understand-anything\Crop-Yield-Prediction\Understand-Anything\understand-anything-plugin" install --frozen-lockfile --force`
