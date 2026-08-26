# RallyAI dashboard

The React SPA behind the Overview / Journal / Statistics / Film Review tabs. See the repo root [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md) for the product overview and architecture; this is just the build.

```
npm install
npm run build   # -> dist/, served by webapp/app.py at "/"
npm run dev     # Vite dev server on :5173, proxying /api and /media to Flask on :5000
```

Stack: Vite + React + TypeScript + Tailwind CSS v4, Recharts for charts, `motion` for animation.
