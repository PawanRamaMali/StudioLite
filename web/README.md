# StudioLite web front-end

The React + Next.js UI for StudioLite. Talks to the FastAPI backend at `http://localhost:8000`.

## Dev

```bash
npm install
npm run dev            # http://localhost:3000
```

Point `NEXT_PUBLIC_API_URL` at a non-default backend if needed. The backend URL is baked at build time.

## Build

```bash
npm run build
npm start              # production server
```

## Layout

```
web/
  app/                 # App Router pages
  components/
    panels/            # One component per tool panel
    ui/                # Buttons, cards, badges
    Sidebar.tsx        # Navigation
  lib/
    api.ts             # Typed API client
    store.ts           # Zustand store
    utils.ts           # cn(), fmt helpers
```

Panels register in `app/page.tsx` and get a sidebar entry in `components/Sidebar.tsx`. Everything else follows the FastAPI backend's REST surface, which is documented at `/docs` on the backend.
