# Repository Guidelines

## Project Structure & Module Organization
- `index.tsx`, `App.tsx`, and `index.html` form the Vite + React entry; UI sections live in `components/` (ProjectList, FloorPlanEditor, UserManagement, etc.).
- `services/` holds Zustand stores (`projectStore`, `userStore`, `authStore`) and API helpers (`geminiService`, `wallDetection`); share data models via `types.ts`, `constants.ts`, and `global.d.ts`.
- `data/` stores sample application profiles; `dist/` is generated build output (do not edit directly).
- `backend/` contains the FastAPI wall-detection service (`main.py`/`main_v2.py`) plus `requirements.txt` for Python dependencies.

## Build, Test, and Development Commands
- `npm install` — install front-end dependencies.
- `npm run dev` — start the Vite dev server (http://localhost:5173 by default).
- `npm run build` — produce a production bundle in `dist/`.
- `npm run preview` — serve the built bundle for smoke testing.
- Backend: `pip install -r backend/requirements.txt` then `uvicorn backend.main:app --reload` (swap to `backend.main_v2:app` if iterating there).

## Coding Style & Naming Conventions
- Use TypeScript with functional React components; prefer hooks for local state and Zustand stores for shared state.
- Two-space indentation, semicolons, and single quotes match existing files; keep imports grouped by module path depth.
- Components and store files are PascalCase; hooks, functions, and variables are camelCase. Keep shared types/interfaces in `types.ts` when they span modules.
- Favor small, focused components; collocate helper functions within the module unless reused elsewhere.

## Testing Guidelines
- No automated suite is configured yet; perform manual passes for project CRUD, floor-plan editing, user auth, and wall-detection preview.
- When adding tests, place `*.test.ts`/`*.test.tsx` beside the module or in a `__tests__` folder; wire an `npm test` script and keep fixtures small and deterministic.
- Run `npm run build` or `npm run preview` before submitting to catch obvious regressions.

## Commit & Pull Request Guidelines
- Commit messages follow the short, imperative style seen in history (`Enhance wall detection pipeline and previews`, `Fix wall drawing activation...`); keep summaries concise and scoped.
- PRs should cover what changed, why, and how to validate (commands/paths); link related issues; add screenshots/GIFs for UI updates; call out schema/API changes or new env vars.

## Security & Configuration Tips
- Keep API keys (e.g., `GEMINI_API_KEY` for the Gemini client) in `.env.local`; never commit secrets.
- The FastAPI backend currently allows all origins; tighten CORS and disable reload/debug flags for production deployments.
