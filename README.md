<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1WdBp4jB4Y3Pe2q3HYbGPecC36Gmo4iCq

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

### Backend wall detection (optional)

To experiment with wall detection locally:

1. Install backend dependencies: `pip install -r backend/requirements.txt`
2. Start FastAPI: `uvicorn backend.main:app --reload`
3. Call `/api/detect-walls-base64` with your floor plan image. Pass `{"detector": "ml"}` to enable the experimental HED-powered pipeline described in [backend/ML_DETECTION.md](backend/ML_DETECTION.md).
