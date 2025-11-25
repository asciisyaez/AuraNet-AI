import { GoogleGenAI } from "@google/genai";
import { AccessPoint, Wall } from "../types";

const apiKey = process.env.API_KEY || '';
const ai = new GoogleGenAI({ apiKey });

export const getOptimizationSuggestions = async (aps: AccessPoint[], walls: Wall[]) => {
  if (!apiKey) {
    return "API Key not configured. Please check environment settings.";
  }

  const systemInstruction = `You are a world-class Wi-Fi Network Engineer expert. 
  Analyze the provided Access Point (AP) placement and Wall configuration. 
  Provide specific, technical, and actionable advice to improve coverage, reduce interference, and optimize channel planning.
  Keep it concise (max 3 bullet points).`;

  const prompt = `
    Current Configuration:
    Access Points: ${JSON.stringify(aps.map(ap => ({ id: ap.id, x: ap.x, y: ap.y, model: ap.model, power: ap.power, channel: ap.channel, band: ap.band, height: ap.height, azimuth: ap.azimuth, tilt: ap.tilt, antennaGain: ap.antennaGain })))}
    Walls: ${JSON.stringify(walls.map(w => ({ material: w.material, length: Math.sqrt(Math.pow(w.x2-w.x1, 2) + Math.pow(w.y2-w.y1, 2)) })))}
    
    Please suggest optimizations.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        systemInstruction: systemInstruction,
      }
    });

    return response.text;
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Failed to generate optimization insights. Please try again later.";
  }
};
