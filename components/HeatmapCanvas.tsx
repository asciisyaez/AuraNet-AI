import React, { useRef, useEffect } from 'react';
import { AccessPoint, Wall } from '../types';

interface HeatmapCanvasProps {
  aps: AccessPoint[];
  walls: Wall[];
  width: number;
  height: number;
  show: boolean;
}

const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({ aps, walls, width, height, show }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !show) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Fill background with "weak signal" color (blue-ish/transparent)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(0, 0, width, height);

    // Create a temporary canvas for blending signals
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');

    if (!tempCtx) return;

    // Draw each AP signal
    aps.forEach(ap => {
      // Radius depends on power. 20dBm ~= strong coverage for 150px approx in this scale
      const radius = ap.power * 15; 
      
      const gradient = tempCtx.createRadialGradient(ap.x, ap.y, 10, ap.x, ap.y, radius);
      
      // Heatmap colors: Red (Strong) -> Yellow -> Green -> Blue (Weak)
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.8)');   // Red -50dBm
      gradient.addColorStop(0.4, 'rgba(245, 158, 11, 0.6)'); // Orange
      gradient.addColorStop(0.7, 'rgba(34, 197, 94, 0.4)');  // Green -65dBm
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');   // Fade out

      tempCtx.beginPath();
      tempCtx.fillStyle = gradient;
      tempCtx.arc(ap.x, ap.y, radius, 0, Math.PI * 2);
      tempCtx.fill();
    });

    // Simple wall attenuation simulation (visual only)
    // We just draw semi-transparent lines over the heatmap to show "blocking"
    tempCtx.globalCompositeOperation = 'destination-out';
    tempCtx.lineCap = 'round';

    walls.forEach(wall => {
      tempCtx.beginPath();
      tempCtx.moveTo(wall.x1, wall.y1);
      tempCtx.lineTo(wall.x2, wall.y2);
      tempCtx.lineWidth = Math.max(wall.thickness, 6);
      tempCtx.globalAlpha = Math.min(0.9, 0.25 + wall.attenuation / 18);
      tempCtx.stroke();
    });
    tempCtx.globalAlpha = 1;

    // Reset composite operation to draw the result to main canvas
    tempCtx.globalCompositeOperation = 'source-over';

    // Copy temp to main
    ctx.drawImage(tempCanvas, 0, 0);

  }, [aps, walls, width, height, show]);

  if (!show) return null;

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute top-0 left-0 pointer-events-none z-[2] opacity-70"
    />
  );
};

export default HeatmapCanvas;
