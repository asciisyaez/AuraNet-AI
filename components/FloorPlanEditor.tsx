import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AccessPoint, Wall, FloorPlan, ScaleReference } from '../types';
import { HARDWARE_TOOLS, ENV_TOOLS } from '../constants';
import HeatmapCanvas from './HeatmapCanvas';
import { Wifi, Router, Square, Trash2, Edit3, Loader2, Info, Image as ImageIcon, Eye, EyeOff, Ruler, Move, ZoomIn, ZoomOut, Maximize, X, Wand2 } from 'lucide-react';
import { getOptimizationSuggestions } from '../services/geminiService';
import { ANTENNA_PATTERNS, AP_LIBRARY, CHANNEL_OPTIONS } from '../data/apLibrary';
import { useProjectStore } from '../services/projectStore';

const COVERAGE_TARGET_DBM = -65;
const DEFAULT_METERS_PER_PIXEL = 0.05;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max));

const generateApId = () => `AP-${Date.now().toString().slice(-4)}`;

const segmentsIntersect = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number, dx: number, dy: number) => {
  const det = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (det === 0) return false;
  const lambda = ((dy - cy) * (dx - ax) + (cx - dx) * (dy - ay)) / det;
  const gamma = ((ay - cy) * (dx - ax) + (cx - ax) * (dy - ay)) / det;
  return lambda > 0 && lambda < 1 && gamma > 0 && gamma < 1;
};

const calculateSignal = (ap: AccessPoint, x: number, y: number, walls: Wall[], metersPerPixel: number) => {
  const horizontalDistance = Math.hypot(ap.x - x, ap.y - y) * metersPerPixel;
  const distance = Math.hypot(horizontalDistance, ap.height);
  const pathLoss = 20 * Math.log10(distance + 1);
  const wallLoss = walls.reduce((loss, wall) => {
    const intersects = segmentsIntersect(ap.x, ap.y, x, y, wall.x1, wall.y1, wall.x2, wall.y2);
    return loss + (intersects ? wall.attenuation : 0);
  }, 0);
  return ap.power + ap.antennaGain - pathLoss - wallLoss;
};

const evaluateCoverage = (
  aps: AccessPoint[],
  walls: Wall[],
  target: number = COVERAGE_TARGET_DBM,
  metersPerPixel: number = DEFAULT_METERS_PER_PIXEL,
  width: number,
  height: number
) => {
  const desiredCoverage = 95;
  const gridSpacing = 60;
  const gridPoints: { x: number; y: number }[] = [];
  for (let x = gridSpacing; x < width; x += gridSpacing) {
    for (let y = gridSpacing; y < height; y += gridSpacing) {
      gridPoints.push({ x, y });
    }
  }

  let covered = 0;
  let bestSignals: number[] = [];

  gridPoints.forEach(point => {
    const bestSignal = Math.max(
      ...aps.map(ap => calculateSignal(ap, point.x, point.y, walls, metersPerPixel))
    );
    bestSignals.push(bestSignal);
    if (bestSignal >= target) covered += 1;
  });

  const coveragePercent = gridPoints.length > 0 ? (covered / gridPoints.length) * 100 : 0;
  const averageSignal = bestSignals.length > 0 ? bestSignals.reduce((a, b) => a + b, 0) / bestSignals.length : -100;
  const imbalancePenalty = aps.reduce((penalty, ap, idx) => {
    return aps.slice(idx + 1).reduce((inner, other) => {
      const distance = Math.hypot(ap.x - other.x, ap.y - other.y);
      return inner + (distance < 120 ? (120 - distance) * 0.5 : 0);
    }, penalty);
  }, 0);

  const coverageGap = Math.max(0, desiredCoverage - coveragePercent);
  const apPenalty = aps.length * 2;
  const signalPenalty = Math.max(0, target - averageSignal) * 0.5;

  const score = coverageGap * 6 + signalPenalty + imbalancePenalty + apPenalty;

  return { coveragePercent, averageSignal, score, apCount: aps.length };
};

const createRandomAp = (width: number, height: number) => {
  const template = AP_LIBRARY[0];
  return {
    id: generateApId(),
    x: clamp(Math.random() * width, 40, width - 40),
    y: clamp(Math.random() * height, 40, height - 40),
    model: template.name,
    band: template.bands[0],
    power: template.defaultPower,
    channel: 'Auto',
    height: template.defaultHeight,
    azimuth: template.defaultAzimuth ?? 0,
    tilt: template.defaultTilt ?? 0,
    antennaGain: template.antennaGain,
    antennaPatternFile: template.patternFile,
    color: '#3b82f6'
  } as AccessPoint;
};

const mutateLayout = (aps: AccessPoint[], width: number, height: number) => {
  const candidate = aps.map(ap => ({ ...ap }));
  const roll = Math.random();

  if (roll < 0.2 && candidate.length > 1) {
    const removeIndex = Math.floor(Math.random() * candidate.length);
    candidate.splice(removeIndex, 1);
  } else if (roll < 0.4 && candidate.length < 12) {
    candidate.push(createRandomAp(width, height));
  } else {
    const apIndex = Math.floor(Math.random() * candidate.length);
    const jitter = Math.max(8, 80 * Math.random());
    const deltaX = (Math.random() - 0.5) * jitter;
    const deltaY = (Math.random() - 0.5) * jitter;
    candidate[apIndex].x = clamp(candidate[apIndex].x + deltaX, 40, width - 40);
    candidate[apIndex].y = clamp(candidate[apIndex].y + deltaY, 40, height - 40);
  }

  return candidate;
};

const runSimulatedAnnealing = (
  aps: AccessPoint[],
  walls: Wall[],
  target: number,
  metersPerPixel: number,
  width: number,
  height: number,
  iterations: number = 50
) => {
  let current = aps.map(ap => ({ ...ap }));
  let best = current.map(ap => ({ ...ap }));
  let { score: bestScore } = evaluateCoverage(best, walls, target, metersPerPixel, width, height);
  let { score: currentScore } = evaluateCoverage(current, walls, target, metersPerPixel, width, height);

  let temperature = 100;
  const coolingRate = 0.95;

  for (let i = 0; i < iterations; i++) {
    const candidate = mutateLayout(current, width, height);
    const { score: candidateScore } = evaluateCoverage(candidate, walls, target, metersPerPixel, width, height);

    const acceptance = Math.exp((currentScore - candidateScore) / Math.max(temperature, 0.01));

    if (candidateScore < currentScore || Math.random() < acceptance) {
      current = candidate;
      currentScore = candidateScore;

      if (currentScore < bestScore) {
        best = current.map(ap => ({ ...ap }));
        bestScore = currentScore;
      }
    }

    temperature *= coolingRate;
  }

  const bestMetrics = evaluateCoverage(best, walls, target, metersPerPixel, width, height);
  return { bestAps: best, metrics: bestMetrics };
};

const FloorPlanEditor: React.FC = () => {
  const projects = useProjectStore((state) => state.projects);
  const selectedProjectId = useProjectStore((state) => state.selectedProjectId ?? state.projects[0]?.id);
  const setSelectedProjectId = useProjectStore((state) => state.setSelectedProjectId);
  const updateProject = useProjectStore((state) => state.updateProject);
  const currentProject = projects.find(project => project.id === selectedProjectId);

  const [aps, setAps] = useState<AccessPoint[]>([]);
  const [walls, setWalls] = useState<Wall[]>([]);
  const [selectedApId, setSelectedApId] = useState<string | null>(null);
  const [selectedWallId, setSelectedWallId] = useState<string | null>(null);
  const [activeEnvToolId, setActiveEnvToolId] = useState<string>('');
  const [wallAttributes, setWallAttributes] = useState<Wall>(() => ({
    id: 'draft',
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 0,
    material: (ENV_TOOLS[0]?.material ?? 'Drywall') as Wall['material'],
    attenuation: ENV_TOOLS[0]?.attenuation ?? 3,
    thickness: ENV_TOOLS[0]?.thickness ?? 8,
    height: ENV_TOOLS[0]?.height ?? 3,
    elevation: ENV_TOOLS[0]?.elevation ?? 0,
    metadata: {
      color: ENV_TOOLS[0]?.color ?? '#94a3b8'
    }
  }));
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);
  const [signalThreshold, setSignalThreshold] = useState(COVERAGE_TARGET_DBM);
  const [floorPlan, setFloorPlan] = useState<FloorPlan>({
    opacity: currentProject?.floorPlan?.opacity ?? 0.6,
    metersPerPixel: currentProject?.floorPlan?.metersPerPixel ?? DEFAULT_METERS_PER_PIXEL,
    imageDataUrl: currentProject?.floorPlan?.imageDataUrl,
    imageName: currentProject?.floorPlan?.imageName,
    width: currentProject?.floorPlan?.width,
    height: currentProject?.floorPlan?.height,
    reference: currentProject?.floorPlan?.reference,
  });
  const [metersPerPixel, setMetersPerPixel] = useState<number>(floorPlan.metersPerPixel ?? DEFAULT_METERS_PER_PIXEL);
  const [scaleLine, setScaleLine] = useState<ScaleReference | null>(currentProject?.floorPlan?.reference ?? null);
  const [scaleInputMeters, setScaleInputMeters] = useState<number>(currentProject?.floorPlan?.reference?.distanceMeters ?? 0);
  const [isDrawingScale, setIsDrawingScale] = useState(false);
  const [draftScaleLine, setDraftScaleLine] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [showFloorPlan, setShowFloorPlan] = useState(true);
  const [populationSize, setPopulationSize] = useState(12);
  const [optimizationIterations, setOptimizationIterations] = useState(30);

  const [heatmapConfig, setHeatmapConfig] = useState({
    colorScale: 'turbo' as 'turbo' | 'viridis' | 'magma',
    minDbm: -90,
    maxDbm: -40,
    coverageThreshold: -65
  });

  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 2000, height: 2000 });

  const [isDraggingAp, setIsDraggingAp] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const [draftWall, setDraftWall] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? projects[0],
    [projects, selectedProjectId]
  );

  useEffect(() => {
    if (!selectedProjectId && projects[0]) {
      setSelectedProjectId(projects[0].id);
    }
  }, [projects, selectedProjectId, setSelectedProjectId]);

  useEffect(() => {
    if (floorPlan.width && floorPlan.height) {
      setCanvasSize({
        width: Math.max(2000, floorPlan.width + 1000),
        height: Math.max(2000, floorPlan.height + 1000)
      });
    }
  }, [floorPlan.width, floorPlan.height]);

  const persistFloorPlan = (updates: Partial<FloorPlan>) => {
    const newFloorPlan = { ...floorPlan, ...updates };
    setFloorPlan(newFloorPlan);
    if (updates.metersPerPixel !== undefined) {
      setMetersPerPixel(updates.metersPerPixel);
    }
    if (currentProject) {
      updateProject(currentProject.id, { floorPlan: newFloorPlan });
    }
  };

  const updateSelectedAp = (updates: Partial<AccessPoint>) => {
    if (!selectedApId) return;
    setAps(prev => prev.map(ap => ap.id === selectedApId ? { ...ap, ...updates } : ap));
  };

  const deleteSelected = () => {
    if (selectedApId) {
      const apIdToDelete = selectedApId;
      setAps(prev => prev.filter(ap => ap.id !== apIdToDelete));
      setSelectedApId(null);
    }
    if (selectedWallId) {
      const wallIdToDelete = selectedWallId;
      setWalls(prev => prev.filter(w => w.id !== wallIdToDelete));
      setSelectedWallId(null);
    }
  };

  const handleFloorPlanUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const img = new window.Image();
      img.onload = () => {
        persistFloorPlan({
          imageDataUrl: dataUrl,
          imageName: file.name,
          width: img.width,
          height: img.height,
        });
        setShowFloorPlan(true);
        setTransform({
          x: (containerRef.current?.clientWidth ?? 800) / 2 - img.width / 2,
          y: (containerRef.current?.clientHeight ?? 600) / 2 - img.height / 2,
          scale: 1
        });
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const clearFloorPlanImage = () => {
    persistFloorPlan({ imageDataUrl: undefined, imageName: undefined, width: undefined, height: undefined });
    setShowFloorPlan(false);
  };

  const applyScaleFromInput = (distanceMeters: number, reference?: ScaleReference | null) => {
    const activeReference = reference ?? scaleLine;
    if (!activeReference || !distanceMeters || activeReference.pixelLength <= 0) return;
    const newMetersPerPixel = distanceMeters / activeReference.pixelLength;
    const updatedReference = { ...activeReference, distanceMeters };
    setScaleLine(updatedReference);
    persistFloorPlan({ metersPerPixel: newMetersPerPixel, reference: updatedReference });
  };

  const handleScaleInputChange = (value: number) => {
    setScaleInputMeters(value);
    applyScaleFromInput(value);
  };

  const screenToCanvas = (screenX: number, screenY: number) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: (screenX - rect.left - transform.x) / transform.scale,
      y: (screenY - rect.top - transform.y) / transform.scale
    };
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const zoomSensitivity = 0.001;
      const delta = -e.deltaY * zoomSensitivity;
      const newScale = clamp(transform.scale + delta, 0.1, 5);

      const rect = containerRef.current!.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const scaleRatio = newScale / transform.scale;
      const newX = mouseX - (mouseX - transform.x) * scaleRatio;
      const newY = mouseY - (mouseY - transform.y) * scaleRatio;

      setTransform({ x: newX, y: newY, scale: newScale });
    } else {
      setTransform(prev => ({
        ...prev,
        x: prev.x - e.deltaX,
        y: prev.y - e.deltaY
      }));
    }
  };

  const handleMouseDown = (e: React.MouseEvent, id?: string, type?: 'ap' | 'wall') => {
    e.stopPropagation();

    if (e.button === 1 || (e.button === 0 && e.nativeEvent.getModifierState('Space'))) {
      setIsPanning(true);
      return;
    }

    const { x, y } = screenToCanvas(e.clientX, e.clientY);

    if (type === 'ap' && id) {
      setSelectedApId(id);
      setSelectedWallId(null);
      setIsDraggingAp(true);
      const ap = aps.find(a => a.id === id);
      if (ap) {
        dragOffset.current = {
          x: x - ap.x,
          y: y - ap.y
        };
      }
    } else if (type === 'wall' && id) {
      setSelectedWallId(id);
      setSelectedApId(null);
    } else if (isDrawingScale) {
      setDraftScaleLine({ x1: x, y1: y, x2: x, y2: y });
    } else if (activeEnvToolId) {
      setDraftWall({ x1: x, y1: y, x2: x, y2: y });
      setSelectedWallId(null);
      setSelectedApId(null);
    } else {
      setSelectedWallId(null);
      setSelectedApId(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const { x, y } = screenToCanvas(e.clientX, e.clientY);

    if (isPanning) {
      setTransform(prev => ({
        ...prev,
        x: prev.x + e.movementX,
        y: prev.y + e.movementY
      }));
      return;
    }

    if (isDraggingAp && selectedApId && dragOffset.current) {
      setAps(prev => prev.map(ap => {
        if (ap.id === selectedApId) {
          return {
            ...ap,
            x: x - dragOffset.current!.x,
            y: y - dragOffset.current!.y
          };
        }
        return ap;
      }));
      return;
    }

    if (isDrawingScale && draftScaleLine) {
      setDraftScaleLine(prev => prev ? { ...prev, x2: x, y2: y } : null);
      return;
    }

    if (activeEnvToolId && draftWall) {
      let targetX = x;
      let targetY = y;

      if (e.shiftKey) {
        const dx = Math.abs(x - draftWall.x1);
        const dy = Math.abs(y - draftWall.y1);
        if (dx > dy) targetY = draftWall.y1;
        else targetX = draftWall.x1;
      }

      setDraftWall(prev => prev ? { ...prev, x2: targetX, y2: targetY } : null);
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsDraggingAp(false);
    dragOffset.current = null;

    if (isDrawingScale && draftScaleLine) {
      const pixelLength = Math.hypot(draftScaleLine.x2 - draftScaleLine.x1, draftScaleLine.y2 - draftScaleLine.y1);
      setScaleLine({
        ...draftScaleLine,
        pixelLength,
        distanceMeters: 0
      });
      setDraftScaleLine(null);
      setIsDrawingScale(false);
    }

    if (activeEnvToolId && draftWall) {
      const newWall: Wall = {
        ...wallAttributes,
        ...draftWall,
        id: `W-${Date.now().toString().slice(-4)}`,
        metadata: { ...wallAttributes.metadata }
      };

      const len = Math.hypot(newWall.x2 - newWall.x1, newWall.y2 - newWall.y1);
      if (len > 0.1) {
        setWalls(prev => [...prev, newWall]);
      }
      setDraftWall(null);
    }
  };

  const addAp = () => {
    const template = AP_LIBRARY[0];
    const centerX = (-transform.x + (containerRef.current?.clientWidth ?? 800) / 2) / transform.scale;
    const centerY = (-transform.y + (containerRef.current?.clientHeight ?? 600) / 2) / transform.scale;

    const newAp: AccessPoint = {
      id: `AP-${Date.now().toString().slice(-4)}`,
      x: centerX,
      y: centerY,
      model: template.name,
      band: template.bands[0],
      power: template.defaultPower,
      channel: 'Auto',
      height: template.defaultHeight,
      azimuth: template.defaultAzimuth ?? 0,
      tilt: template.defaultTilt ?? 0,
      antennaGain: template.antennaGain,
      antennaPatternFile: template.patternFile,
      color: '#3b82f6'
    };

    setAps(prev => [...prev, newAp]);
    setSelectedApId(newAp.id);
    setSelectedWallId(null);
  };

  const runOptimization = async () => {
    if (aps.length === 0) {
      setAiSuggestion('Add at least one access point before running AI optimization.');
      return;
    }

    setIsOptimizing(true);

    await new Promise(resolve => setTimeout(resolve, 100));

    const before = evaluateCoverage(aps, walls, signalThreshold, metersPerPixel || DEFAULT_METERS_PER_PIXEL, canvasSize.width, canvasSize.height);
    const { bestAps, metrics } = runSimulatedAnnealing(aps, walls, signalThreshold, metersPerPixel || DEFAULT_METERS_PER_PIXEL, canvasSize.width, canvasSize.height);

    setAps(bestAps);

    const coverageDelta = metrics.coveragePercent - before.coveragePercent;
    const apDelta = metrics.apCount - before.apCount;

    let insight = `Optimization explored adding/removing APs to hit coverage targets while minimizing density.\n` +
      `Coverage ≥ ${signalThreshold} dBm: ${before.coveragePercent.toFixed(1)}% → ${metrics.coveragePercent.toFixed(1)}% (${coverageDelta >= 0 ? '+' : ''}${coverageDelta.toFixed(1)} pts).\n` +
      `AP count: ${before.apCount} → ${metrics.apCount} (${apDelta >= 0 ? '+' : ''}${apDelta}).`;

    try {
      const text = await getOptimizationSuggestions(bestAps, walls);
      if (text && !text.includes('API Key not configured')) {
        insight += `\n\nGemini notes:\n${text}`;
      }
    } catch (e) {
      console.error("AI suggestion failed", e);
    }

    setAiSuggestion(insight);
    setIsOptimizing(false);
  };

  const handleAutoDetectWalls = async () => {
    if (!floorPlan.imageDataUrl) return;

    try {
      const { detectWalls } = await import('../services/wallDetection');
      const detected = await detectWalls(floorPlan.imageDataUrl, metersPerPixel || DEFAULT_METERS_PER_PIXEL);

      // Ensure unique IDs and all required wall properties
      const uniqueDetected: Wall[] = detected.map((w, i) => ({
        id: `W-Auto-${Date.now()}-${i}`,
        x1: w.x1,
        y1: w.y1,
        x2: w.x2,
        y2: w.y2,
        material: (w.material as Wall['material']) || 'Concrete',
        attenuation: w.attenuation ?? 12,
        thickness: w.thickness ?? 10,
        height: w.height ?? 3,
        elevation: w.elevation ?? 0,
        metadata: {
          color: w.metadata?.color ?? '#475569'
        }
      }));

      setWalls(prev => [...prev, ...uniqueDetected]);
      alert(`Detected ${uniqueDetected.length} walls!`);
    } catch (e) {
      console.error("Wall detection failed", e);
      alert("Failed to detect walls. Please try a cleaner image.");
    }
  };

  const selectedAp = aps.find(a => a.id === selectedApId);
  const selectedWall = walls.find(w => w.id === selectedWallId);

  const activeScaleOverlay = draftScaleLine ?? scaleLine;
  const activeScalePixelLength: number = activeScaleOverlay
    ? ('pixelLength' in activeScaleOverlay
      ? (activeScaleOverlay.pixelLength as number)
      : Math.hypot(activeScaleOverlay.x2 - activeScaleOverlay.x1, activeScaleOverlay.y2 - activeScaleOverlay.y1))
    : 0;
  const scaleMetersLabel = scaleLine?.distanceMeters || (scaleInputMeters > 0 ? scaleInputMeters : undefined);

  const materialStyles: Record<Wall['material'], { color: string; pattern?: string; shadow?: string }> = {
    Brick: {
      color: '#b45309',
      pattern: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.2), rgba(255,255,255,0.2) 8px, transparent 8px, transparent 16px)',
      shadow: '0 0 0 1px rgba(148, 82, 31, 0.35)'
    },
    Concrete: {
      color: '#475569',
      pattern: 'linear-gradient(135deg, rgba(255,255,255,0.12) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.12) 50%, rgba(255,255,255,0.12) 75%, transparent 75%, transparent)',
      shadow: '0 0 0 1px rgba(71, 85, 105, 0.35)'
    },
    Drywall: {
      color: '#94a3b8',
      pattern: 'linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px)',
      shadow: '0 0 0 1px rgba(148, 163, 184, 0.4)'
    },
    Glass: {
      color: '#38bdf8',
      pattern: 'linear-gradient(135deg, rgba(255,255,255,0.35) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0.35) 75%, transparent 75%, transparent)',
      shadow: '0 0 0 1px rgba(56, 189, 248, 0.35)'
    }
  };

  return (
    <div className="flex h-full overflow-hidden" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      {/* Left Toolbar */}
      <div className="w-64 bg-white border-r border-slate-200 p-4 flex flex-col gap-6 overflow-y-auto shrink-0 z-20 shadow-sm">
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
          <p className="text-[11px] font-semibold text-slate-600 mb-2">Active Project</p>
          <select
            value={selectedProject?.id ?? ''}
            onChange={(e) => setSelectedProjectId(e.target.value || undefined)}
            className="w-full text-sm border border-slate-200 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-100"
          >
            <option value="">Auto-select first project</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Add Hardware</h3>
          <div className="grid grid-cols-2 gap-2">
            {HARDWARE_TOOLS.map(tool => (
              <button key={tool.id} onClick={addAp} className="flex flex-col items-center justify-center p-3 border border-slate-200 rounded-lg hover:bg-slate-50 hover:border-blue-500 transition-colors text-slate-600">
                <div className="mb-2 text-blue-600">
                  {tool.icon === 'Router' ? <Router size={20} /> : tool.icon === 'Wifi' ? <Wifi size={20} /> : <Router size={20} />}
                </div>
                <span className="text-[10px] text-center leading-tight">{tool.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Floor Plan</h3>
          <div className="space-y-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <label className="flex items-center justify-between text-sm text-slate-700 cursor-pointer hover:text-blue-600 transition-colors">
              <span className="flex items-center gap-2"><ImageIcon size={16} /> Upload Image</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/svg+xml"
                onChange={handleFloorPlanUpload}
                className="hidden"
              />
            </label>
            {floorPlan.imageName ? (
              <div className="text-xs text-slate-600 bg-white border border-slate-200 rounded px-2 py-1 flex items-center justify-between">
                <span className="truncate max-w-[120px]" title={floorPlan.imageName}>{floorPlan.imageName}</span>
                <button onClick={clearFloorPlanImage} className="text-red-500 hover:text-red-600 text-[11px]">Remove</button>
              </div>
            ) : (
              <div className="text-xs text-slate-500 italic">No floor plan loaded</div>
            )}

            <button
              onClick={handleAutoDetectWalls}
              disabled={!floorPlan.imageDataUrl}
              className="w-full flex items-center justify-center gap-2 px-2 py-1.5 bg-blue-100 text-blue-700 rounded text-xs font-medium hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Wand2 size={12} /> Auto-Detect Walls
            </button>

            <div className="space-y-1">
              <label className="flex items-center justify-between text-[11px] text-slate-600">
                <span>Opacity</span>
                <span className="font-semibold text-slate-800">{Math.round((floorPlan.opacity ?? 0.6) * 100)}%</span>
              </label>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={floorPlan.opacity ?? 0.6}
                onChange={(e) => persistFloorPlan({ opacity: Number(e.target.value) })}
                disabled={!floorPlan.imageDataUrl}
                className="w-full accent-blue-500"
              />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Heatmap Settings</h3>
          <div className="space-y-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Color Scale</label>
              <select
                value={heatmapConfig.colorScale}
                onChange={(e) => setHeatmapConfig(prev => ({ ...prev, colorScale: e.target.value as any }))}
                className="w-full border border-slate-200 rounded px-2 py-1 text-xs"
              >
                <option value="turbo">Turbo</option>
                <option value="viridis">Viridis</option>
                <option value="magma">Magma</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Range ({heatmapConfig.minDbm} to {heatmapConfig.maxDbm} dBm)</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={heatmapConfig.minDbm}
                  onChange={(e) => setHeatmapConfig(prev => ({ ...prev, minDbm: Number(e.target.value) }))}
                  className="w-full border border-slate-200 rounded px-2 py-1 text-xs"
                />
                <input
                  type="number"
                  value={heatmapConfig.maxDbm}
                  onChange={(e) => setHeatmapConfig(prev => ({ ...prev, maxDbm: Number(e.target.value) }))}
                  className="w-full border border-slate-200 rounded px-2 py-1 text-xs"
                />
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Coverage Cutoff ({heatmapConfig.coverageThreshold} dBm)</label>
              <input
                type="range"
                min="-95"
                max="-40"
                value={heatmapConfig.coverageThreshold}
                onChange={(e) => setHeatmapConfig(prev => ({ ...prev, coverageThreshold: Number(e.target.value) }))}
                className="w-full accent-blue-500"
              />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Scale</h3>
          <div className="space-y-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <button
              onClick={() => { setIsDrawingScale(true); setDraftScaleLine(null); }}
              className={`w-full flex items-center justify-center gap-2 text-xs font-semibold rounded-md px-3 py-2 border ${isDrawingScale ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-700 hover:bg-white'}`}
            >
              <Ruler size={14} /> {isDrawingScale ? 'Draw Reference' : 'Set Scale'}
            </button>
            <div className="text-[11px] text-slate-600 flex items-center justify-between">
              <span>Scale</span>
              <span className="font-semibold text-slate-800">{(metersPerPixel || DEFAULT_METERS_PER_PIXEL).toFixed(3)} m/px</span>
            </div>
            {scaleLine && (
              <div className="flex gap-2 items-center">
                <input
                  type="number"
                  value={scaleInputMeters || ''}
                  onChange={(e) => handleScaleInputChange(Number(e.target.value))}
                  placeholder="Distance (m)"
                  className="w-full border border-slate-200 rounded px-2 py-1 text-xs"
                />
                <span className="text-xs text-slate-500">m</span>
              </div>
            )}
          </div>
        </div>
      </div >

      {/* Main Editor Area */}
      < div className="flex-1 bg-slate-100 relative overflow-hidden flex flex-col" >
        {/* Top Toolbar */}
        < div className="absolute top-4 left-4 right-4 z-10 flex justify-between pointer-events-none" >
          <div className="pointer-events-auto bg-white rounded-lg shadow-sm border border-slate-200 p-1 flex gap-1">
            <button
              onClick={() => setTransform(prev => ({ ...prev, scale: prev.scale * 1.2 }))}
              className="p-2 hover:bg-slate-100 rounded text-slate-600" title="Zoom In"
            >
              <ZoomIn size={18} />
            </button>
            <button
              onClick={() => setTransform(prev => ({ ...prev, scale: prev.scale / 1.2 }))}
              className="p-2 hover:bg-slate-100 rounded text-slate-600" title="Zoom Out"
            >
              <ZoomOut size={18} />
            </button>
            <button
              onClick={() => setTransform({ x: 0, y: 0, scale: 1 })}
              className="p-2 hover:bg-slate-100 rounded text-slate-600" title="Reset View"
            >
              <Maximize size={18} />
            </button>
          </div>

          <div className="pointer-events-auto bg-white rounded-lg shadow-sm border border-slate-200 p-1 flex gap-1">
            <label className="flex items-center gap-2 px-3 py-1 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 rounded">
              <input type="checkbox" checked={showFloorPlan} onChange={(e) => setShowFloorPlan(e.target.checked)} className="rounded text-blue-600" />
              <Eye size={14} /> Floor Plan
            </label>
            <div className="w-px bg-slate-200 my-1"></div>
            <label className="flex items-center gap-2 px-3 py-1 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 rounded">
              <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} className="rounded text-blue-600" />
              <Wifi size={14} /> Heatmap
            </label>
          </div>
        </div >

        {/* Canvas Container */}
        < div
          ref={containerRef}
          className="flex-1 relative overflow-hidden cursor-crosshair bg-slate-100"
          onWheel={handleWheel}
          onMouseDown={(e) => handleMouseDown(e)}
          onMouseMove={handleMouseMove}
          style={{ cursor: isPanning ? 'grabbing' : isDraggingAp ? 'move' : 'crosshair' }}
        >
          {/* Grid Background */}
          < div className="absolute inset-0 z-0 opacity-10 pointer-events-none"
            style={{
              backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)',
              backgroundSize: `${20 * transform.scale}px ${20 * transform.scale}px`,
              backgroundPosition: `${transform.x}px ${transform.y}px`
            }}>
          </div >

          {/* Transformed Content */}
          < div
            style={{
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
              transformOrigin: '0 0',
              width: canvasSize.width,
              height: canvasSize.height,
              position: 'absolute',
              top: 0,
              left: 0
            }}
          >
            {/* Floor Plan Image */}
            {
              floorPlan.imageDataUrl && showFloorPlan && (
                <img
                  src={floorPlan.imageDataUrl}
                  alt="Floor plan"
                  className="absolute top-0 left-0 pointer-events-none select-none"
                  style={{ opacity: floorPlan.opacity ?? 0.6 }}
                  draggable={false}
                />
              )
            }

            {/* Heatmap Layer */}
            <HeatmapCanvas
              aps={aps}
              walls={walls}
              width={canvasSize.width}
              height={canvasSize.height}
              show={showHeatmap}
              metersPerPixel={metersPerPixel || DEFAULT_METERS_PER_PIXEL}
              colorScale={heatmapConfig.colorScale}
              minDbm={heatmapConfig.minDbm}
              maxDbm={heatmapConfig.maxDbm}
              coverageThreshold={heatmapConfig.coverageThreshold}
            />

            {/* Walls Layer */}
            {
              [...walls, draftWall && { ...wallAttributes, id: 'draft', ...draftWall }].filter(Boolean).map((wall, idx) => {
                const typedWall = wall as Wall;
                const style = materialStyles[typedWall.material];
                const dx = typedWall.x2 - typedWall.x1;
                const dy = typedWall.y2 - typedWall.y1;
                
                // Calculate wall length and angle
                const length = Math.sqrt(dx * dx + dy * dy);
                const angle = Math.atan2(dy, dx); // Angle in radians
                
                // Center point of the wall
                const centerX = (typedWall.x1 + typedWall.x2) / 2;
                const centerY = (typedWall.y1 + typedWall.y2) / 2;
                
                const isSelected = selectedWallId === typedWall.id;
                
                // Increase clickable area by using a minimum height
                const clickableHeight = Math.max(typedWall.thickness, 12);

                return (
                  <div
                    key={`${typedWall.id}-${idx}`}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      if (typedWall.id !== 'draft') {
                        setSelectedWallId(typedWall.id);
                        setSelectedApId(null);
                      }
                    }}
                    className={`absolute rounded-sm cursor-pointer transition-all ${typedWall.id === 'draft' ? 'opacity-60 border border-dashed border-blue-400' : ''} ${isSelected ? 'ring-2 ring-blue-500 ring-offset-1' : ''}`}
                    style={{
                      left: centerX,
                      top: centerY,
                      width: Math.max(length, 1),
                      height: clickableHeight,
                      backgroundColor: style?.color ?? '#cbd5e1',
                      backgroundImage: style?.pattern,
                      boxShadow: style?.shadow,
                      transform: `translate(-50%, -50%) rotate(${angle}rad)`,
                      transformOrigin: 'center center',
                      zIndex: isSelected ? 25 : 10
                    }}
                  />
                );
              })
            }

            {/* APs Layer */}
            {
              aps.map(ap => (
                <div
                  key={ap.id}
                  onMouseDown={(e) => handleMouseDown(e, ap.id, 'ap')}
                  className={`absolute z-20 flex flex-col items-center justify-center transform -translate-x-1/2 -translate-y-1/2 cursor-move group transition-transform ${selectedApId === ap.id ? 'scale-110' : ''}`}
                  style={{ left: ap.x, top: ap.y }}
                >
                  {selectedApId === ap.id && (
                    <div className="absolute w-24 h-24 border border-blue-400 rounded-full opacity-30 animate-ping pointer-events-none"></div>
                  )}
                  <div className={`w-8 h-8 rounded-full bg-white shadow-md border-2 flex items-center justify-center relative ${selectedApId === ap.id ? 'border-blue-500' : 'border-slate-300'}`} style={{ backgroundColor: ap.color }}>
                    <Wifi size={16} className={selectedApId === ap.id ? 'text-blue-600' : 'text-slate-500'} />
                  </div>
                  {/* Hover Info */}
                  <div className="absolute top-full mt-1 opacity-0 group-hover:opacity-100 transition-opacity bg-black/75 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap pointer-events-none">
                    {ap.model}
                  </div>
                </div>
              ))
            }

            {/* Scale Overlay */}
            {
              activeScaleOverlay && (
                <svg className="absolute top-0 left-0 z-30 pointer-events-none overflow-visible" width={canvasSize.width} height={canvasSize.height}>
                  <line
                    x1={activeScaleOverlay.x1}
                    y1={activeScaleOverlay.y1}
                    x2={activeScaleOverlay.x2}
                    y2={activeScaleOverlay.y2}
                    stroke="#0ea5e9"
                    strokeWidth={2 / transform.scale}
                    strokeDasharray={draftScaleLine ? '4 4' : '0'}
                  />
                  <circle cx={activeScaleOverlay.x1} cy={activeScaleOverlay.y1} r={4 / transform.scale} fill="#0ea5e9" />
                  <circle cx={activeScaleOverlay.x2} cy={activeScaleOverlay.y2} r={4 / transform.scale} fill="#0ea5e9" />
                  <text
                    x={(activeScaleOverlay.x1 + activeScaleOverlay.x2) / 2}
                    y={(activeScaleOverlay.y1 + activeScaleOverlay.y2) / 2 - 10 / transform.scale}
                    fill="#0f172a"
                    fontSize={12 / transform.scale}
                    textAnchor="middle"
                    fontWeight={600}
                    style={{ textShadow: '0 0 2px white' }}
                  >
                    {(scaleMetersLabel ? `${scaleMetersLabel.toFixed(2)} m` : `${activeScalePixelLength.toFixed(1)} px`)}
                  </text>
                </svg>
              )
            }
          </div >

          {/* Floating Property Inspectors */}
          {
            selectedAp && (
              <div 
                className="absolute top-4 right-4 w-64 bg-white rounded-lg shadow-lg border border-slate-200 p-4 z-40 animate-in slide-in-from-right-5 fade-in duration-200"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-xs font-bold text-slate-500 uppercase">Access Point</h3>
                  <button onClick={() => setSelectedApId(null)} className="text-slate-400 hover:text-slate-600"><X size={14} /></button>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-500">ID</span>
                    <span className="text-slate-800 font-mono text-xs">{selectedAp.id}</span>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-slate-500">Model</label>
                    <select
                      className="w-full border border-slate-200 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      value={selectedAp.model}
                      onChange={(e) => {
                        const model = AP_LIBRARY.find(m => m.name === e.target.value);
                        if (model) {
                          updateSelectedAp({
                            model: model.name,
                            band: model.bands[0],
                            power: model.defaultPower,
                            height: model.defaultHeight,
                            azimuth: model.defaultAzimuth ?? 0,
                            tilt: model.defaultTilt ?? 0,
                            antennaGain: model.antennaGain,
                            antennaPatternFile: model.patternFile
                          });
                        }
                      }}
                    >
                      {AP_LIBRARY.map(model => (
                        <option key={model.id} value={model.name}>{model.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="text-xs text-slate-500">Band</label>
                      <select
                        className="w-full border border-slate-200 rounded-md px-2 py-1"
                        value={selectedAp.band}
                        onChange={(e) => updateSelectedAp({ band: e.target.value as AccessPoint['band'] })}
                      >
                        {['2.4GHz', '5GHz', '6GHz'].map(b => (
                          <option key={b} value={b}>{b}</option>
                        ))}
                      </select>
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs text-slate-500">Channel</label>
                      <select
                        className="w-full border border-slate-200 rounded-md px-2 py-1"
                        value={selectedAp.channel}
                        onChange={(e) => {
                          const val = e.target.value === 'Auto' ? 'Auto' : Number(e.target.value);
                          updateSelectedAp({ channel: val as AccessPoint['channel'] });
                        }}
                      >
                        {CHANNEL_OPTIONS.map(ch => (
                          <option key={ch} value={ch}>{ch}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-slate-500 flex justify-between">
                      <span>Tx Power</span>
                      <span>{selectedAp.power} dBm</span>
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="30"
                      value={selectedAp.power}
                      onChange={(e) => updateSelectedAp({ power: Number(e.target.value) })}
                      className="w-full accent-blue-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs text-slate-500 flex justify-between">
                      <span>Azimuth</span>
                      <span>{selectedAp.azimuth}°</span>
                    </label>
                    <input
                      type="range"
                      min="-180"
                      max="180"
                      value={selectedAp.azimuth}
                      onChange={(e) => updateSelectedAp({ azimuth: Number(e.target.value) })}
                      className="w-full accent-blue-500"
                    />
                  </div>

                  <div className="pt-3 border-t border-slate-100">
                    <button
                      onClick={() => deleteSelected()}
                      className="w-full flex items-center justify-center gap-2 px-3 py-2 text-red-600 bg-red-50 hover:bg-red-100 rounded-md text-xs font-medium transition-colors"
                    >
                      <Trash2 size={14} /> Delete Access Point
                    </button>
                  </div>
                </div>
              </div>
            )
          }

          {
            selectedWall && (
              <div 
                className="absolute top-4 right-4 w-64 bg-white rounded-lg shadow-lg border border-slate-200 p-4 z-40 animate-in slide-in-from-right-5 fade-in duration-200"
                onMouseDown={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-xs font-bold text-slate-500 uppercase">Wall Properties</h3>
                  <button onClick={() => setSelectedWallId(null)} className="text-slate-400 hover:text-slate-600"><X size={14} /></button>
                </div>
                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-xs text-slate-500">Material</label>
                    <select
                      className="w-full border border-slate-200 rounded-md px-2 py-1"
                      value={selectedWall.material}
                      onChange={(e) => {
                        const mat = e.target.value as Wall['material'];
                        const tool = ENV_TOOLS.find(t => t.material === mat);
                        setWalls(prev => prev.map(w => w.id === selectedWall.id ? {
                          ...w,
                          material: mat,
                          attenuation: tool?.attenuation ?? w.attenuation,
                          thickness: tool?.thickness ?? w.thickness
                        } : w));
                      }}
                    >
                      {ENV_TOOLS.map(t => (
                        <option key={t.id} value={t.material}>{t.name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="pt-3 border-t border-slate-100">
                    <button
                      onClick={() => deleteSelected()}
                      className="w-full flex items-center justify-center gap-2 px-3 py-2 text-red-600 bg-red-50 hover:bg-red-100 rounded-md text-xs font-medium transition-colors"
                    >
                      <Trash2 size={14} /> Delete Wall
                    </button>
                  </div>
                </div>
              </div>
            )
          }

          {/* AI Insight Popup */}
          {
            aiSuggestion && (
              <div className="absolute bottom-8 left-8 right-8 z-20 bg-white rounded-xl shadow-xl border border-blue-100 p-4 animate-in slide-in-from-bottom-5 fade-in duration-300 max-w-2xl mx-auto">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-blue-700 font-semibold flex items-center gap-2"><Info size={18} /> AI Optimization Insights</h4>
                  <button onClick={() => setAiSuggestion(null)} className="text-slate-400 hover:text-slate-600">×</button>
                </div>
                <div className="text-sm text-slate-700 whitespace-pre-line leading-relaxed">
                  {aiSuggestion}
                </div>
              </div>
            )
          }

          {/* Bottom Action Bar */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 z-10">
            <button
              onClick={runOptimization}
              disabled={isOptimizing}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg text-sm font-medium flex items-center gap-2 transition-all disabled:opacity-70 disabled:cursor-not-allowed"
            >
              {isOptimizing ? <Loader2 className="animate-spin" size={16} /> : <><Wifi size={16} /> Optimize Coverage</>}
            </button>
          </div>
        </div >
      </div >
    </div >
  );
};

export default FloorPlanEditor;
