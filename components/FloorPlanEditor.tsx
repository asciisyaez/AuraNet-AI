import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AccessPoint, Wall, FloorPlan, ScaleReference } from '../types';
import { INITIAL_APS, INITIAL_WALLS, HARDWARE_TOOLS, ENV_TOOLS } from '../constants';
import HeatmapCanvas from './HeatmapCanvas';
import { Wifi, Router, Square, Trash2, Edit3, Loader2, Info, Image as ImageIcon, Eye, EyeOff, Ruler } from 'lucide-react';
import { getOptimizationSuggestions } from '../services/geminiService';
import { ANTENNA_PATTERNS, AP_LIBRARY, CHANNEL_OPTIONS } from '../data/apLibrary';
import { useProjectStore } from '../services/projectStore';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const COVERAGE_TARGET_DBM = -65;
const DEFAULT_METERS_PER_PIXEL = 0.6;

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
  metersPerPixel: number = DEFAULT_METERS_PER_PIXEL
) => {
  const desiredCoverage = 95;
  const gridSpacing = 60;
  const gridPoints: { x: number; y: number }[] = [];
  for (let x = gridSpacing; x < CANVAS_WIDTH; x += gridSpacing) {
    for (let y = gridSpacing; y < CANVAS_HEIGHT; y += gridSpacing) {
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

  const coveragePercent = (covered / gridPoints.length) * 100;
  const averageSignal = bestSignals.reduce((a, b) => a + b, 0) / bestSignals.length;
  const imbalancePenalty = aps.reduce((penalty, ap, idx) => {
    return aps.slice(idx + 1).reduce((inner, other) => {
      const distance = Math.hypot(ap.x - other.x, ap.y - other.y);
      return inner + (distance < 120 ? (120 - distance) * 0.5 : 0);
    }, penalty);
  }, 0);

  const coverageGap = Math.max(0, desiredCoverage - coveragePercent);
  const apPenalty = aps.length * 2;
  const signalPenalty = Math.max(0, target - averageSignal) * 0.5;

  // Lower is better
  const score = coverageGap * 6 + signalPenalty + imbalancePenalty + apPenalty;

  return { coveragePercent, averageSignal, score, apCount: aps.length };
};

const createRandomAp = () => {
  const template = AP_LIBRARY[0];
  return {
    id: generateApId(),
    x: clamp(Math.random() * CANVAS_WIDTH, 40, CANVAS_WIDTH - 40),
    y: clamp(Math.random() * CANVAS_HEIGHT, 40, CANVAS_HEIGHT - 40),
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

const mutateLayout = (aps: AccessPoint[]) => {
  const candidate = aps.map(ap => ({ ...ap }));
  const roll = Math.random();

  if (roll < 0.2 && candidate.length > 1) {
    // Remove an AP
    const removeIndex = Math.floor(Math.random() * candidate.length);
    candidate.splice(removeIndex, 1);
  } else if (roll < 0.4 && candidate.length < 12) {
    // Add a new AP
    candidate.push(createRandomAp());
  } else {
    // Move an AP slightly
    const apIndex = Math.floor(Math.random() * candidate.length);
    const jitter = Math.max(8, 80 * Math.random());
    const deltaX = (Math.random() - 0.5) * jitter;
    const deltaY = (Math.random() - 0.5) * jitter;
    candidate[apIndex].x = clamp(candidate[apIndex].x + deltaX, 40, CANVAS_WIDTH - 40);
    candidate[apIndex].y = clamp(candidate[apIndex].y + deltaY, 40, CANVAS_HEIGHT - 40);
  }

  return candidate;
};

const runSimulatedAnnealing = (
  aps: AccessPoint[],
  walls: Wall[],
  target: number,
  metersPerPixel: number,
  iterations: number = 50
) => {
  let current = aps.map(ap => ({ ...ap }));
  let best = current.map(ap => ({ ...ap }));
  let { score: bestScore } = evaluateCoverage(best, walls, target, metersPerPixel);
  let { score: currentScore } = evaluateCoverage(current, walls, target, metersPerPixel);

  let temperature = 100;
  const coolingRate = 0.95;

  for (let i = 0; i < iterations; i++) {
    const candidate = mutateLayout(current);
    const { score: candidateScore } = evaluateCoverage(candidate, walls, target, metersPerPixel);

    // Acceptance probability (Metropolis criterion)
    const acceptance = Math.exp((currentScore - candidateScore) / Math.max(temperature, 0.01));

    if (candidateScore < currentScore || Math.random() < acceptance) {
      current = candidate;
      currentScore = candidateScore;

      // Track best solution found
      if (currentScore < bestScore) {
        best = current.map(ap => ({ ...ap }));
        bestScore = currentScore;
      }
    }

    // Cool down
    temperature *= coolingRate;
  }

  const bestMetrics = evaluateCoverage(best, walls, target, metersPerPixel);
  return { bestAps: best, metrics: bestMetrics };
};

const FloorPlanEditor: React.FC = () => {
  const projects = useProjectStore((state) => state.projects);
  const selectedProjectId = useProjectStore((state) => state.selectedProjectId ?? state.projects[0]?.id);
  const setSelectedProjectId = useProjectStore((state) => state.setSelectedProjectId);
  const updateProject = useProjectStore((state) => state.updateProject);
  const currentProject = projects.find(project => project.id === selectedProjectId);

  const [aps, setAps] = useState<AccessPoint[]>(INITIAL_APS);
  const [walls, setWalls] = useState<Wall[]>(INITIAL_WALLS);
  const [selectedApId, setSelectedApId] = useState<string | null>(null);
  const [activeEnvToolId, setActiveEnvToolId] = useState<string>(ENV_TOOLS[0]?.id ?? '');
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

  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const editorRef = useRef<HTMLDivElement>(null);
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

  const handleMouseDown = (e: React.MouseEvent, id: string, type: 'ap') => {
    e.stopPropagation();
    if (type === 'ap') {
      setSelectedApId(id);
      setIsDragging(true);
      const ap = aps.find(a => a.id === id);
      if (ap && editorRef.current) {
        const rect = editorRef.current.getBoundingClientRect();
        dragOffset.current = {
          x: e.clientX - rect.left - ap.x,
          y: e.clientY - rect.top - ap.y
        };
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && selectedApId && editorRef.current) {
      const rect = editorRef.current.getBoundingClientRect();
      const newX = e.clientX - rect.left - dragOffset.current.x;
      const newY = e.clientY - rect.top - dragOffset.current.y;
      
      // Boundary checks
      const clampedX = Math.max(0, Math.min(newX, rect.width - 40));
      const clampedY = Math.max(0, Math.min(newY, rect.height - 40));

      setAps(prev => prev.map(ap => ap.id === selectedApId ? { ...ap, x: clampedX, y: clampedY } : ap));
    } else if (draftScaleLine && editorRef.current && isDrawingScale) {
      const rect = editorRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setDraftScaleLine(prev => prev ? { ...prev, x2: x, y2: y } : null);
    } else if (draftWall && editorRef.current) {
      const rect = editorRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setDraftWall(prev => prev ? { ...prev, x2: x, y2: y } : null);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    if (draftScaleLine && isDrawingScale) {
      const pixelLength = Math.hypot(draftScaleLine.x2 - draftScaleLine.x1, draftScaleLine.y2 - draftScaleLine.y1);
      const reference: ScaleReference = {
        ...draftScaleLine,
        pixelLength,
        distanceMeters: scaleInputMeters || scaleLine?.distanceMeters || 0,
      };
      setScaleLine(reference);
      if (scaleInputMeters > 0 && pixelLength > 0) {
        applyScaleFromInput(scaleInputMeters, reference);
      } else {
        persistFloorPlan({ reference });
      }
      setDraftScaleLine(null);
      setIsDrawingScale(false);
      return;
    }

    if (draftWall) {
      const newWall: Wall = {
        ...wallAttributes,
        ...draftWall,
        id: `W-${Date.now().toString().slice(-4)}`,
        metadata: {
          ...wallAttributes.metadata,
        }
      };

      // Ignore tiny drags
      const distance = Math.hypot(draftWall.x2 - draftWall.x1, draftWall.y2 - draftWall.y1);
      if (distance > 4) {
        setWalls(prev => [...prev, newWall]);
      }
      setDraftWall(null);
    }
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (!editorRef.current || isDragging) return;
    const rect = editorRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDrawingScale) {
      setDraftScaleLine({ x1: x, y1: y, x2: x, y2: y });
      return;
    }

    if (!activeEnvToolId) return;

    setDraftWall({ x1: x, y1: y, x2: x, y2: y });
  };

  const addAp = () => {
    const template = AP_LIBRARY[0];
    const newAp: AccessPoint = {
      id: `AP-${Date.now().toString().slice(-4)}`,
      x: 350,
      y: 250,
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
    setAps([...aps, newAp]);
    setSelectedApId(newAp.id);
  };

  const deleteSelected = () => {
    if (selectedApId) {
      setAps(aps.filter(ap => ap.id !== selectedApId));
      setSelectedApId(null);
    }
  };

  const runOptimization = async () => {
    setIsOptimizing(true);
    setAiSuggestion('Evolving candidate layouts for coverage and efficiency...');

    if (aps.length === 0) {
      setAiSuggestion('Add at least one access point before running AI optimization.');
      setIsOptimizing(false);
      return;
    }

    const before = evaluateCoverage(aps, walls, signalThreshold, metersPerPixel);
    const { bestAps, metrics } = runSimulatedAnnealing(aps, walls, signalThreshold, metersPerPixel);
    setAps(bestAps);

    const coverageDelta = metrics.coveragePercent - before.coveragePercent;
    const apDelta = metrics.apCount - before.apCount;

    let insight = `Optimization explored adding/removing APs to hit coverage targets while minimizing density.\n` +
      `Coverage ≥ ${signalThreshold} dBm: ${before.coveragePercent.toFixed(1)}% → ${metrics.coveragePercent.toFixed(1)}% (${coverageDelta >= 0 ? '+' : ''}${coverageDelta.toFixed(1)} pts).\n` +
      `AP count: ${before.apCount} → ${metrics.apCount} (${apDelta >= 0 ? '+' : ''}${apDelta}).\n` +
      `Population ${Math.max(4, populationSize)} • Generations ${Math.max(10, optimizationIterations)}.`;

    const text = await getOptimizationSuggestions(bestAps, walls);
    if (!text.includes('API Key not configured')) {
      insight += `\n\nGemini notes:\n${text}`;
    }

    setAiSuggestion(insight);
    setIsOptimizing(false);
  };

  const selectedAp = aps.find(a => a.id === selectedApId);
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
    <div className="flex h-full" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      {/* Left Toolbar */}
      <div className="w-64 bg-white border-r border-slate-200 p-4 flex flex-col gap-6 overflow-y-auto shrink-0">
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
          {selectedProject && (
            <div className="mt-2 text-[11px] text-slate-600 space-y-1">
              <div className="flex justify-between">
                <span>Units</span>
                <span className="font-semibold text-slate-800">{selectedProject.settings.units}</span>
              </div>
              <div className="flex justify-between">
                <span>Signal Profiles</span>
                <span className="text-right text-slate-700 truncate max-w-[120px]">
                  {selectedProject.settings.defaultSignalProfiles.join(', ')}
                </span>
              </div>
            </div>
          )}
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
            <label className="flex items-center justify-between text-sm text-slate-700">
              <span className="flex items-center gap-2"><ImageIcon size={16} /> Upload (PNG/JPEG/SVG)</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/svg+xml"
                onChange={handleFloorPlanUpload}
                className="text-xs text-slate-500"
              />
            </label>
            {floorPlan.imageName ? (
              <div className="text-xs text-slate-600 bg-white border border-slate-200 rounded px-2 py-1 flex items-center justify-between">
                <span className="truncate" title={floorPlan.imageName}>Loaded: {floorPlan.imageName}</span>
                <button onClick={clearFloorPlanImage} className="text-red-500 hover:text-red-600 text-[11px]">Remove</button>
              </div>
            ) : (
              <div className="text-xs text-slate-500">No image loaded</div>
            )}
            <div className="flex items-center justify-between text-sm text-slate-700">
              <span className="flex items-center gap-2">
                {showFloorPlan ? <Eye size={14} className="text-blue-500" /> : <EyeOff size={14} className="text-slate-400" />}
                Floor plan layer
              </span>
              <label className="inline-flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={showFloorPlan}
                  onChange={(e) => setShowFloorPlan(e.target.checked)}
                  className="rounded text-blue-600"
                  disabled={!floorPlan.imageDataUrl}
                />
                Show
              </label>
            </div>
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
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Environment</h3>
          <div className="grid grid-cols-3 gap-2">
            {ENV_TOOLS.map(tool => (
              <button
                key={tool.id}
                onClick={() => {
                  setActiveEnvToolId(tool.id);
                  setWallAttributes(prev => ({
                    ...prev,
                    material: tool.material as Wall['material'],
                    attenuation: tool.attenuation,
                    thickness: tool.thickness,
                    height: tool.height,
                    elevation: tool.elevation,
                    metadata: {
                      ...prev.metadata,
                      color: tool.color
                    }
                  }));
                }}
                className={`flex flex-col items-center justify-center p-2 border rounded-lg transition-colors text-slate-600 ${activeEnvToolId === tool.id ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 hover:bg-slate-50'}`}
              >
                <Square size={16} className={`mb-1 ${activeEnvToolId === tool.id ? 'text-blue-500' : 'text-slate-400'}`} />
                <span className="text-[10px] text-center">{tool.name}</span>
                <span className="text-[10px] text-center text-slate-400">{tool.attenuation}dB</span>
              </button>
            ))}
          </div>
          <div className="mt-3 space-y-2 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <div className="flex justify-between text-[11px] text-slate-600">
              <span>Material</span>
              <span className="font-semibold text-slate-800">{wallAttributes.material}</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
              <label className="flex flex-col gap-1">
                <span>Attenuation (dB)</span>
                <input
                  type="number"
                  className="border border-slate-200 rounded px-2 py-1 text-sm"
                  value={wallAttributes.attenuation}
                  onChange={e => setWallAttributes(prev => ({ ...prev, attenuation: Number(e.target.value) }))}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>Thickness (px)</span>
                <input
                  type="number"
                  className="border border-slate-200 rounded px-2 py-1 text-sm"
                  value={wallAttributes.thickness}
                  onChange={e => setWallAttributes(prev => ({ ...prev, thickness: Number(e.target.value) }))}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>Height (m)</span>
                <input
                  type="number"
                  className="border border-slate-200 rounded px-2 py-1 text-sm"
                  value={wallAttributes.height}
                  onChange={e => setWallAttributes(prev => ({ ...prev, height: Number(e.target.value) }))}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>Elevation (m)</span>
                <input
                  type="number"
                  className="border border-slate-200 rounded px-2 py-1 text-sm"
                  value={wallAttributes.elevation}
                  onChange={e => setWallAttributes(prev => ({ ...prev, elevation: Number(e.target.value) }))}
                />
              </label>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Layers</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-2 text-sm text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                className="rounded text-blue-600 focus:ring-blue-500"
                checked={!!floorPlan.imageDataUrl && showFloorPlan}
                onChange={(e) => setShowFloorPlan(e.target.checked)}
                disabled={!floorPlan.imageDataUrl}
              />
              <span>Floor Plan</span>
              {floorPlan.imageDataUrl && (
                <span className="text-[11px] text-slate-500">{Math.round((floorPlan.opacity ?? 0.6) * 100)}%</span>
              )}
            </label>
            <label className="flex items-center space-x-2 text-sm text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                className="rounded text-blue-600 focus:ring-blue-500"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
              <span>Heatmap (Signal)</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-slate-700 cursor-pointer">
              <input type="checkbox" className="rounded text-blue-600 focus:ring-blue-500" checked onChange={()=>{}} />
              <span>AP Placement</span>
            </label>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Scale & Units</h3>
          <div className="space-y-3 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <button
              onClick={() => { setIsDrawingScale(true); setDraftScaleLine(null); }}
              className={`w-full flex items-center justify-center gap-2 text-xs font-semibold rounded-md px-3 py-2 border ${isDrawingScale ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-700 hover:bg-white'}`}
            >
              <Ruler size={14} /> {isDrawingScale ? 'Click & drag reference line' : 'Draw reference line'}
            </button>
            {scaleLine && (
              <div className="text-[11px] text-slate-600 bg-white border border-slate-200 rounded px-2 py-1 flex justify-between">
                <span>Pixels</span>
                <span className="font-semibold text-slate-800">{(scaleLine.pixelLength ?? 0).toFixed(1)} px</span>
              </div>
            )}
            <label className="text-xs text-slate-600 space-y-1 block">
              <span>Reference length (meters)</span>
              <input
                type="number"
                value={scaleInputMeters || ''}
                onChange={(e) => handleScaleInputChange(Number(e.target.value))}
                placeholder="e.g. 10"
                className="w-full border border-slate-200 rounded px-2 py-1 text-sm"
              />
            </label>
            <div className="text-[11px] text-slate-600 flex items-center justify-between">
              <span>Meters per pixel</span>
              <span className="font-semibold text-slate-800">{(metersPerPixel || DEFAULT_METERS_PER_PIXEL).toFixed(3)} m/px</span>
            </div>
          </div>
        </div>
        
        <div className="mt-auto">
             <div className="mb-4 bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-600">
                  <span>Population size</span>
                  <input
                    type="number"
                    min={4}
                    max={48}
                    value={populationSize}
                    onChange={(e) => setPopulationSize(Number(e.target.value))}
                    className="w-20 border border-slate-200 rounded px-2 py-1 text-right text-sm"
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-600">
                  <span>Generations</span>
                  <input
                    type="number"
                    min={10}
                    max={200}
                    value={optimizationIterations}
                    onChange={(e) => setOptimizationIterations(Number(e.target.value))}
                    className="w-20 border border-slate-200 rounded px-2 py-1 text-right text-sm"
                  />
                </div>
             </div>
             <div className="mb-2">
                <label className="text-xs font-semibold text-slate-600">Signal Threshold ({signalThreshold}dBm)</label>
                <input
                  type="range"
                  min="-90"
                  max="-40"
                  step="1"
                  value={signalThreshold}
                  onChange={(e) => setSignalThreshold(Number(e.target.value))}
                  className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-2"
                />
             </div>
             <button
              onClick={runOptimization}
              disabled={isOptimizing}
              className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium flex items-center justify-center gap-2 transition-all">
               {isOptimizing ? <Loader2 className="animate-spin" size={16}/> : "Run AI Optimization"}
             </button>
        </div>
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 bg-slate-100 p-8 overflow-hidden relative flex flex-col">
        {/* Editor Toolbar */}
        <div className="absolute top-6 right-8 z-10 bg-white rounded-lg shadow-sm border border-slate-200 p-1 flex gap-1">
            <button className="p-2 hover:bg-slate-100 rounded text-slate-600"><Wifi size={18}/></button>
            <div className="w-px bg-slate-200 my-1"></div>
            <button className="p-2 hover:bg-slate-100 rounded text-slate-600"><Edit3 size={18}/></button>
        </div>

        {/* AI Insight Popup */}
        {aiSuggestion && (
           <div className="absolute bottom-8 left-8 right-8 z-20 bg-white rounded-xl shadow-xl border border-blue-100 p-4 animate-in slide-in-from-bottom-5 fade-in duration-300">
              <div className="flex justify-between items-start mb-2">
                 <h4 className="text-blue-700 font-semibold flex items-center gap-2"><Info size={18}/> AI Optimization Insights</h4>
                 <button onClick={() => setAiSuggestion(null)} className="text-slate-400 hover:text-slate-600">×</button>
              </div>
              <div className="text-sm text-slate-700 whitespace-pre-line leading-relaxed">
                {aiSuggestion}
              </div>
           </div>
        )}

        {/* Canvas Container */}
        <div
          ref={editorRef}
          className="bg-white shadow-lg rounded-sm w-[800px] h-[600px] mx-auto relative select-none cursor-crosshair border border-slate-300"
          onMouseMove={handleMouseMove}
          onMouseDown={handleCanvasMouseDown}
        >
          {/* Grid Background */}
          <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>

          {/* Floor Plan Image */}
          {floorPlan.imageDataUrl && showFloorPlan && (
            <img
              src={floorPlan.imageDataUrl}
              alt={floorPlan.imageName || 'Floor plan'}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none z-5"
              style={{ opacity: floorPlan.opacity ?? 0.6 }}
              draggable={false}
            />
          )}

          {/* Heatmap Layer */}
          <HeatmapCanvas
             aps={aps}
             walls={walls}
             width={CANVAS_WIDTH}
             height={CANVAS_HEIGHT}
             show={showHeatmap}
             metersPerPixel={metersPerPixel || DEFAULT_METERS_PER_PIXEL}
          />

          {/* Scale Overlay */}
          {activeScaleOverlay && (
            <svg className="absolute inset-0 z-30 pointer-events-none" width={CANVAS_WIDTH} height={CANVAS_HEIGHT}>
              <line
                x1={activeScaleOverlay.x1}
                y1={activeScaleOverlay.y1}
                x2={activeScaleOverlay.x2}
                y2={activeScaleOverlay.y2}
                stroke="#0ea5e9"
                strokeWidth={3}
                strokeDasharray={draftScaleLine ? '4 4' : '0'}
              />
              <circle cx={activeScaleOverlay.x1} cy={activeScaleOverlay.y1} r={4} fill="#0ea5e9" />
              <circle cx={activeScaleOverlay.x2} cy={activeScaleOverlay.y2} r={4} fill="#0ea5e9" />
              <text x={(activeScaleOverlay.x1 + activeScaleOverlay.x2) / 2} y={(activeScaleOverlay.y1 + activeScaleOverlay.y2) / 2 - 8} fill="#0f172a" fontSize="12" textAnchor="middle" fontWeight={600}>
                {(scaleMetersLabel ? `${scaleMetersLabel.toFixed(2)} m` : `${activeScalePixelLength.toFixed(1)} px`)}
              </text>
            </svg>
          )}

          {/* Walls */}
          {[...walls, draftWall && { ...wallAttributes, id: 'draft', ...draftWall }].filter(Boolean).map((wall, idx) => {
            const typedWall = wall as Wall;
            const style = materialStyles[typedWall.material];
            const dx = typedWall.x2 - typedWall.x1;
            const dy = typedWall.y2 - typedWall.y1;
            const isVertical = Math.abs(dx) < Math.abs(dy);
            const width = isVertical ? typedWall.thickness : Math.max(Math.abs(dx), 1);
            const height = isVertical ? Math.max(Math.abs(dy), 1) : typedWall.thickness;
            const left = Math.min(typedWall.x1, typedWall.x2) - (isVertical ? typedWall.thickness / 2 : 0);
            const top = Math.min(typedWall.y1, typedWall.y2) - (!isVertical ? typedWall.thickness / 2 : 0);

            return (
              <div
                key={`${typedWall.id}-${idx}`}
                className={`absolute z-10 rounded-sm ${typedWall.id === 'draft' ? 'opacity-60 border border-dashed border-blue-400' : ''}`}
                style={{
                  left,
                  top,
                  width,
                  height,
                  backgroundColor: style?.color ?? '#cbd5e1',
                  backgroundImage: style?.pattern,
                  boxShadow: style?.shadow,
                  opacity: typedWall.id === 'draft' ? 0.7 : 0.9,
                }}
              >
                {typedWall.id !== 'draft' && (
                  <div className="absolute -top-5 left-1/2 -translate-x-1/2 bg-white text-[10px] px-2 py-1 rounded shadow-sm border border-slate-200 flex gap-2">
                    <span className="font-semibold text-slate-700">{typedWall.material}</span>
                    <span className="text-slate-500">{typedWall.thickness}px • {typedWall.height}m</span>
                  </div>
                )}
              </div>
            );
          })}

          {/* Access Points */}
          {aps.map(ap => (
            <div
              key={ap.id}
              onMouseDown={(e) => handleMouseDown(e, ap.id, 'ap')}
              className={`absolute z-20 flex flex-col items-center justify-center transform -translate-x-1/2 -translate-y-1/2 cursor-move group transition-transform ${selectedApId === ap.id ? 'scale-110' : ''}`}
              style={{ left: ap.x, top: ap.y }}
            >
               {/* Signal Rings Animation */}
               {selectedApId === ap.id && (
                  <>
                     <div className="absolute w-24 h-24 border border-blue-400 rounded-full opacity-30 animate-ping"></div>
                     <div className="absolute w-16 h-16 bg-blue-100 rounded-full opacity-40"></div>
                  </>
               )}

               <div className={`w-10 h-10 rounded-full bg-white shadow-md border-2 flex items-center justify-center relative ${selectedApId === ap.id ? 'border-blue-500' : 'border-slate-300'}`}>
                  <Wifi size={20} className={selectedApId === ap.id ? 'text-blue-600' : 'text-slate-500'} />
                  {/* Badge */}
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 border-2 border-white rounded-full"></div>
               </div>
               <span className="mt-1 text-[10px] font-bold text-slate-700 bg-white/80 px-1 rounded shadow-sm">{ap.id}</span>

               {/* Popover actions on hover/select */}
               {selectedApId === ap.id && (
                 <div className="absolute top-full mt-2 flex flex-col gap-1 bg-white p-1 rounded-lg shadow-lg border border-slate-100 min-w-[120px]">
                    <div className="px-2 py-1 border-b border-slate-100">
                        <span className="text-[10px] text-slate-500 block">Model</span>
                        <span className="text-xs font-medium text-slate-800">{ap.model}</span>
                    </div>
                    <button 
                        onClick={(e) => { e.stopPropagation(); deleteSelected(); }}
                        className="flex items-center gap-2 px-2 py-1 text-red-600 hover:bg-red-50 rounded text-xs"
                    >
                        <Trash2 size={12}/> Delete
                    </button>
                 </div>
               )}
            </div>
          ))}

        </div>

        {/* Selected Property Inspector (Floating) */}
        {selectedAp && (
           <div className="absolute bottom-8 right-8 w-64 bg-white rounded-lg shadow-xl border border-slate-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                 <h4 className="font-semibold text-sm text-slate-800">Properties</h4>
                 <button onClick={() => setSelectedApId(null)} className="text-slate-400 hover:text-slate-600">×</button>
              </div>
              <div className="p-4 space-y-4 text-sm">
                 <div className="flex items-center justify-between">
                   <span className="text-slate-500">ID</span>
                   <span className="text-slate-800 font-mono">{selectedAp.id}</span>
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
                        <option key={model.id} value={model.name}>{model.name} · {model.vendor}</option>
                      ))}
                    </select>
                    {selectedAp.antennaPatternFile && (
                      <p className="text-[11px] text-slate-500">Pattern: {selectedAp.antennaPatternFile}</p>
                    )}
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

                 <div className="grid grid-cols-2 gap-3">
                   <div className="space-y-1">
                     <label className="text-xs text-slate-500">Power (dBm)</label>
                     <input
                       type="range"
                       min="5"
                       max="30"
                       value={selectedAp.power}
                       onChange={(e) => updateSelectedAp({ power: Number(e.target.value) })}
                       className="w-full"
                     />
                     <div className="text-right text-[11px] text-slate-500">{selectedAp.power} dBm</div>
                   </div>
                   <div className="space-y-1">
                     <label className="text-xs text-slate-500">Height (m)</label>
                     <input
                       type="number"
                       value={selectedAp.height}
                       min={1}
                       max={8}
                       step={0.1}
                       onChange={(e) => updateSelectedAp({ height: Number(e.target.value) })}
                       className="w-full border border-slate-200 rounded-md px-2 py-1"
                     />
                   </div>
                 </div>

                 <div className="grid grid-cols-2 gap-3">
                   <div className="space-y-1">
                     <label className="text-xs text-slate-500">Azimuth (°)</label>
                     <input
                       type="range"
                       min="-180"
                       max="180"
                       value={selectedAp.azimuth}
                       onChange={(e) => updateSelectedAp({ azimuth: Number(e.target.value) })}
                       className="w-full"
                     />
                     <div className="text-right text-[11px] text-slate-500">{selectedAp.azimuth}°</div>
                   </div>
                   <div className="space-y-1">
                     <label className="text-xs text-slate-500">Tilt (°)</label>
                     <input
                       type="range"
                       min="-30"
                       max="30"
                       value={selectedAp.tilt}
                       onChange={(e) => updateSelectedAp({ tilt: Number(e.target.value) })}
                       className="w-full"
                     />
                     <div className="text-right text-[11px] text-slate-500">{selectedAp.tilt}°</div>
                   </div>
                 </div>

                 <div className="space-y-1">
                   <label className="text-xs text-slate-500">Antenna gain / pattern</label>
                   <div className="grid grid-cols-2 gap-2">
                     <input
                       type="number"
                       value={selectedAp.antennaGain}
                       min={0}
                       max={20}
                       step={0.5}
                       onChange={(e) => updateSelectedAp({ antennaGain: Number(e.target.value) })}
                       className="w-full border border-slate-200 rounded-md px-2 py-1"
                     />
                     <select
                       className="w-full border border-slate-200 rounded-md px-2 py-1"
                       value={selectedAp.antennaPatternFile ?? ''}
                       onChange={(e) => {
                         const pattern = ANTENNA_PATTERNS.find(p => p.file === e.target.value);
                         updateSelectedAp({
                           antennaPatternFile: e.target.value || undefined,
                           antennaGain: pattern ? pattern.gain : selectedAp.antennaGain
                         });
                       }}
                     >
                       <option value="">Custom</option>
                       {ANTENNA_PATTERNS.map(pattern => (
                         <option key={pattern.file} value={pattern.file}>{pattern.label}</option>
                       ))}
                     </select>
                   </div>
                   <input
                     type="file"
                     accept=".ant,.msi,.json,.csv"
                     onChange={(e) => {
                       const file = e.target.files?.[0];
                       if (file) {
                         updateSelectedAp({ antennaPatternFile: file.name });
                       }
                     }}
                     className="w-full text-[11px] text-slate-500"
                   />
                 </div>

                 <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 space-y-1">
                   <div className="flex items-center justify-between">
                     <span>Band</span>
                     <span className="font-medium text-slate-700">{selectedAp.band}</span>
                   </div>
                   <div className="flex items-center justify-between">
                     <span>Channel</span>
                     <span className="font-medium text-slate-700">{selectedAp.channel}</span>
                   </div>
                 </div>
              </div>
           </div>
        )}
      </div>
    </div>
  );
};

export default FloorPlanEditor;
