import React, { useState, useRef } from 'react';
import { AccessPoint, Wall } from '../types';
import { INITIAL_APS, INITIAL_WALLS, HARDWARE_TOOLS, ENV_TOOLS } from '../constants';
import HeatmapCanvas from './HeatmapCanvas';
import { Wifi, Router, Square, Trash2, Edit3, Loader2, Info } from 'lucide-react';
import { getOptimizationSuggestions } from '../services/geminiService';
import { ANTENNA_PATTERNS, AP_LIBRARY, CHANNEL_OPTIONS } from '../data/apLibrary';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const COVERAGE_TARGET_DBM = -65;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max));

const segmentsIntersect = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number, dx: number, dy: number) => {
  const det = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (det === 0) return false;
  const lambda = ((dy - cy) * (dx - ax) + (cx - dx) * (dy - ay)) / det;
  const gamma = ((ay - cy) * (dx - ax) + (cx - ax) * (dy - ay)) / det;
  return lambda > 0 && lambda < 1 && gamma > 0 && gamma < 1;
};

const calculateSignal = (ap: AccessPoint, x: number, y: number, walls: Wall[]) => {
  const horizontalDistance = Math.hypot(ap.x - x, ap.y - y);
  const distance = Math.hypot(horizontalDistance, ap.height);
  const pathLoss = 20 * Math.log10(distance + 1);
  const wallLoss = walls.reduce((loss, wall) => {
    const intersects = segmentsIntersect(ap.x, ap.y, x, y, wall.x1, wall.y1, wall.x2, wall.y2);
    return loss + (intersects ? wall.attenuation : 0);
  }, 0);
  return ap.power + ap.antennaGain - pathLoss - wallLoss;
};

const evaluateCoverage = (aps: AccessPoint[], walls: Wall[], target: number = COVERAGE_TARGET_DBM) => {
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
      ...aps.map(ap => calculateSignal(ap, point.x, point.y, walls))
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

  // Lower is better
  const score = (100 - coveragePercent) * 5 + (target - averageSignal) * 0.5 + imbalancePenalty;

  return { coveragePercent, averageSignal, score };
};

const runSimulatedAnnealing = (aps: AccessPoint[], walls: Wall[], target: number) => {
  let current = aps.map(ap => ({ ...ap }));
  let best = current.map(ap => ({ ...ap }));
  let { score: bestScore } = evaluateCoverage(best, walls, target);
  let { score: currentScore } = evaluateCoverage(current, walls, target);

  const iterations = 250;
  for (let i = 0; i < iterations; i++) {
    const temperature = 1 - i / iterations;
    const candidate = current.map(ap => ({ ...ap }));
    const apIndex = Math.floor(Math.random() * candidate.length);
    const jitter = Math.max(10, 120 * temperature);
    const deltaX = (Math.random() - 0.5) * jitter;
    const deltaY = (Math.random() - 0.5) * jitter;

    candidate[apIndex].x = clamp(candidate[apIndex].x + deltaX, 40, CANVAS_WIDTH - 40);
    candidate[apIndex].y = clamp(candidate[apIndex].y + deltaY, 40, CANVAS_HEIGHT - 40);

    const { score: candidateScore } = evaluateCoverage(candidate, walls, target);
    const acceptance = Math.exp((currentScore - candidateScore) / Math.max(temperature, 0.01));
    if (candidateScore < currentScore || Math.random() < acceptance) {
      current = candidate;
      currentScore = candidateScore;
    }

    if (candidateScore < bestScore) {
      best = candidate.map(ap => ({ ...ap }));
      bestScore = candidateScore;
    }
  }

  const bestMetrics = evaluateCoverage(best, walls, target);
  return { bestAps: best, metrics: bestMetrics };
};

const FloorPlanEditor: React.FC = () => {
  const [aps, setAps] = useState<AccessPoint[]>(INITIAL_APS);
  const [walls, setWalls] = useState<Wall[]>(INITIAL_WALLS);
  const [selectedApId, setSelectedApId] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);
  const [signalThreshold, setSignalThreshold] = useState(COVERAGE_TARGET_DBM);

  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const editorRef = useRef<HTMLDivElement>(null);

  const updateSelectedAp = (updates: Partial<AccessPoint>) => {
    if (!selectedApId) return;
    setAps(prev => prev.map(ap => ap.id === selectedApId ? { ...ap, ...updates } : ap));
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
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
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
    setAiSuggestion('Evaluating layout with simulated annealing...');

    const before = evaluateCoverage(aps, walls, signalThreshold);
    const { bestAps, metrics } = runSimulatedAnnealing(aps, walls, signalThreshold);
    setAps(bestAps);

    let insight = `AI placement updated AP coordinates for stronger coverage.\n` +
      `Coverage >= ${signalThreshold} dBm: ${before.coveragePercent.toFixed(1)}% → ${metrics.coveragePercent.toFixed(1)}%.`;

    const text = await getOptimizationSuggestions(bestAps, walls);
    if (!text.includes('API Key not configured')) {
      insight += `\n\nGemini notes:\n${text}`;
    }

    setAiSuggestion(insight);
    setIsOptimizing(false);
  };

  const selectedAp = aps.find(a => a.id === selectedApId);

  return (
    <div className="flex h-full" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      {/* Left Toolbar */}
      <div className="w-64 bg-white border-r border-slate-200 p-4 flex flex-col gap-6 overflow-y-auto shrink-0">
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
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Environment</h3>
          <div className="grid grid-cols-3 gap-2">
            {ENV_TOOLS.map(tool => (
              <button key={tool.id} className="flex flex-col items-center justify-center p-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors text-slate-600">
                <Square size={16} className="mb-1 text-slate-400" />
                <span className="text-[10px] text-center">{tool.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Layers</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-2 text-sm text-slate-700 cursor-pointer">
              <input type="checkbox" className="rounded text-blue-600 focus:ring-blue-500" checked disabled />
              <span>Floor Plan</span>
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
        
        <div className="mt-auto">
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
        >
          {/* Grid Background */}
          <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>

          {/* Heatmap Layer */}
          <HeatmapCanvas 
             aps={aps} 
             walls={walls} 
             width={800} 
             height={600} 
             show={showHeatmap} 
          />

          {/* Walls */}
          {walls.map(wall => (
             <div 
                key={wall.id}
                className="absolute z-10 bg-slate-800"
                style={{
                  left: Math.min(wall.x1, wall.x2),
                  top: Math.min(wall.y1, wall.y2),
                  width: Math.abs(wall.x2 - wall.x1) || (wall.type === 'Concrete' ? 8 : 4),
                  height: Math.abs(wall.y2 - wall.y1) || (wall.type === 'Concrete' ? 8 : 4),
                  opacity: 0.8
                }}
             />
          ))}

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
