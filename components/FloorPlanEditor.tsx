import React, { useState, useRef } from 'react';
import { AccessPoint, Wall } from '../types';
import { INITIAL_APS, INITIAL_WALLS, HARDWARE_TOOLS, ENV_TOOLS } from '../constants';
import HeatmapCanvas from './HeatmapCanvas';
import { Wifi, Router, Square, Trash2, Edit3, Loader2, Info } from 'lucide-react';
import { getOptimizationSuggestions } from '../services/geminiService';

const FloorPlanEditor: React.FC = () => {
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
    material: ENV_TOOLS[0]?.material ?? 'Drywall',
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

  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const editorRef = useRef<HTMLDivElement>(null);
  const [draftWall, setDraftWall] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

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
    } else if (draftWall && editorRef.current) {
      const rect = editorRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setDraftWall(prev => prev ? { ...prev, x2: x, y2: y } : null);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
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
    if (!editorRef.current || isDragging || !activeEnvToolId) return;
    const rect = editorRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDraftWall({ x1: x, y1: y, x2: x, y2: y });
  };

  const addAp = () => {
    const newAp: AccessPoint = {
      id: `AP-${Date.now().toString().slice(-4)}`,
      x: 350,
      y: 250,
      model: 'Wi-Fi 6E Omni',
      power: 20,
      channel: 'Auto',
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
    setAiSuggestion(null);
    const result = await getOptimizationSuggestions(aps, walls);
    setAiSuggestion(result);
    setIsOptimizing(false);
  };

  const selectedAp = aps.find(a => a.id === selectedApId);

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
              <button
                key={tool.id}
                onClick={() => {
                  setActiveEnvToolId(tool.id);
                  setWallAttributes(prev => ({
                    ...prev,
                    material: tool.material,
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
                <label className="text-xs font-semibold text-slate-600">Signal Threshold (-65dBm)</label>
                <input type="range" className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer mt-2" />
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

          {/* Heatmap Layer */}
          <HeatmapCanvas
             aps={aps}
             walls={walls}
             width={800}
             height={600}
             show={showHeatmap}
          />

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
              <div className="p-4 space-y-3">
                 <div className="grid grid-cols-2 gap-y-2 text-sm">
                    <span className="text-slate-500">ID</span>
                    <span className="text-slate-800 font-mono text-right">{selectedAp.id}</span>
                    
                    <span className="text-slate-500">Model</span>
                    <span className="text-slate-800 text-right">{selectedAp.model}</span>
                    
                    <span className="text-slate-500">Power</span>
                    <div className="flex items-center justify-end gap-2">
                         <span className="text-slate-800">{selectedAp.power} dBm</span>
                    </div>

                    <span className="text-slate-500">Channel</span>
                    <span className="text-slate-800 text-right">{selectedAp.channel}</span>

                    <span className="text-slate-500">Clients</span>
                    <span className="text-slate-800 text-right">0</span>
                 </div>
              </div>
           </div>
        )}
      </div>
    </div>
  );
};

export default FloorPlanEditor;
