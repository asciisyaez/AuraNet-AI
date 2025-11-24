import React, { useState, useRef, useEffect } from 'react';
import { AccessPoint, Wall } from '../types';
import { INITIAL_APS, INITIAL_WALLS, HARDWARE_TOOLS, ENV_TOOLS } from '../constants';
import HeatmapCanvas from './HeatmapCanvas';
import { Wifi, Router, Square, Trash2, Edit3, Loader2, Info, Image, Eye, EyeOff, Ruler } from 'lucide-react';
import { getOptimizationSuggestions } from '../services/geminiService';
import { useScale } from './ScaleContext';

const FloorPlanEditor: React.FC = () => {
  const [aps, setAps] = useState<AccessPoint[]>(INITIAL_APS);
  const [walls, setWalls] = useState<Wall[]>(INITIAL_WALLS);
  const [selectedApId, setSelectedApId] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);
  const { scaleFactor, setScaleFactor } = useScale();

  type Point = { x: number; y: number };

  const [floorPlanImage, setFloorPlanImage] = useState<string | null>(null);
  const [imageOpacity, setImageOpacity] = useState(0.6);
  const [showImageLayer, setShowImageLayer] = useState(true);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationStart, setCalibrationStart] = useState<Point | null>(null);
  const [calibrationEnd, setCalibrationEnd] = useState<Point | null>(null);
  const [pixelLength, setPixelLength] = useState<number | null>(null);
  const [realLengthInput, setRealLengthInput] = useState('');

  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const editorRef = useRef<HTMLDivElement>(null);

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

  const getRelativePosition = (e: React.MouseEvent): Point | null => {
    if (!editorRef.current) return null;
    const rect = editorRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!isCalibrating) return;
    const point = getRelativePosition(e);
    if (!point) return;

    if (!calibrationStart) {
      setCalibrationStart(point);
      setCalibrationEnd(null);
      setPixelLength(null);
    } else {
      setCalibrationEnd(point);
      const distance = Math.hypot(point.x - calibrationStart.x, point.y - calibrationStart.y);
      setPixelLength(distance);
      setIsCalibrating(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedTypes = ['image/png', 'image/jpeg'];
    if (!allowedTypes.includes(file.type)) {
      alert('Please upload a PNG or JPEG file.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setFloorPlanImage(reader.result);
        setShowImageLayer(true);
      }
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    if (isCalibrating) {
      setCalibrationStart(null);
      setCalibrationEnd(null);
      setPixelLength(null);
    }
  }, [isCalibrating]);

  const applyCalibration = () => {
    if (!pixelLength) return;
    const realLength = parseFloat(realLengthInput);
    if (Number.isNaN(realLength) || realLength <= 0) return;

    setScaleFactor(realLength / pixelLength);
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

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Floor Plan</h3>
          <div className="space-y-3 text-sm text-slate-700">
            <label className="block text-xs font-semibold text-slate-600">Upload (PNG/JPEG)</label>
            <label className="flex items-center gap-2 px-3 py-2 border border-dashed border-slate-300 rounded-md bg-slate-50 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors">
              <Image size={16} className="text-slate-500" />
              <span className="text-xs text-slate-600">Choose file</span>
              <input type="file" accept="image/png,image/jpeg" className="hidden" onChange={handleImageUpload} />
            </label>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-slate-700">
                {showImageLayer ? <Eye size={14} className="text-blue-600" /> : <EyeOff size={14} className="text-slate-400" />}
                <span>Show floor plan</span>
              </div>
              <input
                type="checkbox"
                className="rounded text-blue-600 focus:ring-blue-500"
                checked={showImageLayer}
                onChange={(e) => setShowImageLayer(e.target.checked)}
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
                <span>Opacity</span>
                <span>{Math.round(imageOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={imageOpacity}
                onChange={(e) => setImageOpacity(parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-500 uppercase mb-3">Calibration</h3>
          <p className="text-xs text-slate-600 leading-snug mb-2">
            Draw a reference line on the canvas and enter its real-world length to calibrate measurements.
          </p>
          <button
            onClick={() => setIsCalibrating(!isCalibrating)}
            className={`w-full flex items-center justify-center gap-2 px-3 py-2 text-sm rounded-md border transition-colors ${isCalibrating ? 'border-blue-500 text-blue-700 bg-blue-50' : 'border-slate-200 text-slate-700 hover:border-blue-400 hover:bg-blue-50'}`}
          >
            <Ruler size={16} /> {isCalibrating ? 'Click two points...' : 'Start calibration'}
          </button>
          {pixelLength && (
            <div className="mt-2 text-xs text-slate-600">
              Pixel distance: <span className="font-semibold text-slate-800">{pixelLength.toFixed(1)} px</span>
            </div>
          )}
          <div className="mt-3 space-y-2">
            <label className="text-xs font-semibold text-slate-600">Reference length (meters)</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={realLengthInput}
                onChange={(e) => setRealLengthInput(e.target.value)}
                placeholder="e.g. 10"
                className="flex-1 px-3 py-2 border border-slate-200 rounded-md text-sm focus:ring-2 focus:ring-blue-100 focus:outline-none"
              />
              <button
                onClick={applyCalibration}
                disabled={!pixelLength}
                className="px-3 py-2 text-sm bg-blue-600 text-white rounded-md disabled:opacity-50"
              >
                Set
              </button>
            </div>
            <div className="text-xs text-slate-500">
              Current scale: <span className="font-semibold text-slate-800">{scaleFactor.toFixed(3)}</span> units per pixel
            </div>
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
          onClick={handleCanvasClick}
        >
          {/* Grid Background */}
          <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>

          {/* Floor Plan Image */}
          {floorPlanImage && showImageLayer && (
            <img
              src={floorPlanImage}
              alt="Floor plan"
              className="absolute inset-0 w-full h-full object-contain z-[1] pointer-events-none"
              style={{ opacity: imageOpacity }}
            />
          )}

          {/* Heatmap Layer */}
          <HeatmapCanvas
             aps={aps}
             walls={walls}
             width={800}
             height={600}
             show={showHeatmap}
          />

          {isCalibrating && (
            <div className="absolute top-3 left-3 z-30 bg-blue-600 text-white text-xs px-3 py-1 rounded-full shadow-sm">
              Calibration: click two points to measure
            </div>
          )}

          {(calibrationStart || calibrationEnd) && (
            <svg className="absolute inset-0 z-30 pointer-events-none" width={800} height={600} viewBox="0 0 800 600">
              {calibrationStart && (
                <circle cx={calibrationStart.x} cy={calibrationStart.y} r={5} fill="#2563eb" opacity={0.9} />
              )}
              {calibrationStart && calibrationEnd && (
                <>
                  <line x1={calibrationStart.x} y1={calibrationStart.y} x2={calibrationEnd.x} y2={calibrationEnd.y} stroke="#2563eb" strokeWidth={3} strokeDasharray="6 4" />
                  <circle cx={calibrationEnd.x} cy={calibrationEnd.y} r={5} fill="#2563eb" opacity={0.9} />
                </>
              )}
            </svg>
          )}

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
