import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { RoamingDataPoint } from '../types';
import { User, Activity, SignalHigh } from 'lucide-react';

const RoamingSimulator: React.FC = () => {
  const [data, setData] = useState<RoamingDataPoint[]>([]);
  const [currentSignal, setCurrentSignal] = useState(-55);
  const [roamingStatus, setRoamingStatus] = useState<'Good'|'Smooth'|'Poor'>('Good');

  // Simulate incoming data
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      
      // Random walk for signal strength between -40 and -85
      setData(prev => {
        const lastSignal = prev.length > 0 ? prev[prev.length - 1].signalStrength : -55;
        const change = (Math.random() - 0.5) * 10;
        let newSignal = Math.max(-90, Math.min(-30, lastSignal + change));
        
        // Simulate a roam drop occasionally
        if (Math.random() > 0.95) newSignal -= 15; 

        setCurrentSignal(Math.round(newSignal));
        setRoamingStatus(newSignal > -60 ? 'Smooth' : newSignal > -75 ? 'Good' : 'Poor');

        const newPoint = {
          time: timeStr,
          signalStrength: Math.round(newSignal),
          apId: newSignal > -65 ? 'AP-02' : 'AP-01'
        };

        const newData = [...prev, newPoint];
        if (newData.length > 20) newData.shift();
        return newData;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 w-96 absolute right-8 top-24 z-30">
        <div className="flex justify-between items-start mb-4">
            <div>
                <h3 className="font-bold text-slate-800">Real-Time Roaming</h3>
                <p className="text-xs text-slate-500">Live Client Simulation</p>
            </div>
            <div className={`px-2 py-1 rounded text-xs font-bold ${roamingStatus === 'Smooth' ? 'bg-green-100 text-green-700' : roamingStatus === 'Good' ? 'bg-blue-100 text-blue-700' : 'bg-red-100 text-red-700'}`}>
                {roamingStatus}
            </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-slate-50 p-3 rounded border border-slate-100">
                <div className="flex items-center gap-2 text-slate-500 mb-1 text-xs">
                    <SignalHigh size={14}/> Signal
                </div>
                <div className="text-xl font-bold text-slate-800">{currentSignal} dBm</div>
            </div>
            <div className="bg-slate-50 p-3 rounded border border-slate-100">
                <div className="flex items-center gap-2 text-slate-500 mb-1 text-xs">
                    <Activity size={14}/> Band
                </div>
                <div className="text-xl font-bold text-slate-800">5 GHz</div>
            </div>
        </div>

        <div className="h-32 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="colorSignal" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[-90, -30]} hide />
                    <Tooltip 
                        contentStyle={{ fontSize: '12px', borderRadius: '4px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        itemStyle={{ color: '#1e293b' }}
                    />
                    <Area type="monotone" dataKey="signalStrength" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorSignal)" />
                </AreaChart>
            </ResponsiveContainer>
        </div>
        
        <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <div className="flex items-center gap-2">
                 <div className="w-2 h-2 rounded-full bg-blue-500"></div> Connected to AP-02
            </div>
            <span>-55 dBm target</span>
        </div>
    </div>
  );
};

export default RoamingSimulator;
