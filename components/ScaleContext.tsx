import React, { createContext, useContext, useState } from 'react';

interface ScaleContextValue {
  scaleFactor: number;
  setScaleFactor: (value: number) => void;
}

const ScaleContext = createContext<ScaleContextValue | undefined>(undefined);

export const ScaleProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [scaleFactor, setScaleFactor] = useState<number>(1);

  return (
    <ScaleContext.Provider value={{ scaleFactor, setScaleFactor }}>
      {children}
    </ScaleContext.Provider>
  );
};

export const useScale = (): ScaleContextValue => {
  const context = useContext(ScaleContext);
  if (!context) {
    throw new Error('useScale must be used within a ScaleProvider');
  }
  return context;
};
