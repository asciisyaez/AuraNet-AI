import { Wall } from '../types';

export interface WallDetectionResult {
    walls: Wall[];
    preview?: {
        overlay?: string;
        wall_count: number;
        processing_ms?: number;
    };
}

export interface DetectionProgress {
    stage: string;
    percent: number;
    message: string;
}

/**
 * Detects walls from a floor plan image by calling the Python backend.
 * Supports real-time progress updates via SSE.
 */
export const detectWalls = async (
    imageSrc: string,
    metersPerPixel: number,
    onProgress?: (progress: DetectionProgress) => void
): Promise<WallDetectionResult> => {
    // Generate a unique session ID for progress tracking
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Start listening to progress if callback provided
    let eventSource: EventSource | null = null;
    
    if (onProgress) {
        eventSource = new EventSource(`/api/detection-progress/${sessionId}`);
        
        eventSource.onmessage = (event) => {
            try {
                const progress = JSON.parse(event.data) as DetectionProgress;
                onProgress(progress);
            } catch (e) {
                console.error('Error parsing progress:', e);
            }
        };
        
        eventSource.onerror = () => {
            eventSource?.close();
        };
    }
    
    try {
        const response = await fetch('/api/detect-walls-base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageSrc,
                metersPerPixel,
                sessionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();

        return {
            walls: data.walls ?? [],
            preview: data.preview
                ? {
                    overlay: data.preview.overlay,
                    wall_count: data.preview.wall_count ?? 0,
                    processing_ms: data.preview.processing_ms,
                }
                : undefined,
        };
    } finally {
        // Close the event source after a short delay to catch final messages
        if (eventSource) {
            setTimeout(() => eventSource?.close(), 500);
        }
    }
};
