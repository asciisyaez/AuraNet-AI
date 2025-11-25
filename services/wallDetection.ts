import { WallDetectionMode, WallDetectionResult } from '../types';

/**
 * Detects walls from a floor plan image by calling the Python backend.
 */
export const detectWalls = async (
    imageSrc: string,
    metersPerPixel: number,
    mode: WallDetectionMode = 'balanced'
): Promise<WallDetectionResult> => {
    try {
        const response = await fetch('/api/detect-walls-base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageSrc,
                metersPerPixel,
                mode,
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
                    mode: data.preview.mode,
                    wallCount: data.preview.wall_count ?? data.preview.wallCount ?? 0,
                    processingMs: data.preview.processing_ms ?? data.preview.processingMs,
                }
                : undefined,
            diagnostics: data.diagnostics
                ? {
                    edgePixelRatio: data.diagnostics.edge_pixel_ratio ?? data.diagnostics.edgePixelRatio,
                    rawSegments: data.diagnostics.raw_segments ?? data.diagnostics.rawSegments,
                    mergedSegments: data.diagnostics.merged_segments ?? data.diagnostics.mergedSegments,
                    gapClosures: data.diagnostics.gap_closures ?? data.diagnostics.gapClosures,
                    notes: data.diagnostics.notes,
                }
                : undefined,
        };
    } catch (error) {
        console.error('Wall detection failed:', error);
        throw error;
    }
};
