import { Wall } from '../types';

/**
 * Detects walls from a floor plan image by calling the Python backend.
 */
export const detectWalls = async (
    imageSrc: string,
    metersPerPixel: number
): Promise<Wall[]> => {
    try {
        const response = await fetch('/api/detect-walls-base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageSrc,
                metersPerPixel,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        return data.walls;
    } catch (error) {
        console.error('Wall detection failed:', error);
        throw error;
    }
};
