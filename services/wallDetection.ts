import { Wall } from '../types';

/**
 * Configuration for the wall detection algorithm
 */
interface DetectionConfig {
    threshold: number; // 0-255, pixel brightness threshold for "wall"
    minWallLength: number; // pixels
    wallThickness: number; // pixels, used for merging parallel segments
    doorWidthMin: number; // pixels
    doorWidthMax: number; // pixels
}

/**
 * Detects walls from a floor plan image using a Recursive Line Growing (RLG) approach.
 * optimized for orthogonal walls (horizontal/vertical).
 */
export const detectWalls = async (
    imageSrc: string,
    metersPerPixel: number
): Promise<Wall[]> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('Could not get canvas context');

                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const walls = processImage(imageData, metersPerPixel);
                resolve(walls);
            } catch (e) {
                reject(e);
            }
        };
        img.onerror = reject;
        img.src = imageSrc;
    });
};

const processImage = (imageData: ImageData, metersPerPixel: number): Wall[] => {
    const { width, height, data } = imageData;
    const visited = new Uint8Array(width * height); // 0 = unvisited, 1 = visited
    const walls: Wall[] = [];

    // Config based on scale
    const config: DetectionConfig = {
        threshold: 100, // Darker than this is a wall
        minWallLength: 0.5 / metersPerPixel, // 0.5 meters minimum
        wallThickness: 0.2 / metersPerPixel, // ~20cm tolerance
        doorWidthMin: 0.7 / metersPerPixel,
        doorWidthMax: 1.2 / metersPerPixel,
    };

    // Helper to get pixel brightness (0-255)
    const getBrightness = (x: number, y: number) => {
        const i = (y * width + x) * 4;
        // Simple luminance
        return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    };

    // 1. Pre-pass: Binarize and Mark interesting pixels
    // We only care about dark pixels (walls)
    const isWallPixel = (x: number, y: number) => {
        if (x < 0 || x >= width || y < 0 || y >= height) return false;
        return getBrightness(x, y) < config.threshold;
    };

    // 2. Scan for Horizontal Lines
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (visited[y * width + x]) continue;
            if (!isWallPixel(x, y)) continue;

            // Start growing a horizontal line
            let startX = x;
            let endX = x;

            // Grow right
            while (endX < width && isWallPixel(endX, y)) {
                visited[y * width + endX] = 1;
                endX++;
            }

            // Check length
            if (endX - startX > config.minWallLength) {
                // Refine thickness: check if this is just a thick line we've already covered?
                // For simplicity in this MVP, we just take the center line of the thickness
                // But since we mark visited, we naturally thin it out.

                // Add wall
                walls.push({
                    id: `W-Auto-${Date.now()}-${walls.length}`,
                    x1: startX,
                    y1: y,
                    x2: endX,
                    y2: y,
                    thickness: 5, // Default visual thickness
                    material: 'Concrete',
                    attenuation: 15,
                    height: 3,
                    elevation: 0,
                    metadata: { type: 'detected' }
                });
            }
        }
    }

    // 3. Scan for Vertical Lines (similar logic)
    // We reset visited for vertical pass? No, that would double count intersections.
    // Actually, for a robust grid scan, we should do two passes on the original data
    // or handle intersections better.
    // For MVP: Let's do a separate pass on the *original* pixels for vertical, 
    // but we need to be careful not to duplicate the horizontal segments.
    // A simple way is to check if the run is *longer* vertically than horizontally.

    // Let's restart with a better approach:
    // Scan specifically for vertical runs now, ignoring the "visited" state from horizontal
    // BUT, we need to merge them later.
    // Actually, standard RLG usually does: find seed, grow in best direction.

    // Revised approach for Vertical:
    // We'll just scan again. If we generate a wall that overlaps significantly with a horizontal one,
    // it's an intersection. If it's identical, we drop it.

    // For this MVP, let's just scan vertical columns.
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            // We can't easily use 'visited' here if we want to catch corners properly without complex logic.
            // So let's just check pixel value.
            if (!isWallPixel(x, y)) continue;

            let startY = y;
            let endY = y;

            while (endY < height && isWallPixel(x, endY)) {
                endY++;
            }

            if (endY - startY > config.minWallLength) {
                // Check if this is actually a vertical wall or just a slice of a horizontal one.
                // A vertical wall should be "tall and thin".
                // We check neighbors.
                let isVertical = true;
                // Heuristic: if the horizontal run at the midpoint is much longer than width, it's horizontal.
                // But we are doing a simple pass.

                // Let's just add it and filter later.
                walls.push({
                    id: `W-Auto-${Date.now()}-${walls.length}`,
                    x1: x,
                    y1: startY,
                    x2: x,
                    y2: endY,
                    thickness: 5,
                    material: 'Concrete',
                    attenuation: 15,
                    height: 3,
                    elevation: 0,
                    metadata: { type: 'detected' }
                });

                // Skip these pixels to avoid re-processing this line immediately
                y = endY;
            }
        }
    }

    // 4. Post-processing: Merge and Clean
    return mergeWalls(walls, config);
};

const mergeWalls = (walls: Wall[], config: DetectionConfig): Wall[] => {
    // 1. Separate Horizontal and Vertical
    const horizontal = walls.filter(w => Math.abs(w.y1 - w.y2) < 1);
    const vertical = walls.filter(w => Math.abs(w.x1 - w.x2) < 1);

    const merged: Wall[] = [];

    // Helper to merge overlapping collinear segments
    const mergeCollinear = (segments: Wall[], isVertical: boolean) => {
        // Sort by primary coordinate
        segments.sort((a, b) => isVertical ? a.y1 - b.y1 : a.x1 - b.x1);

        // Group by the other coordinate (row or column) with tolerance
        const groups: { [key: number]: Wall[] } = {};
        segments.forEach(w => {
            const key = Math.round(isVertical ? w.x1 : w.y1);
            // Simple clustering could be better, but rounding to nearest pixel is a start
            if (!groups[key]) groups[key] = [];
            groups[key].push(w);
        });

        Object.entries(groups).forEach(([pos, group]) => {
            if (group.length === 0) return;

            // Merge overlapping in this group
            let current = group[0];
            for (let i = 1; i < group.length; i++) {
                const next = group[i];
                // Check overlap or proximity
                const currentEnd = isVertical ? current.y2 : current.x2;
                const nextStart = isVertical ? next.y1 : next.x1;

                if (nextStart <= currentEnd + config.wallThickness) {
                    // Merge
                    if (isVertical) current.y2 = Math.max(current.y2, next.y2);
                    else current.x2 = Math.max(current.x2, next.x2);
                } else {
                    // Gap found - is it a door?
                    const gap = nextStart - currentEnd;
                    if (gap >= config.doorWidthMin && gap <= config.doorWidthMax) {
                        // It's a door! We don't merge, but we could mark it.
                        // For now, just keep them separate walls, which effectively makes a "doorway" (gap).
                    }
                    merged.push(current);
                    current = next;
                }
            }
            merged.push(current);
        });
    };

    mergeCollinear(horizontal, false);
    mergeCollinear(vertical, true);

    // Filter out artifacts (e.g. vertical slices of horizontal walls)
    // If a vertical wall is inside a horizontal wall's thickness, it's likely noise.
    // For MVP, we'll return the merged set.

    return merged;
};
