# Product Requirements Document: AuraNet Planner

## 1. Executive Summary
AuraNet Planner is a web-based, hardware-accelerated wireless network planning tool designed to be a modern, accessible alternative to industry standards like Hamina and Ekahau. It focuses on speed, ease of use, and 3D visualization.

## 2. Core Features (MVP)

### 2.1 Project Management
- **Create/Save/Load**: JSON-based project files.
- **Settings**: Global project settings (measurement units, default signal profiles).

### 2.2 Floor Plan Management
- **Import**: Support for PNG, JPEG, SVG.
- **Auto-Wall Detection**:
    - Automatically detect walls and doors from raster images.
    - **Algorithm**: Hybrid approach using Hough Transform/LSD (Line Segment Detector) for line extraction and/or lightweight CNN (e.g., U-Net) for semantic segmentation if feasible on client-side.
- **Scaling**: Interactive tool to set scale (draw a line and define length).
- **Opacity/Visibility**: Toggle floor plan visibility.

### 2.3 Environment Modeling (Walls & Obstacles)
- **Drawing Tools**: Line tool, Rectangle tool for walls.
- **Material Library**: Pre-defined materials with attenuation values (e.g., Brick -10dB, Drywall -3dB).
- **3D Properties**: Height, thickness, elevation from floor.

### 2.4 Network Design (Access Points)
- **AP Library**:
    - **Community/Vendor Database**: Support for importing industry-standard antenna patterns (`.msi`, `.ant`).
    - **Specs**: Configurable Tx Power, Antenna Gain, Frequency Bands (2.4/5/6 GHz).
- **Placement**: Drag-and-drop placement on the map.
- **AI Auto-Placement**:
    - **Algorithm**: Genetic Algorithm or Simulated Annealing to find optimal AP locations.
    - **Goal**: Minimize AP count while maintaining target signal strength (e.g., -65dBm) across the coverage area.
- **Configuration**:
    - Channel (2.4GHz, 5GHz, 6GHz).
    - Transmit Power (dBm).
    - Height (m).
    - Orientation (Azimuth, Tilt).

### 2.5 Analysis & Visualization (Heatmaps)
- **Signal Strength (RSSI)**:
    - **Propagation Model**: Multi-Wall Model (MWM) for high accuracy in complex environments.
    - **Fallback**: ITU-R P.1238 for quick estimation or open spaces.
    - **Calculation**:
        - Free Space Path Loss (FSPL).
        - Wall Attenuation: Ray casting to determine wall penetrations.
- **Visualization Modes**:
    - Smooth gradients.
    - Configurable color scales (e.g., -65dBm cutoff).
- **3D View**:
    - Render walls, floor, and APs in 3D.
    - Volumetric heatmap (optional for MVP, but nice to have).

## 3. Technical Requirements

### 3.1 Frontend Architecture
- **Framework**: React 18+ (Vite).
- **Language**: TypeScript.
- **State Management**: Zustand or Redux Toolkit (for complex project state).
- **Graphics Engine**:
    - **2D**: HTML5 Canvas (via Konva.js or raw API) for performant drawing of thousands of walls/rays.
    - **3D**: Three.js (via React-Three-Fiber).

### 3.2 Performance Goals
- Support for 50+ APs and 1000+ wall segments without UI lag.
- Real-time heatmap updates (<100ms latency on change).

## 4. User Interface / User Experience
- **Theme**: Dark mode default, "Cyberpunk/Sci-Fi" aesthetic (AuraNet style).
- **Layout**:
    - **Left Sidebar**: Tools (Select, Wall, AP, Floor).
    - **Right Sidebar**: Properties (Selected Object details, Heatmap settings).
    - **Center**: Infinite canvas workspace.
- **Interactions**: Keyboard shortcuts, scroll-to-zoom, pan.

## 5. Future Scope (Post-MVP)
- Auto-wall detection from images (AI).
- Auto-channel planning.
- Bill of Materials (BOM) export.
- Survey data import (pcap/csv).
