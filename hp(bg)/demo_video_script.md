# ATHERA : Chandrayaan-2 Cross-Sensor Image Correspondence
## Full Product Walkthrough & Demo Video Script
**Target Duration:** ~2:45 – 3:15 minutes  
**Format:** Screen Recording + Voiceover Narration  
**Problem Statement:** ISRO / SIH 26166 — Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

---

## 🎬 Pre-Recording Checklist & Setup
1. **Screen Resolution:** 1920×1080 (1080p Full HD) or 2560×1440.
2. **Browser:** Chrome / Brave in Fullscreen (`F11` or `Cmd+Shift+F`), zoom level reset to 100% (`Cmd+0`).
3. **Services Active:**
   - FastAPI Backend: `http://localhost:8000` (healthy with 9 triplets loaded)
   - Next.js Console: `http://localhost:3000`
4. **Recording Tip:** Move your mouse smoothly and deliberately. Hover over interactive chips, badges, and buttons for 1–2 seconds before clicking.

---

## ⏱️ Video Timeline Overview

| Section | Timestamp | Focus / Feature |
| :--- | :--- | :--- |
| **Scene 1** | `0:00 - 0:25` | Operator Authentication Gateway & Live Background |
| **Scene 2** | `0:25 - 0:55` | Hero Mission Control HUD & Problem Overview |
| **Scene 3** | `0:55 - 1:40` | Mission Console: Dashboard QA & Sub-Pixel Registration |
| **Scene 4** | `1:40 - 2:05` | Linked Cursor Multi-Sensor Synchronizer |
| **Scene 5** | `2:05 - 2:25` | Planetary 3D Map & Lunar Orbit Selenodesy |
| **Scene 6** | `2:25 - 2:45` | Archive Modals (Dossier, Vault & Mathematical Theory) |
| **Scene 7** | `2:45 - 3:05` | Ingest Pipeline & Closing Takeaway |

---

## 🎙️ Scene-by-Scene Script

---

### SCENE 1: Operator Authentication Gateway (`0:00 - 0:25`)
**Visual Action:**
- Start recording on the **Login Page** (`http://localhost:3000`).
- Let the viewer appreciate the animated lunar background video (`bgsih.mp4`) and the retro workstation window.
- Move cursor smoothly over the `ISRO SAC // SECURE GATEWAY` tag and the `PORTAL SECURED` pulsing LED.
- Click the **Register** tab briefly, then switch back to **Sign In**.
- Click the **"⚡ Pilot Operator Fast-Pass"** button.
- A brief success state flashes, smoothly transitioning into the Hero Page.

**Voiceover (VO):**
> *"Welcome to ATHERA — an end-to-end photogrammetric cross-sensor correspondence platform developed for ISRO Problem Statement 26166.*  
> *Here at the operator gateway, authenticated access is secured via 256-bit encrypted JWT sessions. With our one-click flight operator fast-pass, we enter the mission control environment immediately."*

---

### SCENE 2: Hero Mission Control HUD (`0:25 - 0:55`)
**Visual Action:**
- Land on the **Hero Page**. The retro terminal HUD sits centered against the atmospheric lunar orbital backdrop.
- Hover over the top bar: `C2 CHANDRAYAAN-2`, `SIH26166 // ISRO SAC`, and the live telemetry chips.
- Move down across the 4 key telemetry cards:
  - Hover on **Optical GSD (0.25 m/px)**
  - Hover on **Sub-Pixel Fit (< 0.28 px)**
  - Hover on **Scale Invariant (20× – 320×)**
  - Hover on **Planetary CRS (Moon2000)**
- Click on the primary **"Mission Dashboard >>"** button (or **"Launch Console ↗"** in the top-right).

**Voiceover (VO):**
> *"At the center of our console is the Mission Control HUD. ATHERA tackles one of the toughest challenges in planetary remote sensing: registering Chandrayaan-2's ultra-high-resolution 0.25-meter OHRC images with 5-meter TMC-2 stereoscopic strips and 80-meter hyperspectral IIRS data.*  
> *Under extreme lunar shadow inversions and massive physical scale gaps, traditional deep learning networks fail. ATHERA replaces them with mathematically grounded Phase Congruency and Channel Features of Oriented Gradients. Let’s launch the live Mission Dashboard."*

---

### SCENE 3: Mission Console — Dashboard QA & Quality Gates (`0:55 - 1:40`)
**Visual Action:**
- Arrive at the **Mission Dashboard** (`Dashboard QA` view).
- In the left sidebar, show the region list. Click through `region_001`, then `region_003`, then back to `region_001`.
- In the main Arena panel:
  - Show the dual sensor inspection panels: **OHRC (0.25m Moving)** on the left, **TMC-2 (5.0m Reference)** on the right.
  - Hover over the tie-points / correspondence vectors rendered between the images.
  - Switch the reference toggle from **TMC-2** to **NASA LRO NAC** to demonstrate lunar basemap reference matching on real CDRs.
  - Scroll down or inspect the **Registration Telemetry Metrics Panel**:
    - Point out **Fit RMSE: 1.27 px**
    - Show **Inlier Ratio & Correspondence Count**
    - Highlight the **4 Quality Gates** (Count Gate, RANSAC Verification, Conditioning Check, and Spatial Uniformity Gate) confirming green `PASSED` status.

**Voiceover (VO):**
> *"Here in the core Mission Console, we see live data across all eight flight regions.*  
> *On the left, our primary moving sensor, OHRC. On the right, the reference sensor — switchable between Chandrayaan-2 TMC-2 and NASA LRO NAC Narrow Angle Camera CDRs.*  
> *Notice the tie-point correspondences: unlike black-box models that cluster points on single crater rims, ATHERA employs pre-match spatial suppression and 10x10 density budgeting to ensure true spatial dispersion across lunar craters.*  
> *Crucially, our system upholds strict scientific integrity through four deterministic Quality Gates. If a transform is ill-conditioned or lacks geometric consensus, it fails cleanly with zero fabricated points."*

---

### SCENE 4: Linked Cursor Multi-Sensor Synchronizer (`1:40 - 2:05`)
**Visual Action:**
- In the left sidebar Menu Explorer, click on **"⊙ Linked Cursor"**.
- The view switches to the 3-sensor synchronized inspection canvas (OHRC, TMC-2, IIRS).
- Hover your mouse over the OHRC image and move it smoothly across a crater:
  - Watch the crosshair cursor simultaneously track across the TMC-2 panel and the IIRS overlay with zero latency.
  - Zoom in and out using the scroll wheel or toolbar controls.
  - Highlight the coordinate and scale ratio readout at the bottom.

**Voiceover (VO):**
> *"Next, we switch to the Linked Cursor synchronizer.*  
> *Here, OHRC, TMC-2, and hyperspectral IIRS crops are tied together in physical Moon2000 selenodesy.*  
> *As the operator navigates a geological feature in the 25-centimeter OHRC view, the cursor dynamically projects through the derived homography matrix into the 5-meter TMC and 80-meter IIRS frames with sub-pixel forward-backward tracking precision.*  
> *This enables planetary geologists to cross-reference mineral spectral signatures with decimeter-scale boulder geology instantaneously."*

---

### SCENE 5: Planetary 3D Map & Orbit Selenodesy (`2:05 - 2:25`)
**Visual Action:**
- In the left sidebar Menu Explorer, click on **"☵ Planetary Map"**.
- Show the interactive 3D Moon Globe / Selenodetic Basemap.
- Rotate the lunar sphere slightly to show the orbital track footprint bounding boxes.
- Click on one of the region pins/polygons on the globe to show its bounding coordinates (North/South latitude, East/West longitude).

**Voiceover (VO):**
> *"In the Planetary Map view, each ingested triplet is localized onto the lunar selenodetic globe according to official ISRO PDS4 corner coordinates.*  
> *Every dataset respects planetary IAU Moon2000 coordinates, ensuring true global spatial provenance across polar and equatorial orbits."*

---

### SCENE 6: Archive Modals — Vault, Dossiers & Theory (`2:25 - 2:45`)
**Visual Action:**
- Click the **"🗄️ Archive Vault"** button in the left sidebar (or top bar).
  - Show the retro modal window displaying all 8 regions with payload filters: **All**, **OHRC**, **TMC**, **IIRS**, **LRO NAC**.
  - Type `001` into the search bar to filter instantly.
  - Close the modal with `✕` or `Esc`.
- Click on **"Theory // Math Model"** in the sidebar:
  - Scroll smoothly through the mathematical formulation: Log-Gabor Phase Congruency, CFOG channel tensors, and closed-loop triplet consistency ($H_{OT} \cdot H_{TI} \cdot H_{IO} \approx I$).
  - Close the modal.

**Voiceover (VO):**
> *"The Archive Vault gives flight operators full access to raw and registered GeoTIFF rasters, difference maps, and photometric overlap metrics.*  
> *Meanwhile, our built-in Theory Dossier documents the complete mathematical foundations — detailing our coarse-to-fine scale pyramid, DEM relief displacement compensation, and closed-loop circular consistency checks."*

---

### SCENE 7: Ingest Pipeline & Closing (`2:45 - 3:05`)
**Visual Action:**
- Click on **"⚡ Ingest & Prepare"** in the sidebar or top navigation.
- The `/ingest` page opens with the Drag & Drop Zone.
- Point to the dropzone showing support for `.zip`, `.tar.gz`, and raw PDS4 `.xml`/`.lbl` bundles.
- Show the **Ingest Results Table** at the bottom with processed jobs.
- Click **"← Back to Console"** or return to the Hero Page.

**Voiceover (VO):**
> *"Finally, ATHERA is ready for live operational deployment through the Ingest Pipeline, capable of ingesting raw PRADAN PDS4 bundles, extracting metadata, and producing registered GeoTIFFs autonomously.*  
> *Scalable, verifiable, and strictly grounded in photogrammetric truth — ATHERA delivers precision lunar image correspondence for Chandrayaan-2 and future missions. Thank you."*

---

## 💡 Practical Recording Tips for the Presenter
- **Tone:** Confident, scientific, clear, and professional.
- **Audio:** Record using a good USB/cardioid mic with minimal background noise.
- **Mouse Dynamics:** Smooth linear motions. Avoid rapid shaking or accidental misclicks.
- **Editing:** If using video editing software (Premiere, DaVinci Resolve, CapCut, iMovie), cut the transitions right on the voiceover cue words like *"Next, we switch to..."* or *"Let's launch the live Mission Dashboard."*
