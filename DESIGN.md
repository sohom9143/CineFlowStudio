---
name: Kinetic Neural Studio
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353942'
  surface-container-lowest: '#0a0e16'
  surface-container-low: '#181c24'
  surface-container: '#1c2028'
  surface-container-high: '#262a33'
  surface-container-highest: '#31353e'
  on-surface: '#dfe2ee'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#dfe2ee'
  inverse-on-surface: '#2c3039'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#c0c1ff'
  on-tertiary: '#1000a9'
  tertiary-container: '#8083ff'
  on-tertiary-container: '#0d0096'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#e1e0ff'
  tertiary-fixed-dim: '#c0c1ff'
  on-tertiary-fixed: '#07006c'
  on-tertiary-fixed-variant: '#2f2ebe'
  background: '#0f131c'
  on-background: '#dfe2ee'
  surface-variant: '#31353e'
typography:
  headline-xl:
    fontFamily: Plus Jakarta Sans
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
  headline-xl-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
  title-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  title-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  code-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 14px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-2xs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-base: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
  dock-padding: 0.625rem
  panel-rail-width: 3.5rem
  inspector-width: 20rem
  media-drawer-height: 16rem
  gutter-canvas: 0.75rem
---

## Brand & Style

This design system establishes a high-performance, studio-grade creative environment tailored for digital artists, technical directors, VFX specialists, and generative AI creators. Drawing inspiration from professional node graphs, nonlinear video editors (NLEs), and next-generation generative rendering engines, the visual language balances ultra-deep contrast with precision functional instrumentation.

The design movement merges **Technical Glassmorphism** with **Precision Cyber-Minimalism**:
- Pitch-black and charcoal canvases eliminate ocular fatigue during extended generation and node-tinkering workflows.
- Dynamic spectral accents—ranging from electric violet to cryogenic cyan—signal real-time computational states, active diffusion passes, model compilation, and selected parameter fields.
- Surfaces present razor-thin hairline borders with subtle directional illumination, simulating precision-machined obsidian instruments and computational control rigs.
- Visual density is calibrated to maximize canvas area while keeping multi-parameter generation docks, media drawers, and timeline rails accessible without layout disruption.

## Colors

The color palette leverages an ultra-deep charcoal architecture layered with spectral photonics for generative feedback and control fidelity.

### Foundational Palette
- **Canvas Base (`#0B0F17`):** The ground infinite canvas and primary viewport void.
- **Surface Elevation 1 (`#111827`):** Side panels, media asset drawers, and static tool docks.
- **Surface Elevation 2 (`#1A2234`):** Floating inspector heads, popovers, node enclosures, and modal surfaces.
- **Surface Elevation 3 (`#232D42`):** Elevated control handles, active node titles, and hover plates.

### Accent & Emission Spectrum
- **Primary Violet (`#8B5CF6`):** The master generative driver. Designates primary creation triggers, diffusion generation runs, keyframe locks, and hero prompts.
- **Secondary Cyan (`#06B6D4`):** Real-time telemetry, camera pathing, motion vectors, output resolution parameters, and audio-reactive tracks.
- **Tertiary Indigo (`#6366F1`):** Logic node pipelines, ComfyUI-style tensor links, routing cables, and neural weight variables.

### Functional States & Ghost Lines
- **Subtle Surface Border (`rgba(255, 255, 255, 0.08)`):** Foundational 1px boundary for panel division.
- **Active Node Highlight (`rgba(139, 92, 246, 0.4)`):** Glow stroke applied to selected processing containers.
- **System Success (`#10B981`):** Render complete, frame cache cached.
- **System Warning (`#F59E0B`):** VRAM threshold caution, token quota threshold reached.
- **System Error (`#EF4444`):** Generation pipeline failure, unlinked node socket.

## Typography

Typography establishes a strict tripartite hierarchy:
1. **Plus Jakarta Sans (Headlines & Mode Selectors):** Geometric precision with modern open counters; conveys cutting-edge technical clarity across interface landmarks and workspace modes.
2. **Inter (Interface & Prompt Inputs):** High x-height and neutral geometry designed for effortless scanning of prompt text, nested parameters, and asset metadata.
3. **JetBrains Mono (Telemetry & Computational Nodes):** Monospaced engine metrics, seeds, latent coordinates, camera angles (FOV/Roll/Pan), execution times, and token cost indicators.

Text rendering must enforce `-webkit-font-smoothing: antialiased` across all platforms. Labels rendered in JetBrains Mono use uppercase letterforms with tracking (`letter-spacing: 0.08em`) to guarantee quick recognition in dense parameter arrays.

## Layout & Spacing

The workstation uses a non-blocking docking architecture centered around an infinite or view-locked viewport canvas. The viewport is enclosed by modular chrome:

- **Left Precision Rail (56px fixed):** Vertical tool selector, node palette, brush/mask tools, workflow graph switcher.
- **Right Inspector Panel (320px fluid/collapsible):** Multi-modal conditioning parameters (CFG scale, sampling steps, motion bucket, seed lock, LoRA weights).
- **Floating Prompt Generation Dock:** Grounded at bottom-center of canvas, floating 24px above viewport boundary, expanding contextually without shifting the canvas camera.
- **Bottom Media Drawer (Collapsible, 256px default):** Filmstrip generation history, asset cache, character turnaround library, render queue.

### Responsiveness & Reflow
- **Desktop (>= 1280px):** Simultaneous open state for timeline/media drawer, inspector, and tool rail around the primary viewport.
- **Tablet (768px - 1279px):** Inspector and media drawer collapse into overlay drawer flyouts triggered via HUD buttons. Prompt dock stays pinned to the bottom.
- **Mobile (< 768px):** Swapping into mobile preview mode; canvas stays prioritized with full-screen sheet modals for prompt composition and node execution graphs.

## Elevation & Depth

Depth is established through multi-layered dark glass, atmospheric luminescence, and directional edge illumination rather than heavy drop shadows.

### Elevation Hierarchy
- **Canvas Base (Level 0):** Pure `#0B0F17` backdrop with optional dynamic rendering grid (dots or isometric lines at 5% opacity).
- **Embedded Rails & Drawers (Level 1):** `#111827` at 94% opacity with `backdrop-filter: blur(16px)` and a subtle 1px border (`rgba(255, 255, 255, 0.07)`).
- **Floating HUDs & Control Pods (Level 2):** `#1A2234` at 85% opacity with `backdrop-filter: blur(24px)`. Soft ambient glow: `box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 1px 1px rgba(255, 255, 255, 0.1)`.
- **Selected Nodes & Active Diffusion Runs (Level 3):** Border glow featuring a dual-ring falloff: `box-shadow: 0 0 0 1px #8B5CF6, 0 0 20px rgba(139, 92, 246, 0.25)`.
- **Modals & Command Palette (Level 4):** Suspended above canvas with deep backing scrim: `background: rgba(11, 15, 23, 0.8)` with full blur (32px), edged with cyber cyan hairline highlights on header boundaries.

## Shapes

The design system employs a refined **Soft (Level 1)** structural rounding language. Precision tools require compact geometry:

- **Micro-Controls (Buttons, Inputs, Sliders, Badges):** `4px` (`0.25rem`) border-radius.
- **Node Boxes, Dropdowns, Floating Pods:** `8px` (`0.5rem`) border-radius.
- **Floating Prompters & Master Modals:** `12px` (`0.75rem`) border-radius.
- **Connector Sockets & Status Orbs:** Circular (`9999px`) for clear continuous flow connections.

This tight corner geometry reinforces the impression of technical instrumentation and hardware-accelerated software.

## Components

### Buttons & Action Triggers
- **Primary ("Generate" / "Queue Run"):** Gradient fill from `#8B5CF6` to `#6366F1` with an inner white hairline highlight `inset 0 1px 0 rgba(255, 255, 255, 0.2)`. Hover shifts luminance upward, active triggers a subtle 0.98 scale compress.
- **Secondary (Tool Selector, Preset Toggles):** Background `#1A2234`, border 1px `rgba(255, 255, 255, 0.08)`, color `#E2E8F0`. Hover triggers border transition to `rgba(6, 182, 212, 0.5)` with `#06B6D4` text tint.
- **Ghost/Icon Action:** Transparent surface, hover fills with `rgba(255, 255, 255, 0.05)`, color `#94A3B8` resting, `#FFFFFF` active.

### Floating Prompt Generation Dock
- Centered horizontally, styled with 16px backdrop blur, containing:
  - Multi-line autosizing textarea with no outline, resting on transparent canvas.
  - Quick-tag pills for model checkpoint, LoRA injection, and aspect ratio selector.
  - Integrated circular generation progress ring rendered in neon cyan.
  - JetBrains Mono token counter and cost estimator tag in the bottom right corner.

### Chips & Model Badges
- Height 22px, `rounded-sm` (4px), font JetBrains Mono (10px).
- Inactive: Background `rgba(255, 255, 255, 0.04)`, border `rgba(255, 255, 255, 0.08)`, text `#94A3B8`.
- Active: Background `rgba(139, 92, 246, 0.15)`, border `rgba(139, 92, 246, 0.4)`, text `#C4B5FD`.

### Precision Sliders & Steppers
- Scrubbable number fields featuring monospaced text readouts.
- Slider track: 4px height, background `#111827`, fill `#06B6D4`.
- Thumb: 12px height × 6px width vertical pill with vertical grip etchings.

### Node Graph Containers (Agent/Flow Elements)
- Header band with socket pin indicators (input left, output right).
- Header carries color accent based on function: Violet for Diffusion, Cyan for Latent/Image Ops, Indigo for Conditioners/Prompts.
- Body displays active thumbnail previews, seed value locked/unlocked toggles, and step progress indicators.

### Character Avatar & Media Asset Library
- Compact grid cards (1:1 or 9:16 aspect ratio).
- Hover reveals quick-action drawer: "Use as Character Reference (IP-Adapter)", "Extend Video", "Extract Depth Map".
- Active item marked by a 1.5px continuous cyan border and corner status checkmark.