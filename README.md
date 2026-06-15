# Intel BlueLens

LLM architecture and hardware execution visualizer for Intel platforms.

Visualizes the gap between abstract neural network architectures and their actual hardware execution paths — mapping model components to Intel CPU (IPEX/oneDNN/AMX), XPU (SYCL/XeTile), and GPU (CUTLASS/FlashAttn) kernels.

---

## Quick Start

### Option 1: Local Development (Recommended for development)

**Prerequisites:** Node.js 18+

```bash
# Using Makefile
make local-install    # Install dependencies
make local-dev        # Start dev server at http://localhost:3000

# Or using npm directly
npm install
npm run dev
```

### Option 2: Local Production Build

```bash
# Build and serve production version
make local-build      # Build for production
make local-serve      # Serve at http://localhost:3003/intel-bluelens/

# Or manual commands
npm run build
node server.js
```

### Option 3: Docker (Recommended for deployment)

```bash
# Quick start with Docker
make build            # Build Docker image
make run              # Run container at http://localhost:3003

# Or use the dev shortcut
make dev              # Build + run + show logs
```

---

## Docker Deployment

### Using Makefile (Recommended)

Build and run locally:
```bash
make build          # Build the Docker image
make run            # Run the container (port 3003)
make logs           # View container logs
```

Quick development cycle:
```bash
make dev            # Build, run, and show logs
```

Deploy to container registry:
```bash
make deploy         # Build, tag, and push to registry
make deploy-run     # Pull from registry and run
```

Other useful commands:
```bash
make help           # Show all available commands
make test           # Build and test the container
make clean          # Stop and remove container
make restart        # Restart the container
```

### Manual Docker Commands

Build the image:
```bash
docker build -t intel-bluelens .
```

Run the container:
```bash
docker run -d --name intel-bluelens -p 3003:3003 intel-bluelens
```

Access the application at `http://localhost:3003`

### Docker Configuration

- **Exposed Port:** 3003
- **Base Path:** `/intel-bluelens/`
- **Public Assets:** Included in build (profiles, traces)
- **Image Type:** Multi-stage build with Node.js Alpine

---

## Production Build

```bash
npm run build
```

Output goes to `dist/`. Serve with any static file host.

```bash
# Preview the production build locally
npm run preview
```

---

## Environment Variables

No API keys required. All data is static / generated client-side.

Optional: set `APP_URL` in `.env.local` if you need self-referential links.

```env
APP_URL=https://your-domain.com
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| React 19 | UI framework |
| Vite | Build tool / dev server |
| Tailwind CSS | Styling |
| motion/react | Animations |
| lucide-react | Icons |

No backend, no database, no external API calls.

