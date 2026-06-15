# Intel BlueLens

LLM architecture and hardware execution visualizer for Intel platforms.

Visualizes the gap between abstract neural network architectures and their actual hardware execution paths — mapping model components to Intel CPU (IPEX/oneDNN/AMX), XPU (SYCL/XeTile), and GPU (CUTLASS/FlashAttn) kernels.

## Quick Start

**Prerequisites:** Node.js 18+

```bash
npm install
npm run dev
```


## Deployment
# Intel BlueLens — Deployment Guide

## Local Development

```bash
npm install
npm run dev
```

App runs at `http://localhost:5173` by default.

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

## Docker

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

```bash
docker build -t intel-bluelens .
docker run -p 8080:80 intel-bluelens
```

---

## Environment Variables

No API keys required. All data is static / generated client-side.

Optional: set `APP_URL` in `.env.local` if you need self-referential links.

```env
APP_URL=https://your-domain.com
```

---

## Static Hosting (Nginx)

For client-side routing, add a fallback rule:

```nginx
location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| React 18 | UI framework |
| Vite | Build tool / dev server |
| Tailwind CSS | Styling |
| motion/react | Animations |
| lucide-react | Icons |

No backend, no database, no external API calls.

