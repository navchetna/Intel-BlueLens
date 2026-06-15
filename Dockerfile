# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy all source files
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:20-alpine

WORKDIR /app

# Copy package files for production dependencies
COPY package.json package-lock.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy built assets and server
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY server.js ./

# Expose port 3003
EXPOSE 3003

# Start the Express server
CMD ["node", "server.js"]
