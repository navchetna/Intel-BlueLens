# Makefile for Intel-BlueLens Docker container
# Variables
IMAGE_NAME ?= intel-bluelens
IMAGE_TAG ?= latest
REGISTRY ?= ghcr.io/adityakulshrestha
FULL_IMAGE_NAME = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
LOCAL_IMAGE_NAME = $(IMAGE_NAME):$(IMAGE_TAG)
CONTAINER_NAME ?= intel-bluelens
PORT ?= 3003

.PHONY: help
help: ## Show this help message
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║           Intel-BlueLens Docker & Local Dev Makefile          ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 LOCAL DEVELOPMENT (without Docker):"
	@echo "  make local-install   Install npm dependencies"
	@echo "  make local-dev       Start development server (port 3000)"
	@echo "  make local-build     Build for production"
	@echo "  make local-serve     Serve production build locally (port 3003)"
	@echo "  make local-clean     Clean dist and node_modules"
	@echo ""
	@echo "🐳 DOCKER COMMANDS:"
	@echo "  make build           Build the Docker image"
	@echo "  make run             Run container (port $(PORT))"
	@echo "  make dev             Build + run + show logs"
	@echo "  make stop            Stop running container"
	@echo "  make restart         Restart container"
	@echo "  make logs            Show container logs"
	@echo "  make clean           Stop and remove container"
	@echo ""
	@echo "🚀 DEPLOYMENT:"
	@echo "  make deploy          Build, tag, and push to registry"
	@echo "  make deploy-run      Pull and run from registry"
	@echo "  make redeploy        Complete redeploy cycle"
	@echo ""
	@echo "🔧 UTILITIES:"
	@echo "  make test            Build and test container"
	@echo "  make shell           Open shell in running container"
	@echo "  make size            Show image size"
	@echo "  make prune           Clean Docker build cache"
	@echo ""
	@echo "Use 'make <target>' to run a command"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# LOCAL DEVELOPMENT (without Docker)
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: local-install
local-install: ## Install npm dependencies locally
	@echo "📦 Installing npm dependencies..."
	npm install
	@echo "✓ Dependencies installed"

.PHONY: local-dev
local-dev: ## Start local development server
	@echo "🚀 Starting development server..."
	@echo "   ➜ Local: http://localhost:3000"
	npm run dev

.PHONY: local-build
local-build: ## Build the project for production locally
	@echo "🔨 Building for production..."
	npm run build
	@echo "✓ Build complete → ./dist"

.PHONY: local-serve
local-serve: ## Serve the production build locally
	@echo "🌐 Starting production server..."
	@echo "   ➜ Server: http://localhost:3003/intel-bluelens/"
	@echo "   ➜ Root redirect: http://localhost:3003/ → /intel-bluelens/"
	node server.js

.PHONY: local-preview
local-preview: ## Preview production build with Vite
	@echo "👀 Previewing production build..."
	npm run preview

.PHONY: local-clean
local-clean: ## Clean build artifacts and dependencies
	@echo "🧹 Cleaning local build..."
	rm -rf dist node_modules
	@echo "✓ Local build cleaned"

.PHONY: local-lint
local-lint: ## Run TypeScript type checking
	@echo "🔍 Running type check..."
	npm run lint

.PHONY: local-full
local-full: local-clean local-install local-build local-serve ## Full local rebuild and serve

# ═══════════════════════════════════════════════════════════════════════════
# DOCKER COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: build
build: ## Build the Docker image locally
	@echo "Building Docker image: $(LOCAL_IMAGE_NAME)"
	docker build -t $(LOCAL_IMAGE_NAME) .
	@echo "✓ Build complete: $(LOCAL_IMAGE_NAME)"

.PHONY: build-no-cache
build-no-cache: ## Build the Docker image without cache
	@echo "Building Docker image (no cache): $(LOCAL_IMAGE_NAME)"
	docker build --no-cache -t $(LOCAL_IMAGE_NAME) .
	@echo "✓ Build complete: $(LOCAL_IMAGE_NAME)"

.PHONY: run
run: ## Run the container locally
	@echo "Starting container: $(CONTAINER_NAME)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):3003 \
		$(LOCAL_IMAGE_NAME)
	@echo "✓ Container started at http://localhost:$(PORT)"

.PHONY: run-interactive
run-interactive: ## Run the container in interactive mode (foreground)
	@echo "Starting container in interactive mode..."
	docker run --rm \
		--name $(CONTAINER_NAME)-interactive \
		-p $(PORT):3003 \
		$(LOCAL_IMAGE_NAME)

.PHONY: stop
stop: ## Stop the running container
	@echo "Stopping container: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME)
	@echo "✓ Container stopped"

.PHONY: restart
restart: stop run ## Restart the container

.PHONY: logs
logs: ## Show container logs
	docker logs -f $(CONTAINER_NAME)

.PHONY: shell
shell: ## Open a shell in the running container
	docker exec -it $(CONTAINER_NAME) /bin/sh

.PHONY: clean
clean: ## Stop and remove container
	@echo "Cleaning up container: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "✓ Container cleaned"

.PHONY: clean-all
clean-all: clean ## Remove container and images
	@echo "Removing Docker images..."
	-docker rmi $(LOCAL_IMAGE_NAME) 2>/dev/null || true
	-docker rmi $(FULL_IMAGE_NAME) 2>/dev/null || true
	@echo "✓ Images removed"

.PHONY: tag
tag: ## Tag image for registry
	@echo "Tagging image for registry: $(FULL_IMAGE_NAME)"
	docker tag $(LOCAL_IMAGE_NAME) $(FULL_IMAGE_NAME)
	@echo "✓ Image tagged: $(FULL_IMAGE_NAME)"

.PHONY: push
push: tag ## Push image to container registry
	@echo "Pushing image to registry: $(FULL_IMAGE_NAME)"
	docker push $(FULL_IMAGE_NAME)
	@echo "✓ Image pushed successfully"

.PHONY: pull
pull: ## Pull image from container registry
	@echo "Pulling image from registry: $(FULL_IMAGE_NAME)"
	docker pull $(FULL_IMAGE_NAME)
	@echo "✓ Image pulled successfully"

.PHONY: deploy
deploy: build tag push ## Build, tag, and push to registry
	@echo "✓ Deployment complete: $(FULL_IMAGE_NAME)"

.PHONY: deploy-run
deploy-run: pull ## Pull and run from registry
	@echo "Deploying from registry..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):3003 \
		--restart unless-stopped \
		$(FULL_IMAGE_NAME)
	@echo "✓ Container deployed and running at http://localhost:$(PORT)"

.PHONY: test
test: build run ## Build and test the container locally
	@echo "Waiting for container to start..."
	@sleep 3
	@echo "Testing endpoint..."
	@curl -f http://localhost:$(PORT) > /dev/null 2>&1 && echo "✓ Container is responding" || echo "✗ Container test failed"
	@make stop

.PHONY: inspect
inspect: ## Inspect the built image
	docker inspect $(LOCAL_IMAGE_NAME)

.PHONY: size
size: ## Show image size
	@docker images $(LOCAL_IMAGE_NAME) --format "Image: {{.Repository}}:{{.Tag}}\nSize: {{.Size}}\nCreated: {{.CreatedSince}}"

.PHONY: prune
prune: ## Clean up Docker build cache
	@echo "Pruning Docker build cache..."
	docker builder prune -f
	@echo "✓ Build cache pruned"

# Development shortcuts
.PHONY: dev
dev: build run logs ## Quick development cycle: build, run, and show logs

.PHONY: redeploy
redeploy: clean deploy deploy-run ## Complete redeployment: clean, build, push, and run from registry
