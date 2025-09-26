.PHONY: build run test clean deps

# 构建应用
build:
	go build -o bin/blink_core_server main.go

# 运行应用
run:
	go run main.go

# 运行测试
test:
	go test -v ./...

# 清理构建文件
clean:
	rm -rf bin/
	rm -f *.db

# 安装依赖
deps:
	go mod download
	go mod tidy

# 格式化代码
fmt:
	go fmt ./...

# 代码检查
lint:
	golangci-lint run

# 生成文档
docs:
	godoc -http=:6060

# Docker构建
docker-build:
	docker build -t blink_core_server .

# Docker运行
docker-run:
	docker run -p 8080:8080 blink_core_server
