# Blink Core Server

基于Go语言的照片管理服务，提供照片上传、人脸聚类等功能。

## 技术栈

- **框架**: Gin
- **数据库**: SQLite3 + GORM
- **缓存**: Redis
- **认证**: JWT
- **日志**: Zap
- **配置**: Viper

## 项目结构

```
blink_core_server/
├── main.go                    # 主程序入口
├── go.mod                     # Go模块文件
├── configs/                   # 配置文件目录
│   └── config.yaml           # 配置文件
├── internal/                  # 内部包
│   ├── config/               # 配置管理
│   ├── database/             # 数据库相关
│   │   ├── models.go         # 数据模型
│   │   └── database.go       # 数据库连接
│   ├── handler/              # HTTP处理器
│   │   ├── auth.go           # 认证处理器
│   │   └── photo.go          # 照片处理器
│   ├── middleware/           # 中间件
│   │   ├── jwt.go            # JWT认证中间件
│   │   └── logger.go         # 日志中间件
│   └── service/              # 业务逻辑层
│       ├── auth.go           # 认证服务
│       └── photo.go          # 照片服务
├── pkg/                      # 公共包
│   ├── logger/               # 日志包
│   └── redis/                # Redis包
├── uploads/                  # 上传文件目录
├── Dockerfile               # Docker配置
├── docker-compose.yml       # Docker Compose配置
└── Makefile                 # 构建脚本
```

## 快速开始

### 1. 安装依赖

```bash
make deps
```

### 2. 配置服务

复制配置文件并修改：

```bash
cp configs/config.yaml configs/config.local.yaml
```

编辑 `configs/config.local.yaml` 文件，配置数据库、Redis、AI服务等。

### 3. 运行服务

```bash
# 开发模式
make run

# 或者构建后运行
make build
./bin/blink_core_server
```

### 4. 使用Docker

```bash
# 构建镜像
docker build -t blink_core_server .

# 运行容器
docker run -p 8080:8080 blink_core_server

# 或使用Docker Compose
docker-compose up -d
```

## API接口

### 认证接口

#### 用户注册
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "password123"
}
```

#### 用户登录
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "testuser",
  "password": "password123"
}
```

#### 刷新令牌
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "token": "your_jwt_token"
}
```

### 照片管理接口

#### 上传照片
```http
POST /api/v1/photos/upload
Authorization: Bearer your_jwt_token
Content-Type: multipart/form-data

file: [照片文件]
```

#### 获取照片列表
```http
GET /api/v1/photos?page=1&page_size=20
Authorization: Bearer your_jwt_token
```

#### 获取单张照片
```http
GET /api/v1/photos/{id}
Authorization: Bearer your_jwt_token
```

#### 删除照片
```http
DELETE /api/v1/photos/{id}
Authorization: Bearer your_jwt_token
```

#### 对照片进行人脸聚类
```http
POST /api/v1/photos/{id}/cluster
Authorization: Bearer your_jwt_token
```

### 聚类管理接口

#### 获取聚类结果
```http
GET /api/v1/clusters
Authorization: Bearer your_jwt_token
```

#### 获取单个聚类
```http
GET /api/v1/clusters/{id}
Authorization: Bearer your_jwt_token
```

#### 重新聚类
```http
POST /api/v1/clusters/recluster
Authorization: Bearer your_jwt_token
```

## 配置说明

### 服务器配置
- `port`: 服务端口，默认8080
- `mode`: 运行模式，debug/release

### 数据库配置
- `dsn`: SQLite数据库文件路径

### Redis配置
- `addr`: Redis服务器地址
- `password`: Redis密码
- `db`: Redis数据库编号

### JWT配置
- `secret`: JWT签名密钥
- `expire_time`: 令牌过期时间（秒）

### AI服务配置
- `ai_base_url`: AI服务基础URL

### 日志配置
- `level`: 日志级别，debug/info/warn/error
- `format`: 日志格式，json/text

## 开发指南

### 添加新的API接口

1. 在 `internal/handler/` 中添加处理器
2. 在 `internal/service/` 中添加业务逻辑
3. 在 `main.go` 中注册路由

### 添加新的数据模型

1. 在 `internal/database/models.go` 中定义模型
2. 运行 `go run main.go` 自动迁移数据库

### 测试

```bash
# 运行所有测试
make test

# 运行特定包的测试
go test ./internal/service/...

# 运行测试并显示覆盖率
go test -cover ./...
```

## 部署

### 生产环境部署

1. 修改配置文件，设置生产环境参数
2. 构建应用：`make build`
3. 使用Docker部署：`docker-compose up -d`

### 监控和日志

- 应用日志输出到标准输出
- 可以通过Docker日志查看：`docker logs blink_core_server`
- 建议使用ELK或类似工具收集和分析日志

## 注意事项

1. 确保AI服务（blink_ai_server）正在运行
2. 确保Redis服务正在运行
3. 上传目录需要有写权限
4. 生产环境请修改JWT密钥
5. 定期备份数据库文件

## 许可证

MIT License
