# blink_core实现

## 功能描述
在blink_core_server项目中实现一个blink_core服务，服务使用go语言编写，实现照片管理的接入功能，如果设计ai相关的功能，blink_core_server会调用blink_ai_server的api来实现，blink_ai_server是一个内部调用的服务， blink_core_server是提供给手机端调用的服务。 协议设计使用http协议，使用json格式进行数据传输。

## 程序要求
*使用gin框架编写
*使用sqlite3数据库
*使用jwt进行token认证
*使用zap进行日志记录
*使用viper进行配置管理
*使用gorm进行数据库操作
*使用redis进行缓存
*目录结构设计好分层：数据层、逻辑层、接入层