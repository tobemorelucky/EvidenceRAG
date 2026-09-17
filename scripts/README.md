# EvidenceRAG 本地 Demo 脚本

以下命令可以从项目根目录、`scripts` 目录或其他当前目录运行。脚本会根据自身位置定位并切换到项目根目录。

## 一键启动

```bat
scripts\start_rag.bat
```

启动脚本将依次：

1. 检查 Docker CLI 与 Docker daemon；必要时尝试启动 Docker Desktop，并最多等待 120 秒。
2. 检查 `docker compose` 服务；仅在服务未全部运行时执行 `docker compose up -d`。
3. 检查 8000 端口上的进程是否确实为 EvidenceRAG；若已运行则复用，不会启动第二个后端。
4. 使用 Conda 环境 `rag` 后台执行：

   ```bat
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

5. 将后端 PID 和日志放在 `%TEMP%\EvidenceRAG`，不在仓库中生成运行文件。

启动成功后访问：

- 前端：<http://127.0.0.1:8000>
- API 文档：<http://127.0.0.1:8000/docs>

首次加载本地 embedding 模型时，后端可能需要几十秒才开始监听端口，脚本最多等待 180 秒。

## 一键停止

```bat
scripts\stop_rag.bat
```

停止脚本会验证 PID 对应的命令行确实属于 EvidenceRAG，再停止后端进程；随后只执行：

```bat
docker compose stop
```

它不会执行 `docker compose down`，也不会删除 container、image 或 volume。PostgreSQL、Redis、Milvus、MinIO 和 etcd 数据目录均被保留。

## 常见错误

- `Docker CLI was not found`：安装或启动 Docker Desktop，确认 `docker` 已加入 PATH。
- Docker daemon 超时：手动打开 Docker Desktop，等待状态变为 Running，再重新执行启动脚本。
- `Conda was not found`：从 Anaconda Prompt 运行，或先执行 `conda init powershell` 并重开终端。
- 8000 端口被其他程序占用：停止占用进程后重试；脚本不会误杀非 EvidenceRAG 程序。
- 后端启动超时：查看 `%TEMP%\EvidenceRAG\backend.log` 和 `backend-error.log`。
