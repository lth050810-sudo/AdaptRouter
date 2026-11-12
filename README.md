# AdaptRouter

Continual Learning to Route across Evolving Tasks and Models.

- [Project website](https://lth050810-sudo.github.io/AdaptRouter/)
- [Table 3 API-router training code and instructions](api_router/README.md)
- [StartLux 0.8B router weights on Hugging Face](https://huggingface.co/lyyyk/AdaptRouter-StartLux-0.8B)

本次研究代码发布范围为 **Table 3 的 StartLux API 专家池主训练实现**，包括全参数训练、验证选模、固定配方 refit 和原生路由推理。其他实验的训练代码不在本次发布范围内。

归档保留了全局主训练代码与权重；原始专家反馈数据及部分基准专项微调代码未包含。完整复现范围、安装步骤与运行命令见 [api_router/README.md](api_router/README.md)。本仓库代码采用 [Apache-2.0](api_router/LICENSE)，模型与数据分别遵循其各自许可。
