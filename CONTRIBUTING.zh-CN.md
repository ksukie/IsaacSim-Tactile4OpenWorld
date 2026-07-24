<p align="right">
  <a href="CONTRIBUTING.md">English</a> · <strong>简体中文</strong>
</p>

# 贡献指南

欢迎有助于保持仓库可复现、来源可追踪且可依法再分发的贡献。

## 提交修改前

1. 大型行为或目录变更应在实现前通过 issue 讨论。
2. 使用正确基线：当前路线使用 `active-isaaclab-2.1.1/`；只有维护历史内容时才修改 `archive-isaaclab-2.3.2/`。
3. 不要提交凭据、个人数据、生成的包元数据、构建产物、模型检查点、专有 SDK 或原生二进制。
4. 复制或改编内容时，保留上游通知，并补充准确来源 URL/修订、SPDX 许可证标识、修改说明和所需引用；适用时同步更新 `THIRD_PARTY_NOTICES.md` 与 `CITATIONS.md`。
5. 提交贡献即表示你确认有权按相关文件声明的许可证提供该内容。每个提交使用 `git commit -s` 添加 Developer Certificate of Origin 签署。

## 验证

提交前运行仓库检查：

```bash
python tools/repository/audit_open_source.py
python tools/repository/build_static_navigation.py
python tools/repository/finalize_layout.py
```

这些检查是静态检查。如果修改影响仿真、CUDA、UIPC 或硬件行为，请记录实际验证的准确环境和运行命令。不得将未执行场景描述为已经测试。

## Pull Request 范围

每个 Pull Request 应聚焦单一目标。说明研究问题或缺陷，列出受影响基线，标明第三方内容，描述验证方法，并明确指出兼容性或许可证变化。

## 文档

所有持续维护、面向用户的文档必须提供内容对应的英文和简体中文版本。两个版本应保持相同标题层级、命令、事实声明和导航。用户指南放在 `docs/en/` 与 `docs/zh-CN/`；生成型清单和发布维护记录放在 `docs/internal/`。

修改接口时，应同步更新中英文安装、使用指南、参考和故障排查页面。只有确实在已声明环境中执行过的命令，才能描述为“已测试”。
