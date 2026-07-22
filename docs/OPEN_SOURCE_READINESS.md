# 开源发布审计

## 结论

截至 2026-07-22，当前工作树已达到**公开源码仓库的工程发布标准**：原创贡献具有根级 BSD-3-Clause 许可证，主要第三方组件的来源、许可证、强 copyleft 边界和论文引用均已登记，未确认授权的 SDK、原生二进制、测试资产和模型权重已从公开载荷移除。

这不是法律意见。最终发布者仍须确认自己或所在机构拥有原创提交的授权权利；如果代码由雇佣、合作、资助或保密协议约束，应在公开前完成内部审批。

## 已完成项

- 根级 [`LICENSE`](../LICENSE)、[`NOTICE`](../NOTICE) 和 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) 已建立。
- `CITATION.cff` 与 [`CITATIONS.md`](../CITATIONS.md) 区分项目引用和方法论文引用。
- `CONTRIBUTING.md` 与 `SECURITY.md` 定义了贡献权利确认、第三方引入和漏洞报告流程。
- 四个 Python 包的错误 MIT 元数据已经纠正；Isaac Lab 模板仓库 URL 和模板标题已移除。
- Isaac Lab/ORBIT、libuipc、MuDA、TetGen、Catch2、Octree、Piper ROS、MoveIt、Taxim、FOTS、GelSight、Franka ROS 与 ManiSkill-ViTac2025 已登记到第三方清单。
- `SonixCamera.dll`、`libSonixCamera.so` 及缺少有效许可证正文的历史相机 SDK 已移出公开仓库。
- 无明确上游许可证的 tactile test shapes、无 provenance 的 `IK_old.pt` 和生成的 `*.egg-info` 已移出公开仓库。
- [`audit_open_source.py`](../tools/repository/audit_open_source.py) 会复查政策文件、关键许可证、原生/不透明载荷、生成目录、包元数据、模板残留和常见凭据形态。

## 多许可证边界

根 BSD-3-Clause 只覆盖原创 OpenWorldTactile 贡献，不会把第三方子树重新许可为 BSD。尤其需要注意：

- libuipc Python 绑定为 GPL-3.0-only；
- TetGen 为 AGPL-3.0-or-later，另提供商业授权路径；
- GelSight 衍生资产按 GPL-3.0-only 分发；
- libuipc、MuDA、Franka ROS 和 ManiSkill-ViTac2025 衍生文件适用 Apache-2.0；
- Piper ROS、Taxim、FOTS 与 Octree 适用各自 MIT 条款；
- Isaac Lab/ORBIT 与 MoveIt 内容保留各自 BSD 通知。

因此，“仓库可开源”不等于“所有内容都可被闭源产品无条件吸收”。二进制发布者必须按最终组合重新评估 GPL/AGPL、动态/静态链接、源码提供和 NOTICE 义务。

## 首次发布清单

1. 创建远端仓库后，将最终 URL 补入四个 `extension.toml` 和 `CITATION.cff`；当前为空，避免伪造或误指向 Isaac Lab 模板仓库。
2. 在发布页明确标注“研究代码、非安全认证产品”，并链接第三方清单。
3. 运行：

   ```bash
   py tools/repository/audit_open_source.py
   py tools/repository/build_static_navigation.py
   py tools/repository/finalize_layout.py
   ```

4. 对计划宣称支持的 Isaac Lab/Isaac Sim/CUDA 组合做一次干净环境验证；若尚未执行，只声明静态检查结果。
5. 若发布带 UIPC/TetGen 的预编译二进制，单独完成 GPL/AGPL 合规审查。源码仓库通过不自动覆盖二进制分发义务。

## 已知保留风险

- FOTS 上游 README 明确声明 MIT，但其当前 `LICENSE` 链接在审计时失效；仓库已保留来源、MIT 意图、本地条款和修改说明。商业发行前建议再次核对上游历史或取得作者确认。
- 若现有原创代码受雇佣关系、学校/机构政策或合作协议约束，根许可证必须由实际权利人批准。
- 资产来源记录无法替代商标、专利、隐私、数据许可或出口管制审查；本仓库只完成了当前可见载荷的开源发布整理。
