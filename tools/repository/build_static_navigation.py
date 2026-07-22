#!/usr/bin/env python3
"""Build static script-navigation and path-adaptation reports.

This tool reads Python source with ``ast`` and reads the existing path-reference
CSV.  It does not import project modules, launch Isaac Sim, install packages, or
modify either version directory.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
DOCS_DIR = WORKSPACE / "docs"
TOOLS_DIR = WORKSPACE / "tools" / "repository"
INVENTORY_CSV = TOOLS_DIR / "ENTRYPOINT_INVENTORY.csv"
MATRIX_MD = DOCS_DIR / "ENTRYPOINT_MATRIX.md"
PATH_SOURCE_CSV = TOOLS_DIR / "external_path_references.csv"
PATH_PLAN_CSV = TOOLS_DIR / "PATH_ADAPTATION_PLAN.csv"
ADAPTATIONS_CSV = TOOLS_DIR / "PORTABILITY_ADAPTATIONS.csv"

ASSET_PATTERN = re.compile(
    r"(?:\$\{[A-Za-z_][A-Za-z0-9_]*\})?"
    r"[A-Za-z0-9_~${}:./\\+\-]*"
    r"\.(?:usd|usda|usdc|urdf|obj|tet|msh|stl|dae|npy|npz|yaml|yml|json|csv|hdf5|h5|png|jpg|jpeg|so|dll)",
    re.IGNORECASE,
)
VERSION_PATTERN = re.compile(
    r"(?:^|_)(v\d+(?:_new)?(?:_\d+[a-z]?)*)(?:_|$)", re.IGNORECASE
)


@dataclass(frozen=True)
class Baseline:
    key: str
    label: str
    root: Path
    script_roots: tuple[Path, ...]
    expected_count: int
    local_package_roots: frozenset[str]


CURRENT_ROOT = WORKSPACE / "active-isaaclab-2.1"
LEGACY_ROOT = WORKSPACE / "archive-isaaclab-2.3"
BASELINES = (
    Baseline(
        key="2.1.1",
        label="Isaac Lab 2.1.1 / OpenWorldTactile-UIPC",
        root=CURRENT_ROOT,
        script_roots=(CURRENT_ROOT / "experiments", CURRENT_ROOT / "tools"),
        expected_count=132,
        local_package_roots=frozenset(
            {
                "api",
                "envs",
                "openworldtactile",
                "openworldtactile_assets",
                "openworldtactile_tasks",
                "openworldtactile_uipc",
                "uipc",
            }
        ),
    ),
    Baseline(
        key="2.3.2",
        label="Isaac Lab 2.3.2 / legacy OpenWorldTactile-GelSight",
        root=LEGACY_ROOT,
        script_roots=(LEGACY_ROOT / "experiments",),
        expected_count=21,
        local_package_roots=frozenset(
            {"isaaclab_assets", "isaaclab_contrib", "api"}
        ),
    ),
)


def workspace_path(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def read_python(path: Path) -> str:
    with tokenize.open(path) as handle:
        return handle.read()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]

    def is_name(value: ast.expr) -> bool:
        return isinstance(value, ast.Name) and value.id == "__name__"

    def is_main(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) and value.value == "__main__"

    return (is_name(left) and is_main(right)) or (is_main(left) and is_name(right))


def module_summary(tree: ast.Module | None, path: Path) -> str:
    if tree is not None:
        doc = ast.get_docstring(tree, clean=True)
        if not doc:
            for statement in tree.body[:4]:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    doc = statement.value.value.strip()
                    break
        if doc:
            first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
            if first:
                return first[:240]
    return path.stem.replace("_", " ")[:240]


def import_names(tree: ast.Module | None) -> list[str]:
    if tree is None:
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                names.add(prefix + node.module)
            else:
                names.update(prefix + alias.name for alias in node.names)
    return sorted(name for name in names if name.lstrip(".") != "__future__")


def asset_references(tree: ast.Module | None) -> list[str]:
    if tree is None:
        return []
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in ASSET_PATTERN.finditer(node.value):
                value = match.group(0).strip("'\"`()[]{}<>,;")
                if value and not value.startswith("."):
                    references.add(value.replace("\\", "/"))
                elif value:
                    references.add(value.replace("\\", "/"))
    return sorted(references, key=str.lower)


def version_label(filename: str) -> str:
    match = VERSION_PATTERN.search(Path(filename).stem)
    return match.group(1).lower() if match else ""


def classify_role(relative: Path, has_guard: bool) -> str:
    name = relative.name.lower()
    parts = {part.lower() for part in relative.parts}
    stem = relative.stem.lower()
    if name == "__init__.py":
        return "包初始化模块"
    if "envs" in parts:
        return "场景/环境配置模块"
    if "api" in parts:
        return "API 辅助模块"
    if stem.startswith("test_") or stem.endswith("_test"):
        return "静态测试入口" if has_guard else "测试模块"
    if stem in {"membrane_local_frame", "tu_tactile_field"}:
        return "算法辅助模块（含独立检查）" if has_guard else "算法辅助模块"
    if "benchmarking" in parts:
        return "性能基准入口"
    if stem.startswith(("build_", "evaluate_", "render_", "run_", "validate_", "view_")):
        return "构建/评估/查看工具入口"
    if stem.startswith(("check_", "export_", "inspect_")) or "tools" in parts:
        return "检查/数据工具入口"
    if stem.endswith("_scene_base") or stem.endswith("_scene"):
        return "场景定义/演示入口" if has_guard else "场景定义模块"
    if stem.startswith("openworldtactile_v"):
        return "OpenWorldTactile 版本实验入口"
    if "tactile-bench" in parts and stem.startswith("openworldtactile_"):
        return "OpenWorldTactile 辅助实验入口"
    return "实验演示入口" if has_guard else "辅助模块"


def status_for(baseline: Baseline, relative: Path, role: str, syntax: str) -> str:
    if syntax != "ok":
        return "静态语法待检查"
    if relative.name == "OpenWorldTactile_v6_2_grasp.py":
        return "当前说明中的默认参考入口；不替代历史版本"
    if role == "OpenWorldTactile 版本实验入口":
        return "阶段/历史版本，重命名后仍保留演进关系"
    if baseline.key == "2.3.2" and relative.as_posix().startswith("experiments/franka/current/"):
        return "legacy current 实验线，按用途选择"
    if role in {"包初始化模块", "场景/环境配置模块", "API 辅助模块", "算法辅助模块"}:
        return "被入口脚本调用，不作为默认运行入口"
    if role == "测试模块":
        return "由测试工具发现/调用，不作为直接入口"
    return "静态识别的可导航脚本"


def analyze_baseline(baseline: Baseline) -> list[dict[str, object]]:
    files = sorted(
        (path for root in baseline.script_roots for path in root.rglob("*.py")),
        key=lambda p: p.as_posix().lower(),
    )
    if len(files) != baseline.expected_count:
        raise RuntimeError(
            f"{baseline.key}: expected {baseline.expected_count} scripts, got {len(files)}"
        )

    module_names = {
        path.relative_to(baseline.root)
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        for path in files
    }
    script_stems = {path.stem for path in files if path.stem != "__init__"}
    script_roots = {name.split(".", 1)[0] for name in module_names if "." in name}
    local_roots = set(baseline.local_package_roots) | script_roots

    rows: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(baseline.root)
        source = read_python(path)
        syntax_status = "ok"
        syntax_detail = ""
        try:
            tree: ast.Module | None = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            tree = None
            syntax_status = "error"
            syntax_detail = f"line {exc.lineno}: {exc.msg}"

        imports = import_names(tree)
        local_imports: list[str] = []
        runtime_import_roots: set[str] = set()
        for name in imports:
            plain = name.lstrip(".")
            if not plain:
                continue
            root = plain.split(".", 1)[0]
            tail = plain.rsplit(".", 1)[-1]
            is_local = (
                name.startswith(".")
                or plain in module_names
                or root in local_roots
                or tail in script_stems
            )
            if is_local:
                local_imports.append(name)
            elif root not in sys.stdlib_module_names:
                runtime_import_roots.add(root)

        has_guard = bool(
            tree
            and any(isinstance(node, ast.If) and is_main_guard(node) for node in tree.body)
        )
        has_main_function = bool(
            tree
            and any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
                for node in tree.body
            )
        )
        assets = asset_references(tree)
        role = classify_role(relative, has_guard)
        direct_candidate = has_guard and role not in {"包初始化模块", "测试模块"}
        static_signals = []
        if has_guard:
            static_signals.append("main_guard")
        if has_main_function:
            static_signals.append("main_function")
        if "argparse" in imports or "ArgumentParser" in source:
            static_signals.append("argparse")
        if "AppLauncher" in source:
            static_signals.append("AppLauncher")
        if relative.stem.startswith("test_"):
            static_signals.append("test_module")

        rows.append(
            {
                "baseline": baseline.key,
                "baseline_label": baseline.label,
                "group": relative.parent.as_posix(),
                "source_path": workspace_path(path),
                "scripts_relative_path": relative.as_posix(),
                "filename": path.name,
                "version_label": version_label(path.name),
                "role": role,
                "navigation_status": status_for(
                    baseline, relative, role, syntax_status
                ),
                "direct_run_candidate": "yes" if direct_candidate else "no",
                "has_main_guard": "yes" if has_guard else "no",
                "has_main_function": "yes" if has_main_function else "no",
                "uses_argparse": "yes" if "argparse" in imports or "ArgumentParser" in source else "no",
                "uses_app_launcher": "yes" if "AppLauncher" in source else "no",
                "syntax_status": syntax_status,
                "syntax_detail": syntax_detail,
                "summary": module_summary(tree, path),
                "direct_local_imports": "; ".join(sorted(set(local_imports))),
                "runtime_import_roots": "; ".join(sorted(runtime_import_roots)),
                "referenced_asset_count": len(assets),
                "referenced_assets": "; ".join(assets),
                "static_signals": "; ".join(static_signals) or "module_only",
            }
        )
    return rows


def compact_list(value: object, limit: int = 3) -> str:
    items = [item.strip() for item in str(value).split(";") if item.strip()]
    if not items:
        return "—"
    shown = items[:limit]
    suffix = f" +{len(items) - limit}" if len(items) > limit else ""
    return ", ".join(shown) + suffix


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_matrix(rows: list[dict[str, object]]) -> str:
    baseline_counts = Counter(str(row["baseline"]) for row in rows)
    direct_counts = Counter(
        str(row["baseline"])
        for row in rows
        if row["direct_run_candidate"] == "yes"
    )
    syntax_errors = [row for row in rows if row["syntax_status"] != "ok"]
    group_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        group_rows[(str(row["baseline"]), str(row["group"]))].append(row)
    with ADAPTATIONS_CSV.open("r", encoding="utf-8", newline="") as handle:
        adaptation_count = len(list(csv.DictReader(handle)))

    lines = [
        "# 脚本入口矩阵",
        "",
        f"本报告只对两个版本目录的 `experiments/`（以及主线 `tools/`）下 Python 文件做静态解析；没有导入项目模块、启动 Isaac Sim、安装依赖、编译或运行实验。仓库品牌为 IsaacSim-Tactile4OpenWorld，项目技术名称为 OpenWorldTactile；{adaptation_count} 个可移植性适配继续保留。",
        "",
        "## 结论",
        "",
        f"- 共登记 {len(rows)} 个 Python 脚本：2.1.1 为 {baseline_counts['2.1.1']} 个，2.3.2 为 {baseline_counts['2.3.2']} 个。",
        f"- 根据 `if __name__ == \"__main__\"` 等静态信号，2.1.1 有 {direct_counts['2.1.1']} 个、2.3.2 有 {direct_counts['2.3.2']} 个可直接导航脚本。该判断不等于环境可运行验证。",
        f"- 静态 AST 语法异常：{len(syntax_errors)} 个。",
        "- 2.1.1 的当前说明默认参考入口是 `OpenWorldTactile_v6_2_grasp.py`；V1–V6.1b 和其他分支继续完整保留。",
        "- 2.3.2 没有被打包说明指定为唯一总入口；`franka_experiments/current`、`sensors` 和 `tactile_rgb_pipeline` 是并列用途入口，不能强行合并成一个主入口。",
        "",
        "## 默认导航与直接依赖链",
        "",
        "```text",
        "OpenWorldTactile_v6_2_grasp.py",
        "  -> OpenWorldTactile_v5_new_9_tu_tactile_field_rendering.py",
        "     -> OpenWorldTactile_v5_new_7g_deformation_force_estimator.py",
        "     -> tu_tactile_field.py",
        "```",
        "",
        "这只是 2.1.1 路线的默认导航，不是删除、覆盖或重命名其他版本。完整实验演进继续以 `OWTBENCH_VERSION_INDEX.md` 和 `VERSION_LINEAGE.md` 为准。",
        "",
        "## 版本与分组汇总",
        "",
        "| 基线 | 分组 | Python 文件 | 静态直接入口 | 辅助/测试模块 |",
        "|---|---|---:|---:|---:|",
    ]
    for (baseline, group), members in sorted(group_rows.items()):
        direct = sum(row["direct_run_candidate"] == "yes" for row in members)
        lines.append(
            f"| {baseline} | `{md_escape(group)}` | {len(members)} | {direct} | {len(members) - direct} |"
        )

    lines.extend(
        [
            "",
            "## 项目根部的补充启动/辅助文件",
            "",
            "这些文件不在上述 153 个静态入口统计内，但属于导航时需要看见的项目级入口或辅助项。",
            "",
            "| 基线 | 文件 | 静态用途 |",
            "|---|---|---|",
            f"| 2.1.1 | [`run.sh`](../{workspace_path(CURRENT_ROOT / 'run.sh')}) | OpenWorldTactile 项目启动/环境包装脚本 |",
            f"| 2.3.2 | [`inspect_gelsight_cfg.py`](../{workspace_path(LEGACY_ROOT / 'tools/inspect_gelsight_cfg.py')}) | GelSight 配置检查辅助脚本 |",
            f"| 2.3.2 | [`set_openworldtactile_camera_view.py`](../{workspace_path(LEGACY_ROOT / 'tools/set_openworldtactile_camera_view.py')}) | OpenWorldTactile 相机视角辅助脚本 |",
            f"| 2.3.2 | [`run-original-gelsight.sh`](../{workspace_path(LEGACY_ROOT / 'tools/run-original-gelsight.sh')}) | 原 GelSight 保存流程包装脚本 |",
            "",
            "## 完整脚本清单",
            "",
            "“可直接导航”仅来自静态主守卫等信号；“运行依赖根”不包含 Python 标准库。资产列是源码字符串中可静态识别的文件引用摘要。完整字段见 `../tools/repository/ENTRYPOINT_INVENTORY.csv`。",
        ]
    )

    for baseline in ("2.1.1", "2.3.2"):
        label = next(item.label for item in BASELINES if item.key == baseline)
        lines.extend(["", f"### {label}", ""])
        groups = sorted(group for key, group in group_rows if key == baseline)
        for group in groups:
            lines.extend(
                [
                    f"#### `{group}`",
                    "",
                    "| 文件名 | 版本 | 角色 | 可直接导航 | 静态信号 | 同项目直接导入 | 运行依赖根 | 资产引用 |",
                    "|---|---|---|---|---|---|---|---|",
                ]
            )
            for row in sorted(
                group_rows[(baseline, group)],
                key=lambda item: str(item["filename"]).lower(),
            ):
                link = f"../{row['source_path']}"
                assets = compact_list(row["referenced_assets"], 2)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"[`{md_escape(row['filename'])}`]({link})",
                            f"`{md_escape(row['version_label'])}`" if row["version_label"] else "—",
                            md_escape(row["role"]),
                            "是" if row["direct_run_candidate"] == "yes" else "否",
                            f"`{md_escape(row['static_signals'])}`",
                            md_escape(compact_list(row["direct_local_imports"], 3)),
                            md_escape(compact_list(row["runtime_import_roots"], 4)),
                            md_escape(assets),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    lines.extend(
        [
            "## 阅读边界",
            "",
            "- `yes` 只表示文件具备静态直接执行结构，不证明依赖齐全或仿真结果正确。",
            "- API、环境配置和无主守卫测试仍完整登记，因为它们是版本关系的一部分。",
            "- 本报告没有给 2.3.2 人为指定一个不存在的统一主入口。",
            "- 当前可移植性机制见 `../tools/repository/PORTABILITY_ADAPTATIONS.csv`。",
            "- 路径适配优先级见 `../tools/repository/PATH_ADAPTATION_PLAN.csv`，缺失依赖见 `DEPENDENCY_GAPS.md`。",
        ]
    )
    return "\n".join(lines)


def python_reference_is_comment(source_file: str, reference: str) -> bool:
    path = WORKSPACE / Path(source_file)
    source = read_python(path)
    matches = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if reference in token.string
    ]
    return bool(matches) and all(token.type == tokenize.COMMENT for token in matches)


def path_scope(source_file: str, reference: str) -> str:
    suffix = Path(source_file).suffix.lower()
    if suffix == ".py" and python_reference_is_comment(source_file, reference):
        return "code_comment"
    if suffix in {".py", ".sh", ".bash"}:
        return "code_or_script"
    if suffix in {".md", ".txt", ".rst"}:
        return "documentation"
    if suffix in {".yaml", ".yml", ".toml", ".json", ".xml"}:
        return "configuration"
    return "other"


def adaptation_for(classification: str, scope: str) -> dict[str, str]:
    if classification == "source_machine_absolute_path":
        if scope == "code_comment":
            return {
                "priority": "low",
                "change_type": "inactive_code_comment",
                "proposed_action": "无需修改；实际代码已使用当前文件位置计算项目相对路径",
                "code_change_required": "no",
                "configuration_change_required": "no",
                "reason": "绝对路径只存在于旧注释，不参与执行",
            }
        if scope == "code_or_script":
            return {
                "priority": "high",
                "change_type": "code_or_script_portability",
                "proposed_action": "后续改为项目相对路径、命令行参数或环境变量",
                "code_change_required": "yes",
                "configuration_change_required": "possible",
                "reason": "来源机绝对路径会直接阻断跨机器运行",
            }
        if scope == "configuration":
            return {
                "priority": "medium",
                "change_type": "configuration_portability",
                "proposed_action": "迁移时审核并改为目标机路径或环境变量",
                "code_change_required": "no",
                "configuration_change_required": "yes",
                "reason": "配置记录绑定来源机，但不需要改 Python 源码",
            }
        return {
            "priority": "low",
            "change_type": "source_record",
            "proposed_action": "保留为来源机记录；复制命令时手动替换路径",
            "code_change_required": "no",
            "configuration_change_required": "no",
            "reason": "文档/记录中的路径不直接改变当前代码执行",
        }
    if classification == "external_environment_asset":
        return {
            "priority": "medium",
            "change_type": "external_asset_mapping",
            "proposed_action": "保持环境变量引用；运行前提供对应 Nucleus 资产或映射",
            "code_change_required": "no",
            "configuration_change_required": "yes",
            "reason": "引用设计为由外部 Isaac Lab/Nucleus 环境提供",
        }
    if classification == "runtime_output_or_temporary_path":
        return {
            "priority": "low",
            "change_type": "runtime_output",
            "proposed_action": "保留运行时生成逻辑；目标环境按需设置输出/临时目录",
            "code_change_required": "no",
            "configuration_change_required": "possible",
            "reason": "属于输出或临时路径，不是重构后的静态输入缺失",
        }
    raise RuntimeError(f"Unknown path classification: {classification}")


def current_path_status(classification: str, scope: str) -> str:
    if scope == "code_comment":
        return "已确认无需修改；仅为非执行注释"
    if classification == "source_machine_absolute_path":
        return "当前保留为文档、文本或配置迁移记录"
    if classification == "external_environment_asset":
        return "当前保留；运行环境仍需提供资产或映射"
    return "当前保留为运行时输出或临时路径"


def build_path_plan() -> list[dict[str, object]]:
    with PATH_SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for index, source in enumerate(source_rows, start=1):
        scope = path_scope(source["source_file"], source["reference"])
        action = adaptation_for(source["classification"], scope)
        rows.append(
            {
                "item_id": f"PATH-{index:04d}",
                "baseline": source["baseline"],
                "source_file": source["source_file"],
                "reference": source["reference"],
                "classification": source["classification"],
                "reference_scope": scope,
                **action,
                "current_status": current_path_status(
                    source["classification"], scope
                ),
            }
        )
    return rows


def main() -> None:
    inventory_rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        inventory_rows.extend(analyze_baseline(baseline))
    if len(inventory_rows) != 153:
        raise RuntimeError(f"Expected 153 Python script rows, got {len(inventory_rows)}")

    inventory_fields = [
        "baseline",
        "baseline_label",
        "group",
        "source_path",
        "scripts_relative_path",
        "filename",
        "version_label",
        "role",
        "navigation_status",
        "direct_run_candidate",
        "has_main_guard",
        "has_main_function",
        "uses_argparse",
        "uses_app_launcher",
        "syntax_status",
        "syntax_detail",
        "summary",
        "direct_local_imports",
        "runtime_import_roots",
        "referenced_asset_count",
        "referenced_assets",
        "static_signals",
    ]
    write_csv(INVENTORY_CSV, inventory_rows, inventory_fields)
    write_text(MATRIX_MD, build_matrix(inventory_rows))

    path_rows = build_path_plan()
    path_fields = [
        "item_id",
        "baseline",
        "source_file",
        "reference",
        "classification",
        "reference_scope",
        "priority",
        "change_type",
        "proposed_action",
        "code_change_required",
        "configuration_change_required",
        "reason",
        "current_status",
    ]
    write_csv(PATH_PLAN_CSV, path_rows, path_fields)

    syntax_errors = sum(row["syntax_status"] != "ok" for row in inventory_rows)
    direct = sum(row["direct_run_candidate"] == "yes" for row in inventory_rows)
    path_counts = Counter(str(row["classification"]) for row in path_rows)
    print(f"entrypoint_inventory={len(inventory_rows)}")
    print(f"direct_run_candidates={direct}")
    print(f"syntax_errors={syntax_errors}")
    print(f"current_path_plan={len(path_rows)}")
    for key in sorted(path_counts):
        print(f"path_{key}={path_counts[key]}")


if __name__ == "__main__":
    main()
