# Prompt Changelog

本文件记录 **Prompt 资产包**（`app/core/agent/prompt/` 下 md/yaml/manifest）的语义版本。  
与 **应用版本**（`pyproject.toml` 的 `version`）分离：仅改文案/Few-shot 时只升此处。

契约版本见 `manifest.yaml` 的 `prompt_pack_format`（当前为 `1`）。  
应用侧支持的格式由 `composer.SUPPORTED_PROMPT_PACK_FORMAT` 定义。

## 版本与 Git Tag 建议

| prompt_version | Git tag（可选） | 说明 |
|----------------|-----------------|------|
| 1.1.0 | `prompt-v1.1.0` | Few-shot 4 组；格式下沉到样例 |
| 1.0.0 | `prompt-v1.0.0` | 模块化 Core + Module + manifest |

打 tag 示例：

```bash
git tag -a prompt-v1.1.0 -m "prompt 1.1.0: few-shot refactor"
```

回滚 Prompt（不动应用代码）示例：

```bash
git checkout prompt-v1.0.0 -- app/core/agent/prompt/
```

---

## [1.1.0] - 2026-05-22

### Changed

- `core/base.md`：输出格式改为「遵循 Few-shot 样例」，细则下沉到 `few_shot/default.yaml`
- `few_shot/default.yaml`：5 组收敛为 4 组（概念 / 对比表 / 边界 / 短答），去掉重复对比样例

### Added

- `manifest.yaml`：`prompt_version`、`prompt_pack_format`
- `CHANGELOG.md`、评测集 `doc/prompt-eval-set.md`

---

## [1.0.0] - 2026-05-20

### Added

- 模块化目录：`core/`、`modules/`、`manifest.yaml`
- `composer` 组装 Policy；`Scene` 场景枚举
- `few_shot/default.yaml` 初版（练习用）
