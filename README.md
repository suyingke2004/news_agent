# News Fact-Check Agent | 新闻事实核查引擎

基于 LangGraph 多智能体验证管道的假新闻识别系统。通过多源交叉比对、多角色辩论和证据聚合，自动判定用户提交的新闻主张真假。

## 架构

```
用户输入 → 主张提取 → 主张拆解 → [并行] 证据检索 → 来源可信度评估
                                              ↓
完整报告 ← 结论综合 ← 多智能体辩论 ← 证据聚合
```

7 个 LangGraph StateGraph 节点，Send fan-out 并行证据检索：

| 节点 | 功能 |
|------|------|
| `claim_extractor` | 从 URL/文本/声明中提取可验证主张 |
| `claim_decomposer` | 将复合主张拆解为原子主张 |
| `evidence_retriever` | 多源检索（NewsAPI / Reddit / RSS / Web） |
| `source_credibility` | 三层域名可信度评分 |
| `evidence_aggregator` | 去重、立场分类（支持/反对/中性） |
| `multi_agent_debate` | Advocate / Skeptic / Neutral 三方辩论 + Judge 裁决 |
| `verdict_synthesizer` | 逐项裁决 + 总体判定 + Markdown 报告 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，至少配置一个 LLM Provider：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 3. 启动

```bash
python app.py
```

访问 `http://localhost:5001`

## 使用方式

1. 在输入框粘贴新闻 URL、新闻段落或具体声明
2. 选择模型 Provider 和模型名称
3. 点击「发送验证」
4. 实时查看 7 阶段验证进度
5. 查看结果：总体判定（TRUE / FALSE / MIXED / UNVERIFIED）、置信度、证据卡片、辩论摘要、完整报告

## 支持的模型

| Provider | 模型 | Base URL |
|----------|------|----------|
| DeepSeek | deepseek-chat, deepseek-reasoner, deepseek-v4-pro | api.deepseek.com |
| OpenAI | gpt-4o, gpt-4o-mini | api.openai.com |
| Zhipu AI | glm-4, glm-4-air | open.bigmodel.cn |
| Alibaba Cloud | qwen-max, qwen-plus | dashscope.aliyuncs.com |
| Moonshot AI | moonshot-v1-8k, moonshot-v1-32k | api.moonshot.cn |

## 判定结果

| 判定 | 含义 | 显示色 |
|------|------|--------|
| **TRUE** | 证据支持该主张为真 | 🟢 绿色 |
| **FALSE** | 证据显示该主张为假或误导 | 🔴 红色 |
| **MIXED** | 证据相互矛盾，结论部分支持 | 🟡 琥珀色 |
| **UNVERIFIED** | 证据不足，无法做出判断 | ⚪ 灰色 |

## 前端特性

- 深色玻璃拟态设计
- 7 阶段实时进度条
- 总体判定徽章 + 置信度圆环
- 证据卡片（支持/反对/中性色标边框）
- 多智能体辩论折叠面板
- Markdown 报告渲染
- 验证历史记录
- 中英文界面切换
- 响应式移动端适配

## 项目结构

```
news_agent/
├── app.py                      # Flask 应用（SSE 流式验证）
├── models.py                   # SQLAlchemy 数据模型
├── graph/
│   ├── state.py                # FactCheckState / Claim / Evidence / Verdict
│   ├── llm_config.py           # 多 Provider LLM 工厂
│   ├── builder.py              # StateGraph 构建 + Send fan-out
│   ├── agent.py                # FactCheckAgent 封装
│   └── nodes/
│       ├── utils.py            # emit_progress / invoke_structured
│       ├── claim_extractor.py  # 主张提取
│       ├── claim_decomposer.py # 主张拆解
│       ├── evidence_retriever.py  # 多源证据检索（工具容错）
│       ├── source_credibility.py  # 来源可信度评分
│       ├── evidence_aggregator.py # 证据聚合 + 立场分类
│       ├── multi_agent_debate.py  # 多智能体辩论
│       └── verdict_synthesizer.py # 结论综合
├── templates/
│   └── verify.html             # 验证前端
├── tools/                      # 数据源工具（NewsAPI / Reddit / RSS / Web）
├── tests/                      # 111 个测试（110 passed）
├── requirements.txt
└── .env.example
```

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行核心 Pipeline 测试
python -m pytest tests/test_claim_extractor.py tests/test_evidence_retriever.py tests/test_multi_agent_debate.py tests/test_verdict_synthesizer.py -v

# 运行边缘用例测试
python -m pytest tests/test_claim_extractor_edge.py tests/test_claim_decomposer_edge.py tests/test_evidence_retriever_resilience.py -v
```

## 技术栈

- **Web**: Flask + SQLAlchemy + SSE
- **AI Pipeline**: LangGraph StateGraph + LangChain
- **LLM**: DeepSeek / OpenAI / Zhipu / Qwen / Moonshot（OpenAI 兼容接口）
- **数据源**: NewsAPI / Reddit (PRAW) / RSS (feedparser) / Web (BeautifulSoup)
- **前端**: Vanilla JS + Jinja2 + marked.js

## 注意事项

- 至少配置一个 LLM Provider 的 API Key
- NewsAPI / Reddit API Key 为可选（缺失时优雅降级，不影响 LLM 辩论和判定）
- `evidence_retriever` 在工具模块不可用时自动跳过，Pipeline 仍可完成
- 默认端口 `5001`，可在 `app.py` 底部修改
