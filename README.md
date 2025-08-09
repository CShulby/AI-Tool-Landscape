# AI-Tool-Landscape
recent AI tools covering various dev areas.

## Orchestration, Prompt & Agent Dev

* CrewAI (multi-agent framework): ([CrewAI][1])
* LangChain (JS/Python framework): ([LangChain][2], [LangChain Docs][3])
* Haystack (RAG/agents): ([Haystack Documentation][4])
* Helicone (LLM gateway + observability): ([Helicone.ai][5])
* Humanloop (prompt management/evals): ([Humanloop][6])
* PromptLayer (prompt CMS/versioning): ([PromptLayer][7], [PromptLayer][8])
* n8n (workflow/AI automation): ([n8n][9])

## Context & Memory (incl. Vector DBs)

* mem0 (memory layer): ([Zep][10])
* Letta / MemGPT (stateful agents with memory): ([Pinecone][11])
* Zep (memory & knowledge graphs): ([Weaviate][12])
* Pinecone: ([Chroma Docs][13])
* Weaviate: ([Qdrant][14])
* Chroma: ([Letta][15])
* Qdrant: ([LangChain][16])

## Tool-Call & Integration Layer

* Composio (tools/integrations for agents): ([Composio][17])
* Toolhouse (tool calling platform): ([toolhouse.ai][18])
* E2B (secure sandboxes for agents/code): ([e2b.dev][19])
* Zapier AI Actions / MCP (connect 6k–8k+ apps via AI actions): ([actions.zapier.com][20], [docs.zapier.com][21], [Zapier][22])
* Make.com AI Agents: ([Composio][23])

## Prompt Gateways & Multi-Model Routing

* OpenRouter (unified API & fallbacks): ([OpenRouter][24])
* Portkey (open-source AI gateway + obs/guardrails): ([Portkey][25], [GitHub][26])
* Not Diamond (multi-model routing): ([Not Diamond][27], [Not Diamond][28])
* TensorZero (gateway + evals/obs/optimization): ([TensorZero][29])
* Martian (router/benchmarking): ([Martian][30])

## Model Interpretability

* TransformerLens (mech-interp library): ([GitHub][31], [Transformer Lens][32])
* LIT – Language Interpretability Tool (Google PAIR): ([Google Research][33], [Pair Code][34])
* Neuronpedia (open interpretability platform): ([Neuronpedia][35])
* OpenAI Microscope (public neuron viz): ([OpenAI][36])
* CircuitsVis (React/Python viz): ([GitHub][37])

## Reinforcement Fine-Tuning (RLHF/RFT/DPO/GRPO)

* TRL (HF): ([Hugging Face][38], [GitHub][39])
* DeepSpeed-Chat (RLHF system): ([GitHub][40], [arXiv][41])
* OpenRLHF / OpenRLHF-M: ([GitHub][42])
* Predibase (RFT platform): ([Predibase][43])

## Models & Inference Providers

* OpenAI (models/API): ([OpenAI Platform][44], [OpenAI][45])
* Anthropic Claude: ([Anthropic][46], [Anthropic][47])
* Google Gemini API: ([Google AI for Developers][48])
* xAI (Grok): ([xAI Docs][49], [xAI][50])
* Meta Llama: ([Llama][51], [AI Meta][52])
* Google Gemma: ([Google DeepMind][53], [Google AI for Developers][54])
* DeepSeek (API): ([DeepSeek API Docs][55])
* Baseten: ([Baseten][56])
* (Also referenced in README: Fireworks, Together, Anyscale—happy to add more sources if you want links inline.)

## Evaluation, Tracing & Experimentation

* LangSmith: ([LangSmith][57], [LangChain][58])
* Braintrust: ([Braintrust][59])
* promptfoo: ([GitHub][60], [Promptfoo][61])
* TruLens: ([TruLens][62], [GitHub][63])
* DeepEval: ([GitHub][64])
* Ragas: ([GitHub][65], [Ragas][66])
* Arize Phoenix: ([Phoenix][67], [Arize AI][68])
* Giskard: ([giskard.ai][69], [GitHub][70])
* (Context on the push for stronger evals): ([TIME][71])

## Browser Automation, Environments & Benchmarks

* Browserbase: ([browserbase.com][72], [docs.browserbase.com][73])
* browserless: ([Browserless][74], [GitHub][75])
* Playwright: ([Playwright][76], [GitHub][77])
* Selenium (WebDriver): ([Selenium][78])
* Puppeteer: ([pptr.dev][79], [Chrome for Developers][80])
* browser-use (agents + browser): ([GitHub][81])
* WebArena (env/benchmark): ([WebArena][82])
* MiniWoB++ (classic benchmark): ([miniwob.farama.org][83], [GitHub][84])

## Input/Output Guardrails & Safety

* NVIDIA NeMo Guardrails: ([NVIDIA Docs][85])
* Guardrails (rail-spec): ([guardrails][86])
* OpenAI Moderation: ([OpenAI Platform][87])
* Azure AI Content Safety: ([Microsoft Learn][88])
* Lakera Guard (prompt-injection & safety): ([docs.lakera.ai][89], [Lakera][90])
* Rebuff (prompt-injection detector): ([GitHub][91])

[1]: https://docs.crewai.com/?utm_source=chatgpt.com "CrewAI Docs"
[2]: https://python.langchain.com/docs/introduction/?utm_source=chatgpt.com "Introduction | 🦜️ LangChain"
[3]: https://docs.langchain.com/?utm_source=chatgpt.com "LangChain docs home - Docs by LangChain"
[4]: https://docs.haystack.deepset.ai/docs/intro?utm_source=chatgpt.com "Haystack Documentation - Deepset"
[5]: https://www.helicone.ai/?utm_source=chatgpt.com "Helicone / AI Gateway & LLM Observability"
[6]: https://humanloop.com/platform/prompt-management?utm_source=chatgpt.com "Prompt Management Tool for Building LLM Apps"
[7]: https://docs.promptlayer.com/?utm_source=chatgpt.com "Welcome to PromptLayer - PromptLayer"
[8]: https://www.promptlayer.com/platform/prompt-management?utm_source=chatgpt.com "Collaborative Prompting Manage your prompts"
[9]: https://n8n.io/?utm_source=chatgpt.com "AI Workflow Automation Platform & Tools - n8n"
[10]: https://www.getzep.com/?utm_source=chatgpt.com "Zep: Context Engineering Platform for AI Agents"
[11]: https://www.pinecone.io/learn/vector-database/?utm_source=chatgpt.com "What is a Vector Database & How Does it Work? Use ..."
[12]: https://weaviate.io/platform?utm_source=chatgpt.com "The AI-Native, Open Source Vector Database"
[13]: https://docs.trychroma.com/getting-started?utm_source=chatgpt.com "Getting Started - Chroma Docs"
[14]: https://qdrant.tech/qdrant-vector-database/?utm_source=chatgpt.com "Qdrant Vector Database, High-Performance ..."
[15]: https://www.letta.com/?utm_source=chatgpt.com "Letta"
[16]: https://python.langchain.com/api_reference/community/memory/langchain_community.memory.zep_memory.ZepMemory.html?utm_source=chatgpt.com "ZepMemory — 🦜🔗 LangChain documentation"
[17]: https://composio.dev/?utm_source=chatgpt.com "Composio - The Skill Layer of AI"
[18]: https://toolhouse.ai/?utm_source=chatgpt.com "Toolhouse - Deploy smarter AI in one click"
[19]: https://e2b.dev/?utm_source=chatgpt.com "E2B | The Enterprise AI Agent Cloud"
[20]: https://actions.zapier.com/?utm_source=chatgpt.com "Zapier AI Actions: Get Started"
[21]: https://docs.zapier.com/platform/reference/ai-actions?utm_source=chatgpt.com "AI Actions"
[22]: https://zapier.com/mcp?utm_source=chatgpt.com "Zapier MCP—Connect your AI to any app instantly"
[23]: https://composio.dev/blog/ai-agent-tools?utm_source=chatgpt.com "AI Agent Tools: Making the Most of LLMs"
[24]: https://openrouter.ai/?utm_source=chatgpt.com "OpenRouter"
[25]: https://portkey.ai/features/ai-gateway?utm_source=chatgpt.com "Enterprise-grade AI Gateway"
[26]: https://github.com/Portkey-AI/gateway?utm_source=chatgpt.com "Portkey-AI/gateway"
[27]: https://www.notdiamond.ai/?utm_source=chatgpt.com "Not Diamond"
[28]: https://docs.notdiamond.ai/docs/what-is-not-diamond?utm_source=chatgpt.com "What is Not Diamond?"
[29]: https://www.tensorzero.com/?utm_source=chatgpt.com "TensorZero · open-source LLM infrastructure"
[30]: https://www.withmartian.com/?utm_source=chatgpt.com "Martian: Model Routing and AI Interpretability Tools"
[31]: https://github.com/TransformerLensOrg/TransformerLens?utm_source=chatgpt.com "TransformerLensOrg/TransformerLens: A library for ..."
[32]: https://transformerlensorg.github.io/TransformerLens/?utm_source=chatgpt.com "TransformerLens Documentation"
[33]: https://research.google/blog/the-language-interpretability-tool-lit-interactive-exploration-and-analysis-of-nlp-models/?utm_source=chatgpt.com "The Language Interpretability Tool (LIT): Interactive ..."
[34]: https://pair-code.github.io/lit/?utm_source=chatgpt.com "Learning Interpretability Tool - People + AI Research"
[35]: https://www.neuronpedia.org/?utm_source=chatgpt.com "Neuronpedia"
[36]: https://openai.com/index/microscope/?utm_source=chatgpt.com "OpenAI Microscope"
[37]: https://github.com/TransformerLensOrg/CircuitsVis?utm_source=chatgpt.com "TransformerLensOrg/CircuitsVis"
[38]: https://huggingface.co/docs/trl/en/index?utm_source=chatgpt.com "TRL - Transformer Reinforcement Learning"
[39]: https://github.com/huggingface/trl?utm_source=chatgpt.com "huggingface/trl: Train transformer language models with ..."
[40]: https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md?utm_source=chatgpt.com "DeepSpeedExamples/applications/DeepSpeed-Chat ..."
[41]: https://arxiv.org/pdf/2308.01320?utm_source=chatgpt.com "DeepSpeed-Chat"
[42]: https://github.com/OpenRLHF/OpenRLHF?utm_source=chatgpt.com "An Easy-to-use, Scalable and High-performance RLHF ..."
[43]: https://predibase.com/blog/introducing-reinforcement-fine-tuning-on-predibase?utm_source=chatgpt.com "The First Reinforcement Fine-Tuning Platform for LLMs"
[44]: https://platform.openai.com/docs/models?utm_source=chatgpt.com "Models - OpenAI API"
[45]: https://openai.com/api/?utm_source=chatgpt.com "API Platform"
[46]: https://www.anthropic.com/claude?utm_source=chatgpt.com "Meet Claude \ Anthropic"
[47]: https://docs.anthropic.com/en/docs/about-claude/models/overview?utm_source=chatgpt.com "Models overview"
[48]: https://ai.google.dev/gemini-api/docs/models?utm_source=chatgpt.com "Gemini models | Gemini API | Google AI for Developers"
[49]: https://docs.x.ai/docs/overview?utm_source=chatgpt.com "xAI Docs: Overview"
[50]: https://x.ai/api?utm_source=chatgpt.com "API"
[51]: https://www.llama.com/?utm_source=chatgpt.com "Llama: Industry Leading, Open-Source AI"
[52]: https://ai.meta.com/blog/llama-4-multimodal-intelligence/?utm_source=chatgpt.com "The Llama 4 herd: The beginning of a new era of natively ..."
[53]: https://deepmind.google/models/gemma/?utm_source=chatgpt.com "Gemma"
[54]: https://ai.google.dev/gemma/docs?utm_source=chatgpt.com "Gemma models overview | Google AI for Developers"
[55]: https://api-docs.deepseek.com/?utm_source=chatgpt.com "DeepSeek API Docs: Your First API Call"
[56]: https://www.baseten.co/?utm_source=chatgpt.com "Baseten: Deploy AI models in production"
[57]: https://docs.smith.langchain.com/evaluation?utm_source=chatgpt.com "Evaluation Quick Start | 🦜️🛠️ LangSmith - LangChain"
[58]: https://www.langchain.com/langsmith?utm_source=chatgpt.com "LangSmith"
[59]: https://www.braintrust.dev/?utm_source=chatgpt.com "Braintrust - The evals and observability platform for building ..."
[60]: https://github.com/promptfoo/promptfoo?utm_source=chatgpt.com "promptfoo/promptfoo: Test your prompts, agents, and RAGs ..."
[61]: https://www.promptfoo.dev/docs/integrations/github-action/?utm_source=chatgpt.com "Testing Prompts with GitHub Actions"
[62]: https://www.trulens.org/?utm_source=chatgpt.com "TruLens: Evals and Tracing for Agents"
[63]: https://github.com/truera/trulens?utm_source=chatgpt.com "truera/trulens: Evaluation and Tracking for LLM ..."
[64]: https://github.com/confident-ai/deepeval?utm_source=chatgpt.com "confident-ai/deepeval: The LLM Evaluation Framework"
[65]: https://github.com/explodinggradients/ragas?utm_source=chatgpt.com "explodinggradients/ragas: Supercharge Your LLM ..."
[66]: https://docs.ragas.io/en/stable/?utm_source=chatgpt.com "Ragas"
[67]: https://phoenix.arize.com/?utm_source=chatgpt.com "Home - Phoenix - Arize AI"
[68]: https://arize.com/docs/phoenix?utm_source=chatgpt.com "Arize Phoenix"
[69]: https://www.giskard.ai/products/open-source?utm_source=chatgpt.com "Open-Source AI testing library"
[70]: https://github.com/Giskard-AI/giskard?utm_source=chatgpt.com "GitHub - Giskard-AI/giskard: 🐢 Open-Source Evaluation & ..."
[71]: https://time.com/7203729/ai-evaluations-safety/?utm_source=chatgpt.com "AI Models Are Getting Smarter. New Tests Are Racing to Catch Up"
[72]: https://www.browserbase.com/?utm_source=chatgpt.com "Browserbase: A web browser for AI agents & applications"
[73]: https://docs.browserbase.com/reference/introduction?utm_source=chatgpt.com "APIs and SDKs"
[74]: https://www.browserless.io/?utm_source=chatgpt.com "Browserless - Browser Automation and Dodge Bot Detectors"
[75]: https://github.com/browserless/browserless?utm_source=chatgpt.com "Deploy headless browsers in Docker. Run on our cloud or ..."
[76]: https://playwright.dev/?utm_source=chatgpt.com "Playwright: Fast and reliable end-to-end testing for modern web apps"
[77]: https://github.com/microsoft/playwright?utm_source=chatgpt.com "microsoft/playwright"
[78]: https://www.selenium.dev/documentation/webdriver/?utm_source=chatgpt.com "WebDriver"
[79]: https://pptr.dev/?utm_source=chatgpt.com "Puppeteer | Puppeteer"
[80]: https://developer.chrome.com/docs/puppeteer?utm_source=chatgpt.com "Puppeteer - Chrome for Developers"
[81]: https://github.com/browser-use/browser-use?utm_source=chatgpt.com "browser-use/browser-use: 🌐 Make websites accessible for ..."
[82]: https://webarena.dev/?utm_source=chatgpt.com "WebArena: A Realistic Web Environment for Building Autonomous ..."
[83]: https://miniwob.farama.org/index.html?utm_source=chatgpt.com "MiniWoB++ Documentation"
[84]: https://github.com/Farama-Foundation/miniwob-plusplus?utm_source=chatgpt.com "Farama-Foundation/miniwob-plusplus: MiniWoB++"
[85]: https://docs.nvidia.com/nemo-guardrails/index.html?utm_source=chatgpt.com "NVIDIA NeMo Guardrails"
[86]: https://guardrailsai.com/docs/?utm_source=chatgpt.com "Introduction | Your Enterprise AI needs Guardrails"
[87]: https://platform.openai.com/docs/guides/moderation?utm_source=chatgpt.com "Moderation - OpenAI API"
[88]: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/?utm_source=chatgpt.com "Azure AI Content Safety documentation"
[89]: https://docs.lakera.ai/docs/quickstart?utm_source=chatgpt.com "Getting Started with Lakera Guard"
[90]: https://www.lakera.ai/blog/guide-to-prompt-injection?utm_source=chatgpt.com "Prompt Injection & the Rise of Prompt Attacks"
[91]: https://github.com/protectai/rebuff?utm_source=chatgpt.com "protectai/rebuff: LLM Prompt Injection Detector"

