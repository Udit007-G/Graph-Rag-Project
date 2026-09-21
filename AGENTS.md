Why This Is Hard

A basic RAG pipeline retrieves some text and generates an answer. That works for simple questions. GraphRAG adds structure — entities, relationships, community context — and that works for harder ones.
But some questions need the system to actually think. Break the problem down. Decide what to look up. Evaluate what came back. Realize something's missing. Go back and retrieve more. Change strategy mid-investigation. Decide when it has enough to answer. That's the agentic part, and that's where this challenge lives.
The Challenge

Core (everyone builds this): An Agentic GraphRAG system that can plan and execute a multi-step investigation. Your system should include:

An agent harness to manage state, tools, context, evidence, and stopping criteria
An orchestrator agent that determines what needs investigating and picks the next action — not a fixed retrieval sequence
Specialized agents for retrieval and reasoning tasks: entity linking, graph traversal, similarity search, document retrieval, aggregation, multi-hop reasoning, and evidence evaluation
The orchestrator's next move should depend on the original question, the graph, what evidence it already has, and what's still missing.

The benchmark: Every question gets answered three ways — RAG, GraphRAG, and Agentic GraphRAG — so we can see exactly where each approach succeeds or fails.

Stretch (for teams who want to go further): Reasoning over time. Real-world facts change — launch dates move, plans get revised, and sources conflict. Extend your agent to detect conflicting versions of a fact, figure out what supersedes what, determine which sources are more authoritative, and handle uncertainty. This is where strong teams pull ahead.

Two ground rules:

Build the benchmark. Your solution must compare all three approaches (RAG, GraphRAG, Agentic GraphRAG) using accuracy, completeness, and token efficiency metrics. That comparison is the point.
Add-ons are welcome. Beyond the core and stretch, build whatever makes your system smarter or more useful. Creative add-ons count in your favour under engineering and innovation.
What You Build With

Savanna: Host the graph and vector DB (free credits provided)
GSQL + graph algorithms: Traversal, pattern matching, community detection, and investigation logic
TigerGraph Vector DB: Semantic search across unstructured documents
TigerGraph MCP: Connect agents and development assistants to TigerGraph
Dataset: We provide a dataset with questions of increasing complexity. You're free to bring your own too. The provided dataset is the common benchmark everyone is scored on, so use your own as a bonus to show your system can handle real or messy data.

How the Two Rounds Work

Round 1 (open to everyone): Submit a working Agentic GraphRAG system covering the core challenge.
Round 2 (top 15 teams advance): Extend the agent to reason over evolving, conflicting, and uncertain facts on a harder dataset. Finalists also submit a demo video and write-up. Top teams present live to the judging panel.
Timeline

September 1 (Tue): Registration opens; guidebook and dataset go live
September 12 (Sat): Registration closes
September 24 (Thu): Round 1 submission deadline
October 1 (Thu): Round 2 / final submission deadline
October 2 to 5 (Fri to Mon): Judging
October 7 (Wed): Results announced
What to Submit

Round 1: Working Agentic GraphRAG system + GitHub repository + architecture diagram + demo video + metrics dashboard (tokens, accuracy, and completeness across all three pipelines). Optional: A social media post about what you're building (counts in your favour).
Round 2 (finalists): Refined system + updated repository + architecture diagram + 3–5 minute demo video + metrics dashboard + short write-up (what you built, how it works, key results, limitations, and what you'd do with more time).
How It's Judged

Accuracy and completeness are scored against a held-out ground-truth dataset. Quality is assessed via automated evaluation, LLM-as-judge, human review, and repository review.

Investigation accuracy, 30%: Answers complex questions correctly using the right evidence across retrieval and reasoning steps
Evidence quality and explainability, 15%: Grounded answers with clear citations and a clear investigation path
Agentic effectiveness and efficiency, 15%: Picks the right retrieval methods, uses agentic steps where they add value, and balances accuracy with token cost
Agentic design, engineering, and code quality, 15%: Architecture, tool use, reliability, reproducibility, and repository quality
Innovation, 15%: Novel investigation methods, graph reasoning, or user experience
Final presentation and Q&A, 10%: Demo quality, technical clarity, and responses to judges
Prizes

# Note :
Read the "corpus" and "questions" folder also. and also check this website for the guide :  https://alluring-beryllium-491.notion.site/Agentic-GraphRAG-Hackathon-Guidebook-34fc2cb129c08146998af3568d7d2594.