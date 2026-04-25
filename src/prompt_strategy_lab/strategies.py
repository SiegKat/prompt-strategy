"""Prompt augmentation strategies.

Each strategy implements a common interface: given a raw user prompt,
return an augmented prompt ready to send to an LLM. The strategies wrap
the user prompt with structural instructions that elicit different
reasoning patterns from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    """Base class. Subclasses override `template` with a format string
    containing a `{user_prompt}` placeholder."""

    name: str = ""
    description: str = ""
    template: str = "{user_prompt}"

    def augment(self, user_prompt: str, **extras: str) -> str:
        return self.template.format(user_prompt=user_prompt, **extras).strip()


@dataclass
class ChainOfThought(Strategy):
    name: str = "chain_of_thought"
    description: str = "Instructs the model to reason step-by-step before answering."
    template: str = """
You are a senior product strategist.
Think step by step about the user's request before answering.

Task: {user_prompt}

Process:
1. Analyze the user's request and the target context.
2. Consider the different stakeholders or users involved.
3. Think about the challenges and pain points relevant to the request.
4. Brainstorm potential ideas, problems, or solutions.
5. For each item, identify the underlying rationale, any assumptions made, and a clarifying question.
6. List at least 5 items based on this analysis.

Each item should include:
- The core idea or pain point
- Assumptions made
- A clarifying question
"""


@dataclass
class Reflection(Strategy):
    name: str = "reflection"
    description: str = "Generates an answer, then asks the model to critique its own output."
    template: str = """
{user_prompt}

After producing your answer, review your own output:
- Identify weaknesses, gaps, or biases.
- Suggest 2 improvements or alternative framings.
"""


@dataclass
class AlternativeApproaches(Strategy):
    name: str = "alternative_approaches"
    description: str = "Forces the model to produce multiple framings from different perspectives."
    template: str = """
{user_prompt}

Produce three alternative perspectives on this request:
1. From end users
2. From operators or instructors
3. From administrators or executives

For each perspective, list 2-3 distinct items.
"""


@dataclass
class Template(Strategy):
    name: str = "template"
    description: str = "Constrains output to a fixed structural template."
    template: str = """
You are a Solution Design Bot.

Task: {user_prompt}

Use the following template:
Problem Summary:
Solution Overview:
Core Features:
Differentiators:
Risks & Open Questions:
Success Metrics:
"""


@dataclass
class Comparative(Strategy):
    name: str = "comparative"
    description: str = "Generates multiple candidate solutions and recommends the best."
    template: str = """
You are a Solution Design Bot.

Task: Generate and compare three different approaches for: {user_prompt}

For each approach, include:
- Core idea
- Key benefits
- Trade-offs
- Risk factors

Conclude with a recommendation: which approach is most viable and why?
"""


@dataclass
class FactCheckList(Strategy):
    name: str = "fact_check_list"
    description: str = "Separates verifiable facts from assumptions for each output item."
    template: str = """
You are a Requirements Engineer Bot.

Task: {user_prompt}

For each requirement or item:
1. Provide the statement.
2. List factual claims (that can be verified).
3. List assumptions (that need validation).
4. Suggest one clarifying question.

Format:
Item: ...
Facts: ...
Assumptions: ...
Question: ...
"""


@dataclass
class FewShot(Strategy):
    name: str = "few_shot"
    description: str = "Conditions the model with examples of the desired output style."
    template: str = """
You are a Solution Design Bot.

Here are examples of the kind of output expected:

Example A:
project/
  app/
    __init__.py
    models.py
    routes.py
    services.py
  tests/
    test_app.py
  requirements.txt
  run.py

Example B:
backend/
  src/
    api/
      controllers.py
      serializers.py
    core/
      config.py
      database.py
    features/
      users.py
      ai_engine.py
  tests/
    unit/
    integration/
  pyproject.toml

Now, in a similar style, address: {user_prompt}

Include:
- Folder layout
- Key files with one-line purpose each
- Short description of each component
"""


@dataclass
class CoTWithReflection(Strategy):
    name: str = "cot_with_reflection"
    description: str = "Two-pass: silent planning, then silent reflection, then final answer."
    template: str = """
You are a Solution Design Bot.

Goal: {user_prompt}

How to think:
1. Do all deep reasoning silently. Do not reveal your chain of thought.
2. Work in two internal stages:
   2a. Plan stage:
       - unpack the problem into subparts
       - consider two or three plausible designs
       - choose one primary design with tradeoffs noted
   2b. Reflection stage:
       - verify that the output format is complete
       - search for missing pieces or contradictions
       - correct any gaps you find

Style rules:
- Be concise and specific.
- Use clear section headers.
- Do not print hidden reasoning.
- Prefer short sentences.

Quality checklist to run silently before answering:
- Does the output map to every requirement in the user request?
- Are security, failure handling, and observability addressed?
- Is the plan feasible for a small first release and extendable later?

Answer now with the final structured result only.
"""


@dataclass
class SelfConsistency(Strategy):
    name: str = "self_consistency"
    description: str = "Internally generates several candidates and returns only the best-merged result."
    template: str = """
You are a Solution Design Bot.

Method: Generate several candidate designs internally and compare them. Do not show the candidates. Select and present only the best consistent plan.

Internal scoring (run silently):
Score each candidate from 1 to 5 on:
- coverage of requirements
- cohesion of modules and clarity of boundaries
- feasibility for a small first release
- scalability path and cost awareness
- security and data protection
Choose the highest-score candidate, or merge parts to maximize the score.

User request: {user_prompt}

Tone and format: concise, professional, clear headers, no hidden notes.
"""


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in [
        ChainOfThought(),
        Reflection(),
        AlternativeApproaches(),
        Template(),
        Comparative(),
        FactCheckList(),
        FewShot(),
        CoTWithReflection(),
        SelfConsistency(),
    ]
}
