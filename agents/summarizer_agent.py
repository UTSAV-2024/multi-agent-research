from utils.json_utils import clean_json_response, safe_json_parse
from services.llm_service import run_agent
def summarizer_agent(topic, sources):

    print(f"\n[SUMMARIZER AGENT] extracting key facts")

    summaries = []

    for i, source in enumerate(sources):

        print(f"[SUMMARIZER AGENT] processing source {i+1}/{len(sources)}")

        prompt = f"""
Extract the most important information about: {topic}

Focus on:
1. key claims
2. dates/events
3. important people
4. controversies
5. statistics

Source title:
{source['title']}

Source content:
{source['content']}

Return ONLY valid JSON in this format:

{{
    "facts": [
        "fact 1",
        "fact 2",
        "fact 3"
    ]
}}

No markdown.
No explanations.
No code blocks.

No explanations.
"""

        result = run_agent(
            "You are a precise fact extraction system. Return only valid JSON.",
            prompt,
            max_tokens=500
        )

        try:

            result = clean_json_response(result)

            parsed = safe_json_parse(
            result,
            fallback={"facts": []}
            )

            facts = parsed.get("facts", [])
            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": facts
            })

        except Exception as e:

            print(f"[SUMMARIZER AGENT] parsing failed: {e}")

            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": [
                    source['content'][:300]
                ]
            })

    print(f"[SUMMARIZER AGENT] extracted facts from {len(summaries)} sources")

    return summaries