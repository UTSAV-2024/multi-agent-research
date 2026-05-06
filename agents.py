from groq import Groq
from dotenv import load_dotenv
from ddgs import DDGS
import os
import json
import time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── BASE AGENT ───
def run_agent(system_prompt, user_prompt, max_tokens=1000):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# ─── AGENT 1: SEARCH AGENT ───
def search_agent(topic, num_results=5):
    print(f"\n[SEARCH AGENT] searching for: {topic}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=num_results))
    except Exception as e:
        return []
    
    sources = []
    for r in results:
        sources.append({
            "title": r.get('title', ''),
            "url": r.get('href', ''),
            "snippet": r.get('body', '')
        })
    
    print(f"[SEARCH AGENT] found {len(sources)} sources")
    return sources

# ─── AGENT 2: SUMMARIZER AGENT ───
def summarizer_agent(topic, sources):
    print(f"\n[SUMMARIZER AGENT] extracting key facts from {len(sources)} sources")
    
    summaries = []
    
    for i, source in enumerate(sources):
        print(f"[SUMMARIZER AGENT] processing source {i+1}/{len(sources)}")
        
        prompt = f"""Extract the 3-5 most important facts from this source about: {topic}

Source title: {source['title']}
Source content: {source['snippet']}

Return ONLY a JSON array of facts like this:
["fact 1", "fact 2", "fact 3"]

No other text. Just the JSON array."""

        result = run_agent(
            "You are a precise fact extractor. Extract only verifiable facts. Return only valid JSON.",
            prompt,
            max_tokens=300
        )
        
        try:
            # clean the result
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            facts = json.loads(result)
            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": facts
            })
        except:
            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": [source['snippet'][:200]]
            })
    
    print(f"[SUMMARIZER AGENT] extracted facts from {len(summaries)} sources")
    return summaries

# ─── AGENT 3: FACT-CHECK AGENT ───
def factcheck_agent(topic, summaries):
    print(f"\n[FACT-CHECK AGENT] cross-referencing facts across sources")
    
    all_facts = []
    for s in summaries:
        for fact in s['facts']:
            all_facts.append(f"- {fact} (source: {s['source']})")
    
    facts_text = "\n".join(all_facts)
    
    prompt = f"""Topic: {topic}

Here are facts extracted from multiple sources:
{facts_text}

Analyze these facts and:
1. Identify facts that AGREE across multiple sources (high confidence)
2. Identify facts that CONTRADICT each other (flag as disputed)
3. Identify facts mentioned only once (lower confidence)

Return a JSON object like this:
{{
    "confirmed_facts": ["fact that multiple sources agree on"],
    "disputed_facts": ["fact A contradicts fact B"],
    "single_source_facts": ["fact mentioned only once"]
}}

Return ONLY valid JSON. No other text."""

    result = run_agent(
        "You are a rigorous fact-checker. Identify agreements and contradictions across sources. Return only valid JSON.",
        prompt,
        max_tokens=800
    )
    
    try:
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        verified = json.loads(result)
    except:
        verified = {
            "confirmed_facts": [f['facts'][0] for f in summaries if f['facts']],
            "disputed_facts": [],
            "single_source_facts": []
        }
    
    print(f"[FACT-CHECK AGENT] confirmed: {len(verified.get('confirmed_facts', []))} facts")
    print(f"[FACT-CHECK AGENT] disputed: {len(verified.get('disputed_facts', []))} facts")
    return verified

# ─── AGENT 4: REPORT AGENT ───
def report_agent(topic, summaries, verified_facts):
    print(f"\n[REPORT AGENT] compiling final report")
    
    confirmed = "\n".join(f"- {f}" for f in verified_facts.get('confirmed_facts', []))
    disputed = "\n".join(f"- {f}" for f in verified_facts.get('disputed_facts', []))
    single = "\n".join(f"- {f}" for f in verified_facts.get('single_source_facts', []))
    sources = "\n".join(f"- {s['source']}: {s['url']}" for s in summaries)
    
    prompt = f"""Write a comprehensive research report on: {topic}

CONFIRMED FACTS (multiple sources agree):
{confirmed}

DISPUTED INFORMATION:
{disputed if disputed else "None identified"}

ADDITIONAL INFORMATION (single source):
{single}

SOURCES CONSULTED:
{sources}

Write a well-structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (confirmed facts)
3. Areas of Uncertainty (disputed or single-source)
4. Sources

Be factual. Be concise. Do not add information not in the facts provided."""

    report = run_agent(
        "You are a professional research analyst. Write clear, factual reports based only on provided information.",
        prompt,
        max_tokens=1500
    )
    
    print(f"[REPORT AGENT] report compiled")
    return report

# ─── ORCHESTRATOR ───
def research(topic):
    print(f"\n{'='*60}")
    print(f"MULTI-AGENT RESEARCH SYSTEM")
    print(f"Topic: {topic}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # agent 1: search
    sources = search_agent(topic, num_results=5)
    if not sources:
        return "Error: could not find sources for this topic."
    
    # agent 2: summarize
    summaries = summarizer_agent(topic, sources)
    
    # agent 3: fact-check
    verified = factcheck_agent(topic, summaries)
    
    # agent 4: report
    report = report_agent(topic, summaries, verified)
    
    elapsed = round(time.time() - start, 2)
    
    print(f"\n{'='*60}")
    print(f"RESEARCH COMPLETE in {elapsed}s")
    print(f"{'='*60}\n")
    print(report)
    
    return report, summaries, verified, elapsed

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    research(topic)