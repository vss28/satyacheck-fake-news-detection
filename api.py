"""
FastAPI Backend — Indian Fake News Detection & Real-Time Feed
=============================================================
Serves the modern frontend UI, provides analysis endpoint with XLM-RoBERTa +
Source Credibility post-processing, and integrates live Indian news feeds.
"""

import os
import urllib.parse
import xml.etree.ElementTree as ET
import urllib.request
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize pipeline lazily so FastAPI starts instantly
app = FastAPI(title="Indian Fake News AI Engine", version="2.0")

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from predict_india import IndianFakeNewsPipeline
        _pipeline = IndianFakeNewsPipeline()
    return _pipeline

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class AnalyzeRequest(BaseModel):
    text: str
    source: Optional[str] = ""

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Frontend building in progress...</h2>")

@app.post("/api/analyze")
async def analyze_news(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        return JSONResponse({"error": "Please provide some text to analyze."}, status_code=400)

    try:
        print(f"🔍 [Analyze Request] Text: '{req.text.strip()[:60]}...' | Source: '{req.source}'")
        pipeline = get_pipeline()
        result = pipeline.predict(req.text.strip(), req.source or "")
        print(f"   -> Verdict: {result.get('verdict')} (Confidence: {result.get('confidence')})")
        return JSONResponse(result)
    except Exception as e:
        print(f"❌ [Analyze Error] {str(e)}")
        return JSONResponse({"error": f"Inference error: {str(e)}"}, status_code=500)

@app.get("/api/health")
async def health_check():
    pipeline = get_pipeline()
    return {
        "status": "online",
        "model": "XLM-RoBERTa (Multilingual Indian)",
        "custom_trained": pipeline.is_custom_trained,
        "has_ensemble": pipeline.has_ensemble
    }

import html
import re

def clean_html_snippet(raw_html: str) -> str:
    """Removes HTML tags and cleans up Google News RSS description."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.get("/api/search_news")
@app.get("/api/live_feed")
async def search_live_indian_news(query: str = "India News", limit: int = 15, lang: str = "en"):
    """
    Searches real-time Indian news from Google News RSS.
    Returns 15-25 matching articles with extracted snippets and AI fact-check analysis.
    Supports English ('en') and Hindi ('hi').
    """
    if not query or not query.strip():
        query = "India Breaking News"

    try:
        print(f"📡 [Live Search] Query: '{query}' | Lang: {lang} | Limit: {limit}")
        encoded_query = urllib.parse.quote(query.strip())
        
        if lang == "hi":
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=hi&gl=IN&ceid=IN:hi"
        else:
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")
        max_items = min(50, max(5, int(limit)))
        selected_items = items[:max_items]

        articles = []
        pipeline = get_pipeline()

        for item in selected_items:
            try:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                desc_raw = item.find("description").text if item.find("description") is not None else ""
                snippet = clean_html_snippet(desc_raw)

                source_tag = item.find("source")
                source_name = source_tag.text if source_tag is not None else ""
                if not source_name and " - " in title:
                    source_name = title.rsplit(" - ", 1)[-1].strip()

                clean_title = title.rsplit(" - ", 1)[0].strip() if " - " in title else title
                if not clean_title:
                    continue

                analysis = pipeline.predict(clean_title, source_name)

                articles.append({
                    "title": clean_title,
                    "full_title": title,
                    "snippet": snippet[:250] + "..." if len(snippet) > 250 else snippet,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source_name,
                    "analysis": analysis
                })
            except Exception as item_err:
                print(f"⚠️ [Search Item Warning] {item_err}")
                continue

        print(f"✅ [Live Search Done] Returning {len(articles)} analyzed articles.")

        return {
            "query": query,
            "count": len(articles),
            "articles": articles
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to search news: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Indian Fake News AI Server on http://localhost:8000 ...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
