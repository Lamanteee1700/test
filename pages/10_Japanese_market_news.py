# japan_news_page.py
import streamlit as st
import feedparser
from mistralai import Mistral

# --- Configuration: Yahoo Japan Media Feeds ---
JAPANESE_FINANCIAL_RSS_FEEDS = {
    "Forbes Japan": "https://news.yahoo.co.jp/rss/media/forbes/all.xml",
    "Diamond Online": "https://news.yahoo.co.jp/rss/media/dzai/all.xml",
    "Teikoku Databank": "https://news.yahoo.co.jp/rss/media/teikokudb/all.xml",
    "Yahoo Business": "https://news.yahoo.co.jp/rss/media/business/all.xml",
    "Finasee": "https://news.yahoo.co.jp/rss/media/finasee/all.xml"
}

RSS_DESCRIPTIONS = {
    "Forbes Japan": "Business and finance news with insights on companies and markets.",
    "Diamond Online": "Economic and financial news from a leading Japanese business magazine.",
    "Teikoku Databank": "Corporate credit reports, market trends, and industry insights.",
    "Yahoo Business": "General business news and stock market updates in Japan.",
    "Finasee": "Finance-focused news with emphasis on stocks, investments, and market trends."
}

# --- Fetch RSS with caching ---
@st.cache_data(show_spinner=False)
def fetch_news_rss(rss_urls, top_n=30):
    all_entries = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    # Sort by published date if available
    all_entries.sort(key=lambda x: getattr(x, "published_parsed", None), reverse=True)
    return all_entries[:top_n]

# --- Summarization using Mistral AI ---
def summarize_news_mistral(news_entries):
    combined_text = ""
    for entry in news_entries:
        combined_text += f"Title: {entry.title}\nSummary: {getattr(entry, 'summary', '')}\nLink: {entry.link}\n\n"

    prompt = (
        "You are an AI financial journalist. From the following Japanese business and market news, "
        "choose the most pertinent items, summarize each in 2-3 sentences, "
        "and structure the output clearly with the title, short summary, and link to the original article. "
        "Prioritize relevance to markets, finance, and corporate news:\n\n"
        f"{combined_text}"
    )

    try:
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

# --- Streamlit Page Function ---
def show_japan_news_page():
    st.title("📰 Japanese Market News (AI-Summarized)")
    st.markdown("""
    Latest financial news from **Yahoo Japan media sources**, merged and summarized by **Mistral AI**.  
    *Free tier, limited requests — summaries may take a few seconds.*
    """)

    # Display sources
    st.markdown("**Sources included:**")
    for src, desc in RSS_DESCRIPTIONS.items():
        st.markdown(
            f"<div style='padding:3px 0;'><strong>{src}</strong>: <span style='color:#555'>{desc}</span></div>",
            unsafe_allow_html=True
        )

    # Fetch and merge feeds
    news_items = fetch_news_rss(list(JAPANESE_FINANCIAL_RSS_FEEDS.values()), top_n=30)

    if news_items:
        ai_summary = summarize_news_mistral(news_items)

        # Split AI output into articles
        articles = ai_summary.split("\n\n")  # assumes AI separates articles with double line breaks

        # Two-column layout
        for i in range(0, len(articles), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(articles):
                    article = articles[i + j]
                    icon = "📈" if "株" in article or "市場" in article else "🏢"
                    col.markdown(
                        f"""
                        <div style='
                            background: linear-gradient(135deg, #e0f7fa, #ffffff);
                            border-radius: 15px;
                            padding: 15px;
                            margin-bottom: 10px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        '>
                            <div style='font-size:18px;'>{icon} {article.replace('\n', '<br>')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("No news available at the moment.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
    <strong>Financial Derivatives Dashboard</strong><br>
    Educational platform for options, swaps, and structured products analysis<br>
    Built with Streamlit, NumPy, SciPy, Plotly & Yahoo Finance
    </div>
    """, unsafe_allow_html=True)

# --- Run page independently ---
if __name__ == "__main__":
    show_japan_news_page()
