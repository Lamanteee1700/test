# pages/10_Japanese_market_news.py
import streamlit as st
import feedparser
from mistralai import Mistral
import re
from datetime import datetime

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
@st.cache_data(show_spinner=False, ttl=1800)  # Cache for 30 minutes
def fetch_news_rss(rss_urls, top_n=30):
    all_entries = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            all_entries.extend(feed.entries)
        except Exception as e:
            st.warning(f"Could not fetch from {url}: {str(e)}")
            continue
    
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
        "choose the 8-10 most pertinent financial and market-related items. For each selected article, provide:\n"
        "1. A clear, engaging headline\n"
        "2. A concise 2-3 sentence summary focusing on the key financial/business impact\n"
        "3. The original link\n\n"
        "Format each article as:\n"
        "**HEADLINE**\n"
        "Summary text here.\n"
        "Link: [URL]\n\n"
        "Prioritize articles about: stock markets, corporate earnings, economic indicators, central bank policy, "
        "major corporate developments, and market trends.\n\n"
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
        return f"AI summary unavailable: {str(e)}\n\nPlease check your Mistral API configuration in Streamlit secrets."

def parse_ai_summary(summary_text):
    """Parse the AI summary into structured articles"""
    articles = []
    
    # Split by double newlines and process each section
    sections = summary_text.split('\n\n')
    
    current_article = {}
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if this looks like a headline (starts with ** or is all caps/title case)
        if section.startswith('**') and section.endswith('**'):
            # Save previous article if exists
            if current_article and 'headline' in current_article:
                articles.append(current_article)
            
            # Start new article
            current_article = {
                'headline': section.replace('**', '').strip(),
                'summary': '',
                'link': ''
            }
        elif section.startswith('Link:') or section.startswith('link:'):
            # Extract link
            link = section.replace('Link:', '').replace('link:', '').strip()
            if current_article:
                current_article['link'] = link
        else:
            # This is summary text
            if current_article and not section.startswith('AI summary unavailable'):
                if current_article['summary']:
                    current_article['summary'] += ' ' + section
                else:
                    current_article['summary'] = section
    
    # Add the last article
    if current_article and 'headline' in current_article:
        articles.append(current_article)
    
    return articles

# --- Streamlit Page Function ---
def show_japan_news_page():
    st.set_page_config(page_title="Japanese Market News", layout="wide")
    
    # Header
    st.title("📰 Japanese Market News")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">AI-Curated Financial News from Japan</h3>
        <p style="color: #e6f3ff; margin: 0.5rem 0 0 0;">
            Latest financial news from Yahoo Japan media sources, analyzed and summarized by Mistral AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sources information
    with st.expander("📡 News Sources", expanded=False):
        st.markdown("**Monitored Sources:**")
        
        # Create a nice table for sources
        source_data = []
        for source, description in RSS_DESCRIPTIONS.items():
            source_data.append({
                "Source": source,
                "Description": description
            })
        
        st.dataframe(
            source_data, 
            use_container_width=True,
            hide_index=True
        )
        
        st.info("💡 News is updated every 30 minutes. AI summaries focus on the most financially relevant articles.")

    # Fetch news with loading indicator
    with st.spinner("🔄 Fetching latest news..."):
        news_items = fetch_news_rss(list(JAPANESE_FINANCIAL_RSS_FEEDS.values()), top_n=30)

    if not news_items:
        st.error("❌ No news articles could be retrieved at this time. Please try again later.")
        return

    st.success(f"✅ Retrieved {len(news_items)} articles from {len(JAPANESE_FINANCIAL_RSS_FEEDS)} sources")

    # Generate AI summary with progress
    with st.spinner("🤖 AI is analyzing and summarizing the news..."):
        ai_summary = summarize_news_mistral(news_items)

    # Parse the summary into structured articles
    articles = parse_ai_summary(ai_summary)

    if not articles:
        # Fallback: display raw summary if parsing fails
        st.warning("⚠️ Could not parse AI summary into structured articles. Showing raw summary:")
        st.markdown(ai_summary)
        return

    # Display articles in a nice layout
    st.subheader(f"📈 Top Financial News ({len(articles)} articles)")
    
    # Display articles in cards
    for i, article in enumerate(articles):
        if not article.get('headline'):
            continue
            
        # Determine icon based on content
        headline_lower = article['headline'].lower()
        if any(word in headline_lower for word in ['株', 'stock', '市場', 'market', '上昇', '下落']):
            icon = "📈"
        elif any(word in headline_lower for word in ['企業', 'company', '決算', 'earnings']):
            icon = "🏢" 
        elif any(word in headline_lower for word in ['銀行', 'bank', '金融', 'finance']):
            icon = "🏦"
        elif any(word in headline_lower for word in ['政策', 'policy', '政府', 'government']):
            icon = "🏛️"
        else:
            icon = "💼"
        
        # Create article card
        with st.container():
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #2a5298;
            ">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <div style="font-size: 2rem; margin-top: 0.2rem;">{icon}</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.8rem 0; color: #1e3c72; font-size: 1.1rem; line-height: 1.3;">
                            {article['headline']}
                        </h4>
                        <p style="margin: 0 0 1rem 0; color: #333; line-height: 1.5; font-size: 0.95rem;">
                            {article['summary']}
                        </p>
                        {f'<a href="{article["link"]}" target="_blank" style="color: #2a5298; text-decoration: none; font-weight: 500; font-size: 0.9rem;">🔗 Read full article</a>' if article.get('link') else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Add refresh button and timestamp
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Refresh News", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Footer with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Last Updated:</strong> {current_time}</p>
        <p><strong>Financial Derivatives Dashboard</strong> | Educational platform for options, swaps, and structured products analysis</p>
        <p>Built with Streamlit • Powered by Yahoo Finance & Mistral AI</p>
    </div>
    """, unsafe_allow_html=True)

# --- Run page independently ---
if __name__ == "__main__":
    show_japan_news_page()
