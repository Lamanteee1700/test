# pages/10_Japanese_market_news.py
import streamlit as st
import feedparser
from mistralai import Mistral
import re
from datetime import datetime

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

    # === GLOBAL MARKET OUTLOOK ===
    st.markdown("---")
    st.subheader("🌍 Global Market Outlook & Japan Investment Perspective")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #28a745;">
        <h4 style="color: #155724; margin-top: 0;">Key Global Investment Themes</h4>
        <p style="color: #495057; margin-bottom: 0;">
            Based on institutional analysis and current market conditions affecting Japan and global markets
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    outlook_col1, outlook_col2 = st.columns(2)
    
    with outlook_col1:
        st.markdown("#### Japan's Investment Position")
        
        st.markdown("""
        **Macro Stability Amid Global Uncertainty**
        
        Japan maintains relative stability as global trade tensions escalate. Despite potential tariff risks, 
        the country's economic fundamentals remain solid with continued wage growth averaging 2.8% in H2 2024 
        and inflation stabilizing around target levels after decades of deflation.
        
        **Key Supportive Factors:**
        - **Mild Reflation Success**: Fourth consecutive year of 2%+ inflation
        - **Corporate Governance Revolution**: Share buybacks up 96% year-over-year
        - **Tourism Boom**: Record 36.87 million visitors contributing 7.5% of GDP
        - **Capital Investment**: Accelerating capex spending amid AI and productivity focus
        """)
        
        st.markdown("""
        **Trade Relationship Dynamics**
        
        The US-Japan relationship appears stable under current leadership, with recent meetings suggesting 
        Japan may avoid targeted tariffs. Japan's bilateral trade surplus with the US has moderately 
        declined over two decades and remains smaller relative to China and Europe deficits.
        """)
    
    with outlook_col2:
        st.markdown("#### Market Positioning & Outlook")
        
        st.markdown("""
        **Equity Market Thesis**
        
        Japanese equities remain attractive despite two consecutive years of 20%+ returns. 
        The market continues to offer value with TOPIX trading at P/E ratios 30% below 
        the S&P 500, while corporate earnings have matched US performance since 2010.
        
        **Multi-Year Investment Drivers:**
        - **Sustainable Inflation**: Breaking decades of deflationary mindset
        - **Corporate Reform**: Improved ROE and shareholder-friendly policies  
        - **Under-Ownership**: International and domestic investors remain underweight
        """)
        
        # Market outlook metrics
        st.markdown("""
        <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>2025 Market Expectations:</strong><br>
            • TOPIX Target: 3,075 - 3,175 (low-teens upside)<br>
            • Earnings Growth: 9.0% (2025), 8.5% (2026)<br>
            • Forward P/E Multiple: 15x<br>
            • Preferred Sectors: Financials, Industrials, Consumer Discretionary
        </div>
        """, unsafe_allow_html=True)
    
    # Policy and Currency Section
    st.markdown("#### Monetary Policy & Currency Dynamics")
    
    policy_col1, policy_col2 = st.columns(2)
    
    with policy_col1:
        st.markdown("""
        **Bank of Japan Approach**
        
        The BoJ maintains a gradual normalization path following its recent 25bp hike. 
        While some board members advocate for bringing rates to 1% (near neutral range), 
        current economic conditions don't suggest urgency for aggressive tightening.
        
        **Policy Considerations:**
        - Stable wage growth supporting reflation cycle
        - Global uncertainty favoring cautious approach  
        - Political pressure for JPY strength balanced against economic needs
        """)
    
    with policy_col2:
        st.markdown("""
        **Currency Outlook**
        
        The Japanese yen has emerged as a top performer among major currencies, 
        though movements remain closely tied to US Treasury yield changes. 
        Further JPY strength likely requires additional US rate declines.
        
        **Investment Implications:**
        - Increasingly comfortable with unhedged Japanese equity exposure
        - JPY serves as natural hedge against weaker global risk sentiment
        - Defensive positioning amid geopolitical uncertainties
        """)
    
    # Risk Assessment
    st.markdown("#### Risk Assessment & Strategic Considerations")
    
    st.markdown("""
    <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <strong>Key Risks to Monitor:</strong><br>
        • Global trade policy escalation despite current US-Japan stability<br>
        • Spillover effects from broader trade uncertainty affecting business investment<br>
        • Consumption recovery pace dependent on sustained wage growth<br>
        • Manufacturing PMI recovery sustainability for earnings acceleration
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #d1ecf1; border: 1px solid #bee5eb; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <strong>Strategic Opportunities:</strong><br>
        • Structural undervaluation relative to global peers<br>
        • Corporate governance improvements driving shareholder returns<br>
        • Productivity investments potentially breaking growth ceiling<br>
        • Tourism sector providing sustained economic support
    </div>
    """, unsafe_allow_html=True)
    
    # Source attribution
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.9rem; color: #6c757d; font-style: italic; text-align: center; padding: 1rem;">
        <strong>Source:</strong> Analysis based on insights from J.P. Morgan Private Bank Asia Investment Strategy, 
        "Why Japan remains one of our top calls" (February 13, 2025). Market data integrated from real-time sources.
    </div>
    """, unsafe_allow_html=True)


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
    for i, entry in enumerate(news_entries):
        combined_text += f"[{i+1}] Title: {entry.title}\nSummary: {getattr(entry, 'summary', 'No summary available')}\nLink: {entry.link}\n\n"

    prompt = f"""You are a financial news analyst specializing in Japanese markets. Your task is to select and summarize the most important financial news from the provided articles.

INSTRUCTIONS:
1. Select the 6-8 most relevant articles about Japanese finance, business, and markets
2. Focus on: stock market movements, corporate earnings, economic indicators, Bank of Japan policy, major corporate developments, industry trends, and market analysis
3. For each selected article, provide a structured response in EXACTLY this format:

ARTICLE_START
HEADLINE: [Write a clear, engaging headline in English]
SUMMARY: [Write 2-3 sentences explaining the key financial impact and significance]
LINK: [Include the original URL]
ARTICLE_END

EXAMPLE FORMAT:
ARTICLE_START
HEADLINE: Tokyo Stock Exchange Rises on Tech Sector Gains
SUMMARY: The Nikkei 225 gained 2.1% as technology stocks led the market higher amid positive earnings reports. Investors are optimistic about the sector's growth prospects for the next quarter. This follows the Bank of Japan's dovish stance on monetary policy.
LINK: https://example.com/news-article
ARTICLE_END

NEWS ARTICLES TO ANALYZE:
{combined_text}

Remember: Use EXACTLY the format shown above with ARTICLE_START and ARTICLE_END markers. This is crucial for proper parsing."""

    try:
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI_SUMMARY_ERROR: {str(e)}"

def parse_ai_summary(summary_text):
    """Parse the AI summary into structured articles using improved parsing logic"""
    articles = []
    
    # Check for API error first
    if summary_text.startswith("AI_SUMMARY_ERROR:"):
        return []
    
    # Method 1: Try structured parsing with ARTICLE_START/END markers
    if "ARTICLE_START" in summary_text and "ARTICLE_END" in summary_text:
        article_blocks = summary_text.split("ARTICLE_START")[1:]  # Skip the first empty element
        
        for block in article_blocks:
            if "ARTICLE_END" not in block:
                continue
                
            # Clean the block
            article_content = block.split("ARTICLE_END")[0].strip()
            
            # Extract components
            article = {"headline": "", "summary": "", "link": ""}
            
            lines = article_content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("HEADLINE:"):
                    article["headline"] = line.replace("HEADLINE:", "").strip()
                    current_section = "headline"
                elif line.startswith("SUMMARY:"):
                    article["summary"] = line.replace("SUMMARY:", "").strip()
                    current_section = "summary"
                elif line.startswith("LINK:"):
                    article["link"] = line.replace("LINK:", "").strip()
                    current_section = "link"
                else:
                    # Continue previous section
                    if current_section == "summary":
                        article["summary"] += " " + line
                    elif current_section == "headline" and not article["summary"]:
                        article["headline"] += " " + line
            
            # Only add if we have at least a headline
            if article["headline"]:
                articles.append(article)
    
    # Method 2: Fallback parsing for different formats
    if not articles:
        # Try to parse markdown-style format
        sections = re.split(r'\n\s*\n', summary_text)
        current_article = {}
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check for headlines (various formats)
            if (section.startswith('**') and section.endswith('**')) or \
               (section.startswith('#') and len(section.split('\n')) == 1) or \
               (len(section) < 100 and section.isupper()) or \
               (section.startswith('HEADLINE:')) or \
               (len(section.split('\n')) == 1 and len(section) < 120 and not section.startswith('http')):
                
                # Save previous article
                if current_article.get('headline'):
                    articles.append(current_article)
                
                # Start new article
                headline = section.replace('**', '').replace('#', '').replace('HEADLINE:', '').strip()
                current_article = {
                    'headline': headline,
                    'summary': '',
                    'link': ''
                }
            
            elif section.startswith(('Link:', 'URL:', 'http://', 'https://')):
                # Extract link
                if current_article:
                    link = section
                    for prefix in ['Link:', 'URL:', 'LINK:']:
                        link = link.replace(prefix, '').strip()
                    current_article['link'] = link
            
            else:
                # This should be summary text
                if current_article and len(section) > 20:  # Avoid very short fragments
                    if current_article['summary']:
                        current_article['summary'] += ' ' + section
                    else:
                        current_article['summary'] = section
        
        # Add the last article
        if current_article.get('headline'):
            articles.append(current_article)
    
    # Method 3: Final fallback - create articles from any meaningful content
    if not articles and len(summary_text) > 100:
        # Split by double newlines and try to create at least one article
        paragraphs = [p.strip() for p in summary_text.split('\n\n') if p.strip()]
        
        if paragraphs:
            # Use the first substantial paragraph as headline, rest as summary
            headline = paragraphs[0]
            # Remove common prefixes/suffixes that might make it look bad
            headline = re.sub(r'^(AI summary|Summary|Article|News):\s*', '', headline, flags=re.IGNORECASE)
            headline = headline.replace('**', '').strip()
            
            summary_parts = paragraphs[1:] if len(paragraphs) > 1 else [summary_text[:500] + "..."]
            summary = ' '.join(summary_parts)
            
            articles.append({
                'headline': headline[:150] if len(headline) > 150 else headline,  # Reasonable length
                'summary': summary,
                'link': ''
            })
    
    # Clean up articles and ensure quality
    cleaned_articles = []
    for article in articles:
        # Skip articles with very short or poor quality content
        if (len(article.get('headline', '')) < 10 or 
            len(article.get('summary', '')) < 30 or
            'AI summary unavailable' in article.get('summary', '')):
            continue
            
        # Clean up headline
        headline = article['headline']
        headline = re.sub(r'^[\d\.\)]+\s*', '', headline)  # Remove numbering
        headline = headline.replace('**', '').strip()
        
        # Clean up summary
        summary = article['summary']
        summary = re.sub(r'Link:\s*https?://[^\s]+', '', summary)  # Remove inline links
        summary = summary.strip()
        
        cleaned_articles.append({
            'headline': headline,
            'summary': summary,
            'link': article.get('link', '')
        })
    
    return cleaned_articles[:8]  # Limit to 8 articles max

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

    # Handle API errors gracefully
    if ai_summary.startswith("AI_SUMMARY_ERROR:"):
        error_msg = ai_summary.replace("AI_SUMMARY_ERROR:", "").strip()
        st.error(f"❌ AI Summary Error: {error_msg}")
        st.info("💡 Please check your Mistral API key in Streamlit secrets or try refreshing the page.")
        
        # Fallback: Show recent raw headlines
        st.subheader("📰 Recent Headlines (Fallback)")
        for i, item in enumerate(news_items[:10]):
            with st.expander(f"📄 {item.title[:100]}..."):
                st.write(f"**Source:** {item.get('source', 'Unknown')}")
                st.write(f"**Summary:** {getattr(item, 'summary', 'No summary available')[:300]}...")
                st.write(f"**Link:** {item.link}")
        return

    # Parse the summary into structured articles
    articles = parse_ai_summary(ai_summary)

    if not articles:
        # Show debugging info and raw content
        st.warning("⚠️ Could not parse AI summary into structured articles.")
        
        # Show the raw summary for debugging
        with st.expander("🔍 Debug: Raw AI Response", expanded=True):
            st.text_area("Raw AI Summary:", ai_summary, height=300)
        
        # Try to show something useful
        st.markdown("**Raw Summary:**")
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
