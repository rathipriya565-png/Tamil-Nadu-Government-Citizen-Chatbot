
import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tamil Nadu Government Citizen Chatbot",
    layout="wide"
)

st.title("🏛️ Tamil Nadu Government Citizen Chatbot")
st.caption("ML + RAG Based Citizen Service Assistant")


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("Tamil_Nadu_Combined_Dataset (3).csv")


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df["clean_query"] = df["user_query"].apply(clean_text)


# =========================================================
# ML MODEL
# =========================================================

X = df["clean_query"]
y = df["intent"]

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=3000
)

X_tfidf = tfidf.fit_transform(X)

final_model = LinearSVC(
    C=1.0,
    random_state=42
)

final_model.fit(X_tfidf, y)


# =========================================================
# INTENT PREDICTION
# =========================================================

def predict_intent(user_query):

    query_lower = user_query.lower()

    document_patterns = [
        "document",
        "documents",
        "doc",
        "docs",
        "proof",
        "papers",
        "what documents",
        "enna documents",
        "ena documents",
        "documents venum",
        "documents thevai",
        "enna proof",
        "ena proof",
        "proof venum",
        "proof thevai",
        "ஆவணம்",
        "ஆவணங்கள்",
        "ஆவணங்கள் தேவை",
        "என்ன ஆவணங்கள்",
        "என்ன ஆவணம்",
        "தேவையான ஆவணங்கள்",
        "சான்று",
        "சான்றுகள்"
    ]

    benefit_patterns = [
        "benefit",
        "benefits",
        "benefit enna",
        "what benefit",
        "enna benefit",
        "பயன்",
        "பயன்கள்",
        "என்ன பயன்",
        "என்ன பயன்கள்",
        "நன்மை",
        "நன்மைகள்"
    ]

    eligibility_patterns = [
        "eligibility",
        "eligible",
        "who can apply",
        "yaar apply",
        "yaar apply pannalam",
        "தகுதி",
        "தகுதியானவர்",
        "யார் விண்ணப்பிக்கலாம்",
        "யாரெல்லாம் விண்ணப்பிக்கலாம்",
        "யார் விண்ணப்பிக்க முடியும்"
    ]

    application_patterns = [
        "how to apply",
        "apply panna",
        "apply panrathu",
        "eppadi apply",
        "eppadi apply panrathu",
        "application process",
        "விண்ணப்பிப்பது எப்படி",
        "எப்படி விண்ணப்பிப்பது",
        "விண்ணப்பிப்பது",
        "விண்ணப்பிக்க"
    ]

    if any(x in query_lower for x in document_patterns):
        return "document_help"

    if any(x in query_lower for x in benefit_patterns):
        return "benefit_query"

    if any(x in query_lower for x in eligibility_patterns):
        return "eligibility_query"

    if any(x in query_lower for x in application_patterns):
        return "application_help"

    query_vector = tfidf.transform(
        [clean_text(user_query)]
    )

    return final_model.predict(query_vector)[0]


# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

rag_df = df.copy()

text_columns = [
    "service_name",
    "department",
    "category",
    "description",
    "eligibility",
    "benefits",
    "documents_required",
    "application_process",
    "record_type"
]

available_columns = [
    col for col in text_columns
    if col in rag_df.columns
]

rag_df["knowledge_text"] = (
    rag_df[available_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)


# =========================================================
# RAG VECTOR DATABASE
# =========================================================

rag_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000
)

rag_vectors = rag_vectorizer.fit_transform(
    rag_df["knowledge_text"]
)


# =========================================================
# RAG RETRIEVAL
# =========================================================

def retrieve_documents(query, top_k=3):

    query_lower = query.lower()

    service_aliases = {
        "ஓபிசி": "other backward class",
        "ஓபிசி சான்றிதழ்": "other backward class",
        "obc certificate": "other backward class",

        "குடியிருப்பு சான்றிதழ்": "residence certificate",
        "வசிப்பிட சான்றிதழ்": "residence certificate",

        "முதல் பட்டதாரி சான்றிதழ்": "first graduate certificate"
    }

    enhanced_query = query

    for alias, english_name in service_aliases.items():

        if alias in query_lower:
            enhanced_query += " " + english_name

    query_vector = rag_vectorizer.transform(
        [enhanced_query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        rag_vectors
    ).flatten()

    scores = similarity_scores.copy()

    for i, service_name in enumerate(
        rag_df["service_name"].fillna("")
    ):

        service_lower = str(service_name).lower()

        if service_lower in enhanced_query.lower():
            scores[i] += 2.0

        if (
            "obc" in enhanced_query.lower()
            and "other backward class" in service_lower
        ):
            scores[i] += 2.0

    top_indices = scores.argsort()[-top_k:][::-1]

    results = rag_df.iloc[top_indices].copy()

    results["similarity_score"] = scores[top_indices]

    return results


# =========================================================
# CLEAN VALUES
# =========================================================

def clean_value(value):

    if pd.isna(value):
        return "The requested information is not available in the retrieved records."

    if str(value).strip().lower() == "nan":
        return "The requested information is not available in the retrieved records."

    if str(value).strip() == "":
        return "The requested information is not available in the retrieved records."

    return str(value)


# =========================================================
# RAG ANSWER GENERATION
# =========================================================

def generate_rag_answer(query, intent):

    retrieved = retrieve_documents(
        query,
        top_k=3
    )

    top_record = retrieved.iloc[0]

    service = clean_value(
        top_record.get("service_name")
    )

    department = clean_value(
        top_record.get("department")
    )

    if intent == "document_help":

        information = clean_value(
            top_record.get("documents_required")
        )

        return f"""
🏛️ **Service:** {service}

🏢 **Department:** {department}

📄 **Documents Required:**

{information}
"""

    elif intent == "eligibility_query":

        information = clean_value(
            top_record.get("eligibility")
        )

        return f"""
🏛️ **Service:** {service}

🏢 **Department:** {department}

👤 **Eligibility:**

{information}
"""

    elif intent == "benefit_query":

        information = clean_value(
            top_record.get("benefits")
        )

        return f"""
🏛️ **Service:** {service}

🏢 **Department:** {department}

🎁 **Benefits:**

{information}
"""

    elif intent == "application_help":

        information = clean_value(
            top_record.get("application_process")
        )

        return f"""
🏛️ **Service:** {service}

🏢 **Department:** {department}

📝 **Application Procedure:**

{information}
"""

    else:

        return f"""
🏛️ **Service:** {service}

🏢 **Department:** {department}

ℹ️ The requested information is not available in the retrieved records.
"""


# =========================================================
# CHATBOT
# =========================================================

st.header("💬 Ask Government Services")

st.write(
    "Ask your question in English, Tanglish or Tamil."
)

user_query = st.text_input(
    "Your Question",
    placeholder="Example: OBC certificate ku enna documents venum?"
)

if st.button("Ask"):

    if user_query.strip() == "":

        st.warning("Please enter your question.")

    else:

        intent = predict_intent(user_query)

        answer = generate_rag_answer(
            user_query,
            intent
        )

        st.subheader("🤖 Chatbot Response")

        st.markdown(answer)

        with st.expander("🔍 Technical Details"):

            st.write(
                "Predicted Intent:",
                intent
            )

            retrieved = retrieve_documents(
                user_query,
                top_k=3
            )

            st.dataframe(
                retrieved[
                    [
                        "service_name",
                        "department",
                        "similarity_score"
                    ]
                ],
                use_container_width=True
            )


# =========================================================
# DASHBOARD
# =========================================================

st.header("📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Queries",
    len(df)
)

col2.metric(
    "Service Records",
    df["service_name"].nunique()
)

col3.metric(
    "Departments",
    df["department"].nunique()
)

col4.metric(
    "Intents",
    df["intent"].nunique()
)


# =========================================================
# INTENT DISTRIBUTION
# =========================================================

st.subheader("🎯 Intent Distribution")

st.bar_chart(
    df["intent"].value_counts()
)


# =========================================================
# LANGUAGE DISTRIBUTION
# =========================================================

st.subheader("🌐 Language Distribution")

st.bar_chart(
    df["language"].value_counts()
)


# =========================================================
# DEPARTMENT DISTRIBUTION
# =========================================================

st.subheader("🏢 Department Distribution")

st.bar_chart(
    df["department"].value_counts()
)


# =========================================================
# SERVICE DISTRIBUTION
# =========================================================

st.subheader("📋 Service Distribution")

service_counts = (
    df["service_name"]
    .value_counts()
    .reset_index()
)

st.dataframe(
    service_counts,
    use_container_width=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("🤖 ML Model Performance")

try:

    model_results = pd.read_csv(
        "model_performance.csv"
    )

    st.dataframe(
        model_results.round(4),
        use_container_width=True
    )

except FileNotFoundError:

    st.info(
        "Model performance file not found."
    )
