import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import json
import time
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Literal

# ==================== 1. Streamlit Configuration ====================
st.set_page_config(
    page_title="Sentiment Analysis Pipeline",
    page_icon="",
    layout="wide"
)

# Configuration
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
THRESHOLD = 0.1  # Probability threshold for bad sentiment
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for final decision

# ==================== 2. Model Architecture (Must match training) ====================
class PersianVocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
    
    def build_from_dict(self, word2idx):
        """Build vocabulary from dictionary"""
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
    
    def tokenize_persian(self, text):
        """Tokenize Persian text into words"""
        if not isinstance(text, str):
            return []
        
        # Clean text - keep Persian characters, spaces, and common punctuation
        text = re.sub(r'[^\u0600-\u06FF\s\.\,\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split by spaces
        words = text.strip().split()
        
        return words
    
    def numericalize(self, text, max_length=100):
        """Convert text to sequence of indices"""
        words = self.tokenize_persian(text)
        
        # Add SOS and EOS tokens
        if len(words) > max_length - 2:
            words = words[:max_length-2]
        words = ['<SOS>'] + words + ['<EOS>']
        
        # Convert to indices
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.word2idx['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
            indices[-1] = self.word2idx['<EOS>']
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 output_dim=3, n_layers=2, dropout=0.3):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        output = self.dropout(context_vector)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

# ==================== 3. Load Model from .pth file (with vocabulary) ====================
@st.cache_resource
def load_model_and_vocab(pth_path='best_sentiment_model.pth'):
    """Load model and vocabulary from saved .pth file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        
        # Get vocabulary from checkpoint
        if 'vocab_word2idx' in checkpoint:
            # New format: vocabulary saved in checkpoint
            vocab = PersianVocabulary()
            vocab.build_from_dict(checkpoint['vocab_word2idx'])
            max_length = checkpoint.get('max_length', 100)
        elif 'vocab' in checkpoint:
            # Old format: vocab object saved
            vocab = checkpoint['vocab']
            max_length = checkpoint.get('max_length', 100)
        else:
            # If no vocabulary saved, create a minimal one
            st.warning("No vocabulary found in checkpoint. Creating minimal vocabulary.")
            vocab = PersianVocabulary()
            max_length = 100
        
        # Get model parameters
        vocab_size = checkpoint.get('vocab_size', len(vocab))
        
        # Recreate model
        model = SentimentClassifier(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            output_dim=3,
            n_layers=2,
            dropout=0.1
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        st.success(f" Model loaded successfully from {pth_path}")
        st.success(f" Vocabulary size: {len(vocab)} words")
        
        return model, vocab, max_length, device, checkpoint
        
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        st.error(f"Checkpoint keys: {list(checkpoint.keys()) if 'checkpoint' in locals() else 'No checkpoint'}")
        return None, None, 100, device, None

# ==================== 4. Save Model with Vocabulary ====================
def save_model_with_vocab(model, vocab, model_path='best_sentiment_model_with_vocab.pth', max_length=100):
    """Save model with vocabulary to a .pth file"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_word2idx': vocab.word2idx,  # Save just the dictionary
        'vocab_size': len(vocab),
        'max_length': max_length,
        'model_config': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'output_dim': 3,
            'n_layers': 2,
            'dropout': 0.3
        }
    }
    
    torch.save(checkpoint, model_path)
    st.success(f"Model saved with vocabulary to {model_path}")
    return model_path

# ==================== 5. Prediction Functions ====================
class SentimentPredictor:
    def __init__(self, model, vocab, max_length, device):
        self.model = model
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        self.model.eval()
        
        # Label mapping (1=Good, 2=Neutral, 3=Bad)
        self.label_map = {0: 1, 1: 2, 2: 3}
        self.label_names = {
            1: "Good (Positive)",
            2: "Neutral", 
            3: "Bad (Negative)"
        }
    
    def predict(self, text, return_probabilities=False):
        """Predict sentiment for Persian text"""
        try:
            # Clean and numericalize text
            indices = self.vocab.numericalize(text, self.max_length)
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, prediction = torch.max(outputs, dim=1)
            
            # Convert to original labels (1,2,3)
            suggestion = self.label_map[prediction.item()]
            probs = probabilities[0].cpu().numpy()
            
            # Check if bad probability exceeds threshold
            is_bad_ml = probs[2] > THRESHOLD
            
            result = {
                'suggestion': suggestion,
                'probabilities': {
                    'good': float(probs[0]),
                    'neutral': float(probs[1]),
                    'bad': float(probs[2])
                },
                'is_bad_ml': is_bad_ml,
                'confidence': float(probs[2])
            }
            
            if return_probabilities:
                return suggestion, result
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# ==================== 6. OpenRouter Integration ====================
def get_openrouter_client():
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "sentiment-analysis-pipeline",
        },
    )

def check_with_openrouter(text, system_prompt=None):
    """Check if text is bad sentiment using OpenRouter"""
    if system_prompt is None:
        system_prompt = """You are a sentiment analysis assistant for Persian text. Analyze if the text expresses BAD sentiment (dissatisfaction, complaint, negative experience).
        
        Respond ONLY with this JSON format (no markdown, no extra text):
        {
            "is_bad": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation in English"
        }
        
        Consider:
        - Negative words (بد, ضعیف, خراب, etc.)
        - Complaints about quality/service
        - Expressions of disappointment
        - Recommendations against purchase"""
    
    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this Persian text: {text}"},
            ],
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        return {"is_bad": False, "confidence": 0.0, "reasoning": f"OpenRouter Error: {str(e)}"}

# ==================== 7. Visualization Functions ====================
def create_pipeline_diagram(step_status):
    """Create flowchart showing current pipeline status"""
    # Define nodes
    nodes = {
        'start': {'label': 'Start', 'x': 0, 'y': 2, 'color': '#4CAF50'},
        'ml': {'label': 'ML Model', 'x': 2, 'y': 2, 'color': '#2196F3'},
        'threshold': {'label': f'Check >{THRESHOLD:.0%}', 'x': 4, 'y': 2, 'color': '#FF9800'},
        'openrouter': {'label': 'OpenRouter Check', 'x': 6, 'y': 2, 'color': '#9C27B0'},
        'decision': {'label': 'Final Decision', 'x': 8, 'y': 2, 'color': '#607D8B'},
        'ok': {'label': '✅ OK', 'x': 8, 'y': 1, 'color': '#4CAF50'},
        'bad': {'label': '⚠️ BAD', 'x': 8, 'y': 3, 'color': '#F44336'}
    }
    
    # Prepare figure
    fig = go.Figure()
    
    # Add nodes
    for node_id, node_info in nodes.items():
        is_active = node_id in step_status.get('active_nodes', [])
        color = node_info['color'] if is_active else '#E0E0E0'
        
        fig.add_trace(go.Scatter(
            x=[node_info['x']],
            y=[node_info['y']],
            mode='markers+text',
            marker=dict(
                size=40,
                color=color,
                line=dict(width=3, color='white')
            ),
            text=[node_info['label']],
            textposition="middle center",
            textfont=dict(size=11, color='white'),
            name=node_info['label'],
            hoverinfo='text'
        ))
    
    # Add edges (connections)
    edges = [
        ('start', 'ml'),
        ('ml', 'threshold'),
        ('threshold', 'openrouter'),
        ('threshold', 'decision'),
        ('openrouter', 'decision'),
        ('decision', 'ok'),
        ('decision', 'bad')
    ]
    
    for start, end in edges:
        if start in nodes and end in nodes:
            start_node = nodes[start]
            end_node = nodes[end]
            
            # Check if this edge should be highlighted
            is_active = (start, end) in step_status.get('active_edges', [])
            line_color = 'green' if is_active else 'gray'
            line_width = 3 if is_active else 1
            
            fig.add_trace(go.Scatter(
                x=[start_node['x'], end_node['x']],
                y=[start_node['y'], end_node['y']],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add arrow for active edges
            if is_active:
                fig.add_annotation(
                    x=end_node['x'],
                    y=end_node['y'],
                    ax=start_node['x'],
                    ay=start_node['y'],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    text="",
                    showarrow=True,
                    arrowhead=3,
                    arrowwidth=2,
                    arrowcolor="green"
                )
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Sentiment Analysis Pipeline Flow",
            x=0.5,
            font=dict(size=20)
        ),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, 9]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 4]
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_sentiment_gauge(probabilities, current_suggestion):
    """Create gauge chart for sentiment probabilities"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Good', 'Neutral', 'Bad'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    sentiments = ['good', 'neutral', 'bad']
    colors = ['#4CAF50', '#FF9800', '#F44336']
    
    for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
        value = probabilities[sentiment] * 100
        
        # Determine if this is the current prediction
        current = False
        if sentiment == 'good' and current_suggestion == 1:
            current = True
        elif sentiment == 'neutral' and current_suggestion == 2:
            current = True
        elif sentiment == 'bad' and current_suggestion == 3:
            current = True
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title=dict(
                    text=f"{sentiment.upper()}",
                    font=dict(size=14, color='black' if not current else color)
                ),
                number=dict(suffix="%", font=dict(size=20)),
                domain={'row': 0, 'column': i},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': color, 'thickness': 0.8},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': color if current else "gray",
                    'steps': [
                        {'range': [0, 100], 'color': '#F5F5F5'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.8,
                        'value': THRESHOLD * 100
                    }
                }
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        title_text="Sentiment Probability Distribution",
        title_x=0.5
    )
    
    return fig

def create_timeline(steps):
    """Create timeline visualization of processing steps"""
    if not steps:
        return go.Figure()
    
    df = pd.DataFrame(steps)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['step'],
            y=df['duration'],
            text=df['status'],
            marker_color=df['color'],
            textposition='auto',
            textfont=dict(color='white', size=12)
        )
    ])
    
    fig.update_layout(
        title="Processing Timeline",
        xaxis_title="Step",
        yaxis_title="Duration (seconds)",
        showlegend=False,
        height=250,
        plot_bgcolor='white'
    )
    
    return fig

# ==================== 8. LangGraph Pipeline ====================
class PipelineState(TypedDict):
    text: str
    ml_result: Optional[dict]
    openrouter_result: Optional[dict]
    final_decision: Optional[str]
    steps_data: list
    pipeline_status: dict

def build_pipeline_graph(predictor):
    def ml_prediction_node(state: PipelineState) -> PipelineState:
        state['pipeline_status']['current_step'] = "Running ML Model..."
        state['pipeline_status']['active_nodes'].append('ml')
        state['pipeline_status']['active_edges'].append(('start', 'ml'))
        with st.spinner("Analyzing with ML model..."):
            start = time.time()
            result = predictor.predict(state['text'])
            elapsed = time.time() - start
            state['steps_data'].append({'step': 'ML Model', 'duration': round(elapsed, 3), 'status': 'Completed', 'color': '#2196F3'})
        state['ml_result'] = result
        return state

    def check_threshold_node(state: PipelineState) -> PipelineState:
        if state['ml_result'] is None:
            return state
        state['pipeline_status']['current_step'] = f"Checking threshold ({THRESHOLD:.0%})..."
        state['pipeline_status']['active_nodes'].append('threshold')
        state['pipeline_status']['active_edges'].append(('ml', 'threshold'))
        return state

    def route_after_threshold(state: PipelineState) -> Literal["openrouter_verify", "skip_openrouter", "error_end"]:
        if state['ml_result'] is None:
            return "error_end"
        return "openrouter_verify" if state['ml_result']['is_bad_ml'] else "skip_openrouter"

    def openrouter_verify_node(state: PipelineState) -> PipelineState:
        state['pipeline_status']['current_step'] = "Bad sentiment detected! Sending to OpenRouter..."
        state['pipeline_status']['active_nodes'].append('openrouter')
        state['pipeline_status']['active_edges'].append(('threshold', 'openrouter'))
        with st.spinner("Verifying with OpenRouter..."):
            start = time.time()
            result = check_with_openrouter(state['text'])
            elapsed = time.time() - start
            state['steps_data'].append({'step': 'OpenRouter Check', 'duration': round(elapsed, 3), 'status': 'Completed', 'color': '#9C27B0'})
        state['openrouter_result'] = result
        return state

    def final_decision_node(state: PipelineState) -> PipelineState:
        state['pipeline_status']['active_nodes'].append('decision')
        state['pipeline_status']['active_edges'].append(('openrouter', 'decision'))
        confirms = (state['openrouter_result'].get('is_bad', False) and
                    state['openrouter_result'].get('confidence', 0) > CONFIDENCE_THRESHOLD)
        if confirms:
            state['final_decision'] = "BAD"
            state['pipeline_status']['active_nodes'].append('bad')
            state['pipeline_status']['active_edges'].append(('decision', 'bad'))
        else:
            state['final_decision'] = "OK"
            state['pipeline_status']['active_nodes'].append('ok')
            state['pipeline_status']['active_edges'].append(('decision', 'ok'))
        state['pipeline_status']['current_step'] = f"Analysis complete: {state['final_decision']}"
        return state

    def skip_openrouter_node(state: PipelineState) -> PipelineState:
        state['pipeline_status']['current_step'] = "Not bad sentiment - skipping OpenRouter"
        state['openrouter_result'] = None
        state['final_decision'] = "OK"
        state['pipeline_status']['active_nodes'].append('decision')
        state['pipeline_status']['active_edges'].append(('threshold', 'decision'))
        state['pipeline_status']['active_nodes'].append('ok')
        state['pipeline_status']['active_edges'].append(('decision', 'ok'))
        state['pipeline_status']['current_step'] = "Analysis complete: OK"
        return state

    def error_end_node(state: PipelineState) -> PipelineState:
        state['openrouter_result'] = None
        state['final_decision'] = "OK"
        return state

    builder = StateGraph(PipelineState)
    builder.add_node("ml_prediction", ml_prediction_node)
    builder.add_node("check_threshold", check_threshold_node)
    builder.add_node("openrouter_verify", openrouter_verify_node)
    builder.add_node("final_decision", final_decision_node)
    builder.add_node("skip_openrouter", skip_openrouter_node)
    builder.add_node("error_end", error_end_node)
    builder.set_entry_point("ml_prediction")
    builder.add_edge("ml_prediction", "check_threshold")
    builder.add_conditional_edges(
        "check_threshold",
        route_after_threshold,
        {"openrouter_verify": "openrouter_verify", "skip_openrouter": "skip_openrouter", "error_end": "error_end"}
    )
    builder.add_edge("openrouter_verify", "final_decision")
    builder.add_edge("skip_openrouter", END)
    builder.add_edge("final_decision", END)
    builder.add_edge("error_end", END)
    return builder.compile()

def run_analysis_pipeline(text, predictor):
    graph = build_pipeline_graph(predictor)
    initial = PipelineState(
        text=text,
        ml_result=None,
        openrouter_result=None,
        final_decision=None,
        steps_data=[],
        pipeline_status={'active_nodes': ['start'], 'active_edges': [], 'current_step': 'Starting analysis...'}
    )
    result = graph.invoke(initial)
    if result['ml_result'] is None:
        return None
    return {
        'text': result['text'],
        'ml_result': result['ml_result'],
        'openrouter_result': result['openrouter_result'],
        'final_decision': result['final_decision'],
        'steps_data': result['steps_data'],
        'pipeline_status': result['pipeline_status']
    }

# ==================== 9. Streamlit UI ====================
def main():
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    if 'current_status' not in st.session_state:
        st.session_state['current_status'] = "Ready"
    
    # App title
    st.title("Sentiment Analysis Pipeline")
    st.markdown("""
    This app analyzes Persian text sentiment using a two-step pipeline:
    1. **ML Model**: Predicts sentiment (1=Good, 2=Neutral, 3=Bad)
    2. **OpenRouter Verification**: Double-checks bad sentiments using a free LLM
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Threshold controls
        global THRESHOLD, CONFIDENCE_THRESHOLD
        THRESHOLD = st.slider(
            "Bad Sentiment Threshold",
            0.0, 1.0, 0.1, 0.01,
            help="ML model probability threshold for 'bad' classification"
        )
        
        CONFIDENCE_THRESHOLD = st.slider(
            "OpenRouter Confidence Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Minimum confidence for OpenRouter to confirm bad sentiment"
        )
        
        st.divider()
        
        # Statistics
        st.header("Statistics")
        if st.session_state['analysis_history']:
            total = len(st.session_state['analysis_history'])
            bad_count = sum(1 for r in st.session_state['analysis_history'] if r['final_decision'] == "BAD")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total)
            with col2:
                st.metric("Bad", bad_count)
            
            st.metric("Bad Rate", f"{(bad_count/total*100):.1f}%" if total > 0 else "0%")
        else:
            st.info("No analyses yet")
        
        st.divider()
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state['analysis_history'] = []
            st.rerun()
    
    # Load model
    model, vocab, max_length, device, checkpoint = load_model_and_vocab()
    
    if model is None or vocab is None:
        st.error("""
        Could not load model. Please ensure:
        1. You have `best_sentiment_model.pth` in the current directory
        2. The .pth file contains the vocabulary (word2idx dictionary)
        
        If your .pth file doesn't contain vocabulary, you can:
        """)
        
        # Option to create a new model with vocabulary
        if st.button("Create New Model with Vocabulary"):
            # Create a minimal vocabulary
            vocab = PersianVocabulary()
            # Add some common Persian words
            common_words = ["خوب", "بد", "متوسط", "عالی", "ضعیف", "مثبت", "منفی"]
            for i, word in enumerate(common_words, start=4):
                vocab.word2idx[word] = i
                vocab.idx2word[i] = word
            
            # Create a simple model
            model = SentimentClassifier(
                vocab_size=len(vocab),
                embedding_dim=128,
                hidden_dim=256,
                output_dim=3,
                n_layers=2,
                dropout=0.3
            )
            
            # Save with vocabulary
            model_path = save_model_with_vocab(model, vocab)
            st.success(f"Created new model at {model_path}")
            st.rerun()
        
        return
    
    # Create predictor
    predictor = SentimentPredictor(model, vocab, max_length, device)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input section
        st.subheader("Enter Persian Text")
        
        # Quick example buttons
        examples = st.columns(3)
        example_texts = {
            "Good": "این محصول واقعا عالی بود. کیفیت فوق العاده",
            "Neutral": "محصول متوسطی بود، نه خوب نه بد",
            "Bad": "بدترین خرابیم بود، پولم را هدر دادم"
        }
        
        selected_example = None
        for i, (sentiment, text) in enumerate(example_texts.items()):
            with examples[i]:
                if st.button(f"{sentiment} Example", use_container_width=True):
                    selected_example = text
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="مثال: محصول بدی بود، کیفیت پایینی داشت...",
            key="text_input"
        )
        
        # Analyze button
        analyze_clicked = st.button(
            "Analyze Sentiment",
            type="primary",
            disabled=not text_input.strip(),
            use_container_width=True
        )
    
    with col2:
        # Status panel
        st.subheader("Status")
        current_step = st.session_state.get('current_status', 'Ready')
        
        # Progress mapping
        progress_map = {
            "Ready": 0,
            "Starting analysis...": 10,
            "Running ML Model...": 30,
            f"Checking threshold ({THRESHOLD:.0%})...": 50,
            "Bad sentiment detected! Sending to OpenRouter...": 70,
            "Not bad sentiment - skipping OpenRouter": 80,
            "Analysis complete: OK": 100,
            "Analysis complete: BAD": 100
        }
        
        progress = progress_map.get(current_step, 0)
        
        # Status display
        status_color = {
            "Ready": "blue",
            "Starting analysis...": "blue",
            "Running ML Model...": "blue",
            f"Checking threshold ({THRESHOLD:.0%})...": "orange",
            "Bad sentiment detected! Sending to OpenRouter...": "red",
            "Not bad sentiment - skipping OpenRouter": "green",
            "Analysis complete: OK": "green",
            "Analysis complete: BAD": "red"
        }.get(current_step, "gray")
        
        st.markdown(f"""
        <div style="
            background-color: {status_color}20;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid {status_color};
            margin-bottom: 15px;
        ">
            <h4 style="margin: 0; color: {status_color};">{current_step}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(progress / 100)
        st.caption(f"Progress: {progress}%")
    
    # Process analysis
    if analyze_clicked and text_input.strip():
        # Reset status
        st.session_state['current_status'] = "Starting analysis..."
        
        # Create containers for results
        pipeline_container = st.container()
        results_container = st.container()
        details_container = st.container()
        
        # Show initial pipeline diagram
        with pipeline_container:
            st.subheader("Pipeline Flow")
            initial_status = {
                'active_nodes': ['start'],
                'active_edges': []
            }
            fig = create_pipeline_diagram(initial_status)
            pipeline_placeholder = st.empty()
            pipeline_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Run the pipeline
        result = run_analysis_pipeline(text_input, predictor)
        
        if result:
            # Update session state
            st.session_state['current_status'] = result['pipeline_status']['current_step']
            st.session_state['analysis_history'].append(result)
            
            # Update pipeline diagram
            with pipeline_container:
                fig = create_pipeline_diagram(result['pipeline_status'])
                pipeline_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Show results
            with results_container:
                st.subheader("Analysis Results")
                
                # Result cards
                cols = st.columns(4)
                
                with cols[0]:
                    suggestion = result['ml_result']['suggestion']
                    sentiment = "Good" if suggestion == 1 else "Neutral" if suggestion == 2 else "Bad"
                    st.metric("ML Prediction", f"{sentiment} ({suggestion})")
                
                with cols[1]:
                    bad_prob = result['ml_result']['probabilities']['bad']
                    st.metric("Bad Probability", f"{bad_prob:.1%}")
                
                with cols[2]:
                    if result['openrouter_result']:
                        conf = result['openrouter_result'].get('confidence', 0)
                        st.metric("OpenRouter Confidence", f"{conf:.1%}")
                    else:
                        st.metric("OpenRouter Check", "Skipped")
                
                with cols[3]:
                    decision = result['final_decision']
                    color = "red" if decision == "BAD" else "green"
                    icon = "⚠️" if decision == "BAD" else "✅"
                    st.markdown(f"""
                    <div style="
                        background-color: {color}20;
                        padding: 10px;
                        border-radius: 5px;
                        border-left: 5px solid {color};
                    ">
                        <h3 style="margin: 0; color: {color};">Final Decision</h3>
                        <h1 style="margin: 0;">{icon} {decision}</h1>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show detailed analysis
            with details_container:
                tab1, tab2, tab3 = st.tabs(["Probabilities", "OpenRouter Details", "Timeline"])
                
                with tab1:
                    # Sentiment gauges
                    fig = create_sentiment_gauge(
                        result['ml_result']['probabilities'],
                        result['ml_result']['suggestion']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability table
                    prob_data = {
                        'Sentiment': ['Good (1)', 'Neutral (2)', 'Bad (3)'],
                        'Probability': [
                            f"{result['ml_result']['probabilities']['good']:.2%}",
                            f"{result['ml_result']['probabilities']['neutral']:.2%}",
                            f"{result['ml_result']['probabilities']['bad']:.2%}"
                        ],
                        'Threshold Check': [
                            'N/A',
                            'N/A',
                            '✓ Pass' if result['ml_result']['probabilities']['bad'] > THRESHOLD else '✗ Fail'
                        ]
                    }
                    st.dataframe(pd.DataFrame(prob_data), use_container_width=True)
                
                with tab2:
                    if result['openrouter_result']:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.info("**OpenRouter Response**")
                            st.json(result['openrouter_result'])
                        
                        with col2:
                            st.info("**Interpretation**")
                            
                            is_bad = result['openrouter_result'].get('is_bad', False)
                            confidence = result['openrouter_result'].get('confidence', 0)
                            
                            if is_bad and confidence > CONFIDENCE_THRESHOLD:
                                st.error("### ✅ OpenRouter Confirms: BAD SENTIMENT")
                                st.write(f"**Confidence**: {confidence:.1%} (>{CONFIDENCE_THRESHOLD:.0%})")
                            elif is_bad:
                                st.warning("### ⚠️ OpenRouter Detects Bad but Low Confidence")
                                st.write(f"**Confidence**: {confidence:.1%} (<{CONFIDENCE_THRESHOLD:.0%})")
                            else:
                                st.success("### ✅ OpenRouter Says: NOT BAD")
                            
                            st.write(f"**Reasoning**: {result['openrouter_result'].get('reasoning', 'No reasoning provided')}")
                    else:
                        st.info("OpenRouter verification was skipped because the ML model did not detect bad sentiment above the threshold.")
                
                with tab3:
                    if result['steps_data']:
                        fig = create_timeline(result['steps_data'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Steps table
                        steps_df = pd.DataFrame(result['steps_data'])
                        st.dataframe(steps_df, use_container_width=True)
            
            # Show history
            with st.expander("Analysis History (Last 10)"):
                if st.session_state['analysis_history']:
                    history_data = []
                    for i, res in enumerate(st.session_state['analysis_history'][-10:], 1):
                        history_data.append({
                            '#': i,
                            'Text': res['text'][:50] + '...' if len(res['text']) > 50 else res['text'],
                            'ML': f"Suggestion {res['ml_result']['suggestion']}",
                            'Bad Prob': f"{res['ml_result']['probabilities']['bad']:.1%}",
                            'Decision': res['final_decision']
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No history yet")
    
    # Check OpenRouter status
    with st.sidebar:
        st.divider()
        try:
            client = get_openrouter_client()
            response = client.models.list()
            model_ids = [m.id for m in response]
            if OPENROUTER_MODEL in model_ids or any(OPENROUTER_MODEL.split(':')[0] in m for m in model_ids):
                st.success("✅ OpenRouter is connected")
            else:
                st.warning(f"⚠️ {OPENROUTER_MODEL} availability unknown")
        except:
            st.error("OpenRouter connection failed")
            
        st.caption(f"Model: {OPENROUTER_MODEL}")
        st.caption(f"Bad threshold: >{THRESHOLD:.0%}")
        st.caption(f"OpenRouter confidence: >{CONFIDENCE_THRESHOLD:.0%}")
        
        # Model info
        st.divider()
        st.header("Model Info")
        if checkpoint:
            st.caption(f"Vocabulary size: {len(vocab)}")
            st.caption(f"Max length: {max_length}")
            if 'val_accuracy' in checkpoint:
                st.caption(f"Validation accuracy: {checkpoint['val_accuracy']:.2%}")

# ==================== 10. Run the App ====================
if __name__ == "__main__":
    main()
