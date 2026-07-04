# Persian Sentiment Analysis Pipeline — LangChain + OpenRouter

A Streamlit app for Persian text sentiment analysis using a PyTorch BiLSTM model with LLM verification via OpenRouter, orchestrated as a LangGraph state machine.

## Architecture

The analysis pipeline is orchestrated with **LangGraph** as a state machine:

```
Start → ML Model → Threshold Check → OpenRouter Verify → Final Decision
                                   ↘ (skip) → OK
```

- **ML Model** — Bidirectional LSTM with attention trained on Digikala Persian e-commerce reviews (~73% accuracy)
- **Threshold Check** — If bad-sentiment probability > threshold (default 0.1), the text is sent to OpenRouter
- **OpenRouter** — `google/gemma-4-31b-it:free` via OpenRouter API verifies bad sentiment with a structured JSON response
- **Final Decision** — Combines ML + OpenRouter signals; confirmed bad if OpenRouter confidence > 0.6

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `best_sentiment_model.pth` | Trained PyTorch model checkpoint (~1.1 MB) |
| `digikala_sentiment.csv` | Training dataset (3262 Persian reviews) |
| `vocab_dict.json` | Tokenizer vocabulary |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install streamlit openai torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu plotly pandas numpy langchain langchain-openai langgraph
```

## Run

```bash
streamlit run app.py
```

## Configuration

Edit the constants at the top of `app.py`:

- `OPENROUTER_API_KEY` — Your OpenRouter API key
- `OPENROUTER_MODEL` — Model slug (uses `google/gemma-4-31b-it:free`)
- `THRESHOLD` — ML bad-probability threshold
- `CONFIDENCE_THRESHOLD` — OpenRouter minimum confidence to confirm bad sentiment

Thresholds can also be adjusted via the Streamlit sidebar sliders at runtime.
